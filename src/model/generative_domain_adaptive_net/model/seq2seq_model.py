"""
Simple seq2seq architecture for testing.
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell as GRU
import time
import os
import logging
from tqdm import trange
from model.layers import encoder_layer,\
                         bidirectional_encoder_layer, \
                         decoder_layer,\
                         crossentropy
from utils.Helpers import calculate_accuracies,\
                          SYMB_BEGIN,\
                          SYMB_END
from nltk.tokenize.moses import MosesDetokenizer

# Maximum word length, used for limiting word lengths if using character model
MAX_WORD_LEN = 10


class Seq2Seq:
    def __init__(self, n_layers, dictionaries, vocab_size, n_hidden_encoder,
                 n_hidden_decoder, embed_dim, train_emb, answer_injection, batch_size, use_bi_encoder,
                 use_attention):
        # Input variables
        self.n_hidden_encoder = n_hidden_encoder  # Number of hidden units in encoder
        self.n_hidden_decoder = n_hidden_decoder  # Number of hidden units in decoder
        self.n_layers = n_layers  # The number of layers in encoder and decoder
        self.embed_dim = embed_dim  # The size of the initial embedding vectors (e.g. GloVe)
        self.train_emb = train_emb              # Bool: train embeddings or not
        self.batch_size = batch_size            # Max. batch size

        self.word_dictionary = dictionaries[0]  # The word dictionary, used in accuracy
        self.vocab_size = vocab_size            # Size of the word vocabulary (unique word tokens)
        self.symbol_begin = self.word_dictionary[SYMB_BEGIN]  # Integer of start of sequence mark
        self.symbol_end = self.word_dictionary[SYMB_END]  # Integer of start of sequence mark
        self.answer_injection = answer_injection
        self.use_bi_encoder = use_bi_encoder
        self.use_attention = use_attention

        # If a bidirectional encoder is used, then make sure the decoder has twice the units
        # to match the concatenated encoder state size (which is 2 * n_hidden_encoder)
        if self.use_bi_encoder:
            self.n_hidden_decoder *= 2

        # Graph initialization
        # See their explanation below in the build_graph() method
        self.document = None
        self.query = None
        self.answer = None
        self.answer_mask = None
        self.document_mask = None
        self.query_mask = None
        self.learning_rate = None
        self.keep_prob = None
        self.prediction = None
        self.test = None
        self.updates = None
        # Accuracy and Loss measures
        self.loss = None                    # The categorical cross-entropy loss
        self.EM_accuracy = None                # The exact match (EM) accuracy

        # Tensorboard variables
        # Used to report accuracy and loss values to tensorboard during training/validation
        # Exact Match
        self.em_acc_metric = None
        self.em_acc_metric_update = None
        self.em_valid_acc_metric = None
        self.em_valid_acc_metric_update = None

        self.loss_summ = None
        self.em_acc_summ = None
        self.em_valid_acc_summ = None
        self.merged_summary = None

    def build_graph(self, grad_clip, embed_init, seed, max_doc_len, max_qry_len):
        # ================================================================================================
        #                                   DEFINING GRAPH PLACEHOLDERS
        # ================================================================================================

        # Placeholder for integer representations of the document and query tokens.
        # These are tensors of shape [batch_size, max_length] where max_length is the length of the longest
        # document or query in the current batch.
        self.document = tf.placeholder(tf.int32, [None, max_doc_len], name="document")  # Document words
        self.query = tf.placeholder(tf.int32, [None, max_qry_len], name="query")  # Query words
        # Placeholder for the ground truth answer's index in the document.
        # A tensor of shape [batch_size, 2]
        # The values refer to the answer's index in the document. Can be either the index among
        # tokens or chars.
        #
        # [[answer_start_0, answer_end_0]
        #  [answer_start_1, answer_end_1]
        #  [............................]
        #  [answer_start_n, answer_end_n]] - where batch_size = n
        #
        self.answer = tf.placeholder(
            tf.int32, [None, 2], name="answer")

        # Placeholder for document and query masks.
        # These are the same as the document and query placeholders above, except that they are binary,
        # having 0's where there is no token, and 1 where there is.
        # Example:
        # Assuming max_doc_len = 4 and batch_size = 3
        #                   <---4---->                                <---4---->
        # self.document = [[2, 5, 4, 7]  ----> self.document_mask = [[1, 1, 1, 1]  <-- document 1
        #                  [3, 2, 6, 0]                              [1, 1, 1, 0]  <-- document 2
        #                  [2, 1, 0, 0]]                             [1, 1, 0, 0]] <-- document 3
        #
        # The masks are used to calculate the sequence length of each text sample going into
        # the bi-directional RNN.
        self.document_mask = tf.placeholder(
            tf.int32, [None, max_doc_len], name="document_mask")
        self.query_mask = tf.placeholder(
            tf.int32, [None, max_qry_len], name="query_mask")
        self.answer_mask = tf.placeholder(
            tf.float32, [None, max_doc_len], name="answer_mask")

        # Model parameters
        # Initial learning rate
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        # Keep probability = 1 - dropout probability
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # =================================================================================================
        #                                BUILDING THE GRAPH
        # =================================================================================================

        # Embedding the document and query words.
        # For each word (represented by an integer) we look up the vector representation in either
        # a pre-trained word-vector dictionary or one that is initialized now.
        # See figure 1. in the paper (link at the top of this file). These embeddings are the leftmost
        # horizontal arrows in the figure going into the boxes named 'Embed'. Documents are blue, while the query
        # is green in the figure.
        # By default the word-vectors used are GloVe (Global Vectors) see the paper:
        # "GloVe: Global Vectors for Word Representation" by Pennington et al. 2014
        # Link to code, demo, paper: https://nlp.stanford.edu/projects/glove/
        #
        # This process means the documents go from 2 to 3 dimensional tensors:
        # before: [batch_size, max_document_length] -->
        # after: [batch_size, max_document_length, embedding_dimensions]
        #
        # The word-vectors are shaped [vocabulary_size, embedding_dimension]
        # so there is a word-vector for each unique word in the vocabulary
        #
        # EXAMPLE, assuming that:
        # batch_size = 2, max_document_length = 3, embedding_dim = 2, vocabulary_size = 3
        #                 <- 2->                          <---3--->
        # word_vectors = [[1, 2]  <- word 1   document = [[0, 1, 0]  <- document 1
        #                 [3, 4]  <- word 2               [1, 2, 0]] <- document 2
        #                 [5, 6]] <- word 3
        #
        # Then the document embeddings will be:
        #                   <----------3----------->
        #                    <- 2->  <- 2->  <- 2->
        # document_embed = [[[1, 2], [3, 4], [1, 2]]  <- document 1
        #                   [[3, 4], [5, 6], [1, 2]]] <- document 2
        #

        # Creating the variable for the word_vectors
        if embed_init is None:  # If there are no pre-trained word vectors
            word_vectors = tf.get_variable(
                "word_vectors", [self.vocab_size, self.embed_dim],
                initializer=tf.glorot_normal_initializer(seed, tf.float32),
                trainable=self.train_emb)
        else:  # Else, we use the pre-trained word-vectors
            word_vectors = tf.Variable(embed_init, trainable=self.train_emb,
                                       name="word_vectors")

        # Embedding the document and query in the above word_vectors
        document_embedding = tf.nn.embedding_lookup(
            word_vectors, self.document, name="document_embedding")
        query_embedding = tf.nn.embedding_lookup(
            word_vectors, self.query, name="query_embedding")

        # Assert embedding shapes are [None, max_length, embedding_dimensions]
        assert document_embedding.shape.as_list() == [None, max_doc_len, self.embed_dim],\
            "Expected document embedding shape [None, {}, {}] but got {}".format(
                max_doc_len, self.embed_dim, document_embedding.shape)
        assert query_embedding.shape.as_list() == [None, max_qry_len, self.embed_dim],\
            "Expected document embedding shape [None, {}, {}] but got {}".format(
                max_doc_len, self.embed_dim, query_embedding.shape)

        # Concatenating the answer mask with the last layer's document embedding input
        if self.answer_injection:
            answer_mask_expanded = tf.expand_dims(self.answer_mask, axis=2)
            document_embedding = tf.concat([document_embedding, answer_mask_expanded], axis=2)

        # Assert document embedding (after answer_mask concatenation) shapes are:
        # [None, max_length, embedding_dimensions]
        assert document_embedding.shape.as_list() == [None, max_doc_len, self.embed_dim+1], \
            "Expected document embedding shape [None, {}, {}] but got {}".format(
                max_doc_len, self.embed_dim+1, document_embedding.shape)

        # -----------------------------------------
        #               Encoder Layer
        # -----------------------------------------

        # Pass the document to the encoder layer (either bidirectional or unidirectional)
        if self.use_bi_encoder:
            encoder_output, encoder_states = \
                bidirectional_encoder_layer(GRU, self.n_layers, self.document_mask, document_embedding,
                                            self.n_hidden_encoder, max_doc_len, self.keep_prob)
        else:
            encoder_output, encoder_states = \
                encoder_layer(GRU, self.n_layers, self.document_mask, document_embedding,
                              self.n_hidden_encoder, max_doc_len, self.keep_prob)

        current_batch_size = tf.to_int32(tf.shape(self.query)[0])

        logits_training, logits_inference = decoder_layer(encoder_states, encoder_output,
                                                          query_embedding, self.query_mask,
                                                          self.document_mask, word_vectors, GRU,
                                                          max_qry_len, self.vocab_size,
                                                          self.n_layers, self.n_hidden_decoder,
                                                          self.keep_prob, self.use_attention,
                                                          self.symbol_begin, self.symbol_end,
                                                          current_batch_size)

        # Getting the output from the decoder layer
        logits_training = tf.identity(logits_training.rnn_output, name="logits")
        sample_ids_inference = tf.identity(logits_inference.sample_id, name="predictions")
        self.prediction = sample_ids_inference

        # -----------
        #    LOSS
        # -----------
        # Cast and query mask as float32 for loss calculation
        query_mask_float = tf.cast(self.query_mask, dtype=tf.float32)

        # Slice query and query_mask to match first two dimensions with logits
        # Get the current size of logits' second dimension and the current batch size
        logits_shape_1 = tf.to_int32(tf.shape(logits_training)[1])
        # batch_size = tf.to_int32(tf.shape(query_mask_float)[0])
        # Perform the slice
        query_mask_sliced = tf.slice(query_mask_float, [0, 0], [current_batch_size, logits_shape_1])
        query_sliced = tf.slice(self.query, [0, 0], [current_batch_size, logits_shape_1])

        self.loss = tf.contrib.seq2seq.sequence_loss(logits_training,
                                                     query_sliced,
                                                     query_mask_sliced)

        # -------------------
        #      ACCURACY
        # -------------------

        # TODO: code this up, probably some sort of f1 score between generated query and actual

        # Define Optimizer
        vars_list = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_list), grad_clip)
        self.updates = optimizer.apply_gradients(zip(grads, vars_list))
        # Save variables
        self.save_vars()

        # Tensorboard summaries
        # TODO: keep this for now, will check later if it can stay for seq2seq
        # self.em_acc_summ = tf.summary.scalar('exact_match_accuracy', self.em_acc_metric_update)
        # self.loss_summ = tf.summary.scalar('cross-entropy_loss', self.loss)
        # self.merged_summary = tf.summary.merge_all()
        # self.em_valid_acc_summ = tf.summary.scalar('em_valid_acc_metric',
        #                                            self.em_valid_acc_metric_update)

    def save_vars(self):
        """
        for restoring model
        """
        tf.add_to_collection('document', self.document)
        tf.add_to_collection('answer', self.answer)
        tf.add_to_collection('document_mask', self.document_mask)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('accuracy', self.EM_accuracy)
        tf.add_to_collection('updates', self.updates)
        tf.add_to_collection('learning_rate', self.learning_rate)

    def train(self, sess, training_data, dropout, learning_rate, iteration, writer, epoch, max_it):
        """
        Performs one training iteration with input tuple of training data.
        """
        document_array, document_character_array, query_array, query_character_array,\
            answer_array, document_mask_array, query_mask_array, answer_mask_array,\
            type_character_array, type_character_mask, filenames = training_data

        feed_dict = {self.document: document_array, self.query: query_array,
                     self.answer: answer_array, self.document_mask: document_mask_array,
                     self.query_mask: query_mask_array, self.keep_prob: 1 - dropout,
                     self.learning_rate: learning_rate}

        # Feature marking the answer words in the document
        if self.answer_injection:
            feed_dict[self.answer_mask] = answer_mask_array

        if iteration % 50 == 0:  # Get updated summary for Tensorboard every Xth iteration
            loss, updates = \
                sess.run([self.loss, self.updates], feed_dict)

            # writer.add_summary(merged_summ, (epoch * max_it + iteration))
        else:  # Otherwise, get regular updates
            loss, updates = \
                sess.run([self.loss, self.updates], feed_dict)

        # Calculate F1 Score and Exact Match accuracy over the batch
        # f1_score, exact_match_accuracy = calculate_accuracies(answer_array, predicted_answer_array,
        #                                                       document_array, self.word_dictionary)

        # return loss, f1_score, exact_match_accuracy, updates
        return loss, updates

    def validate(self, sess, valid_batch_loader,
                 inverse_word_dictionary,
                 iteration=None, writer=None,
                 epoch=None, max_it=None):
        """
        Validate/Test the model
        """
        it = loss = em_accuracy = f1_score = n_example = 0

        # Text predictions for inference
        prediction_text = []

        tr = trange(
            len(valid_batch_loader),
            desc="loss: {:.3f}, EM_accuracy: {:.3f}, F1_accuracy: {:.3f}".format(0.0, 0.0, 0.0),
            leave=False,
            ascii=True)
        start_time = time.time()
        for validation_data in valid_batch_loader:
            it += 1
            document_array, document_character_array, query_array, query_character_array,\
                answer_array, document_mask_array, query_mask_array, answer_mask_array,\
                type_character_array, type_character_mask, filenames = validation_data

            feed_dict = {self.document: document_array, self.query: query_array,
                         self.answer: answer_array, self.document_mask: document_mask_array,
                         self.query_mask: query_mask_array, self.keep_prob: 1.,
                         self.learning_rate: 0.}

            # Feature marking the answer words in the document
            if self.answer_injection:
                feed_dict[self.answer_mask] = answer_mask_array

            loss_, prediction = \
                sess.run([self.loss, self.prediction], feed_dict)

            # Run current document, query and generated query through inverse word-dictionary
            # for printing at end of validation
            current_prediction = [[inverse_word_dictionary[word_value] for word_value in row]
                                  for row in prediction]
            current_document = [[inverse_word_dictionary[word_value] for word_value in row]
                                for row in document_array]
            current_query = [[inverse_word_dictionary[word_value] for word_value in row]
                             for row in query_array]

            prediction_text.append([current_document, current_query, current_prediction])

            # Calculate F1 Score and Exact Match accuracy over the batch
            # f1_score_, exact_match_accuracy_ = \
            #     calculate_accuracies(answer_array, predicted_answer_array,
            #                          document_array, self.word_dictionary)

            n_example += document_array.shape[0]

            loss += loss_
            # em_accuracy += exact_match_accuracy_
            # f1_score += f1_score_
            # tr.set_description("loss: {:.3f}, EM_accuracy: {:.3f}, F1_Score: {:.3f}".
            #                    format(loss_, exact_match_accuracy_, f1_score_))
            tr.update()

        tr.close()
        # if writer is not None:
        #     writer.add_summary(em_valid_acc_summary, (epoch * max_it + iteration))

        loss /= n_example
        em_accuracy /= it
        f1_score /= it
        time_spent = (time.time() - start_time) / 60
        statement = "loss: {:.3f}, EM_Acc: {:.3f}, F1_Score: {:.3f}, time: {:.1f}(m)" \
            .format(loss, em_accuracy, f1_score, time_spent)
        logging.info(statement)
        # Logging example document, ground truth question and generated question
        detokenizer = MosesDetokenizer()

        for i in range(len(prediction_text)):
            doc = prediction_text[i][0][0]
            doc = [word for word in doc if word != "@pad"]
            qry = prediction_text[i][1][0]
            qry = [word for word in qry if word != "@pad"]
            gen_qry = prediction_text[i][2][0]
            doc = detokenizer.detokenize(doc, return_str=True)
            qry = detokenizer.detokenize(qry, return_str=True)
            gen_qry = detokenizer.detokenize(gen_qry, return_str=True)

            logging.info("\n Document: {}\n".format(doc))
            logging.info("\n Query: {}".format(qry))
            logging.info("\n Generated query: {}".format(gen_qry))

        return loss, em_accuracy, f1_score

    def predict(self, sess, batch_loader):

        output = []
        for samples in batch_loader:
            document_array, document_character_array, query_array, query_character_array,\
                answer_array, document_mask_array, query_mask_array, answer_mask_array,\
                type_character_array, type_character_mask, filenames = samples

            feed_dict = {self.document: document_array, self.query: query_array,
                         self.document_char: document_character_array,
                         self.query_char: query_character_array,
                         self.answer: answer_array, self.document_mask: document_mask_array,
                         self.query_mask: query_mask_array, self.token: type_character_array,
                         self.char_mask: type_character_mask, self.keep_prob: 1.,
                         self.learning_rate: 0.}

            # Feature marking the answer words in the document
            if self.answer_injection:
                feed_dict[self.answer_mask] = answer_mask_array

            document, query, answer,\
                predicted_answer, answer_start_probabilities,\
                answer_end_probabilities, attention_tensors = \
                sess.run([self.document, self.query, self.answer,
                          self.predicted_answer, self.start_probabilities,
                          self.end_probabilities, self.attention_tensors], feed_dict)
            output.append((document, query, answer,
                           predicted_answer, answer_start_probabilities,
                           answer_end_probabilities, attention_tensors))

        return output

    def restore(self, sess, checkpoint_dir, epoch):
        """
        restore model
        """
        checkpoint_path = os.path.join(checkpoint_dir,
                                       'model_epoch{}.ckpt'.format(epoch))

        print("\nRestoring model from: {}\n".format(checkpoint_path))

        loader = tf.train.import_meta_graph(checkpoint_path + '.meta')
        loader.restore(sess, checkpoint_path)
        logging.info("model restored from {}".format(checkpoint_path))
        # restore variables from checkpoint
        self.document = tf.get_collection('document')[0]
        self.answer = tf.get_collection('answer')[0]
        self.document_mask = tf.get_collection('document_mask')[0]
        self.query_mask = tf.get_collection('query_mask')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]
        self.loss = tf.get_collection('loss')[0]
        self.EM_accuracy = tf.get_collection('accuracy')[0]
        self.updates = tf.get_collection('updates')[0]
        self.learning_rate = tf.get_collection('learning_rate')[0]

    def save(self, sess, saver, checkpoint_dir, epoch):
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch{}.ckpt'.format(epoch))
        saver.save(sess, checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))