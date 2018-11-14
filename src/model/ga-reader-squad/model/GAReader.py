"""
Gated-Attention Reader model, implemented according to the paper:
"Gated-Attention Readers for Text Comprehension" by Bhuwan Dhingra/Hanxiao Liu et al.
    arXiv:  1606.01549v3
    github: https://github.com/bdhingra/ga-reader
For the sake of brevity, unless otherwise stated all references saying "see paper" below refer to this paper.
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell as GRU
import time
import os
import logging
from tqdm import trange
from model.layers import gated_attention,\
                               pairwise_interaction,\
                               attention_sum,\
                               crossentropy
from utils.Helpers import prepare_input

# Maximum word length, used for limiting word lengths if using character model
MAX_WORD_LEN = 10


class GAReader:
    def __init__(self, n_layers, vocab_size, vocab_size_char,
                 n_hidden, n_hidden_dense, embed_dim, train_emb, n_hidden_char,
                 use_feat, gating_fn, save_attn=False):
        # Input variables
        self.n_hidden = n_hidden                # The number of hidden units in the GRU cell
        self.n_hidden_dense = n_hidden_dense    # The number of hidden units in the final dense layer
        self.n_layers = n_layers                # The number of layers (or 'hops', see paper fig. 1) in the model
        self.embed_dim = embed_dim              # The size of the initial embedding vectors (e.g. GloVe)
        self.train_emb = train_emb              # Bool: train embeddings or not
        self.use_qe_comm_feature = use_feat                # Bool: use qe-comm feature or not (see paper, section 3.1.4)
        # The "activation" function at the end of the gated attention layer
        # Default is tf.multiply()
        self.gating_fn = gating_fn
        self.save_attn = save_attn              # Bool: save attention matrices during forward pass or not
        self.vocab_size = vocab_size            # Size of the word vocabulary (unique word tokens)

        # Input (only for character model)
        self.n_hidden_char = n_hidden_char      # The number of hidden units in the character GRU cell
        self.vocab_size_char = vocab_size_char  # Number of different characters in vocabulary
        self.use_chars = self.n_hidden_char != 0  # Bool: Whether or not to train a character model

        # Graph initialization
        # See their explanation below in the build_graph() method
        self.doc = None
        self.qry = None
        self.answer = None
        self.doc_mask = None
        self.qry_mask = None
        self.doc_char = None
        self.qry_char = None
        self.token = None
        self.char_mask = None
        self.feat = None
        self.learning_rate = None
        self.keep_prob = None
        self.attentions = None
        self.attention_tensors = None
        self.pred = None
        self.start_probs = None
        self.end_probs = None
        self.loss = None
        self.pred_ans = None
        self.test = None
        self.accuracy = None
        self.updates = None

        # Tensorboard variables
        # Used to report accuracy and loss values to tensorboard during training/validation
        self.acc_metric = None
        self.acc_metric_update = None
        self.valid_acc_metric = None
        self.valid_acc_metric_update = None
        self.loss_summ = None
        self.acc_summ = None
        self.valid_acc_summ = None
        self.merged_summary = None

    def build_graph(self, grad_clip, embed_init, seed, max_doc_len, max_qry_len):
        # ===========================
        # DEFINING GRAPH PLACEHOLDERS
        # ===========================

        # Placeholder for integer representations of the document and query tokens.
        # These are tensors of shape [batch_size, max_length] where max_length is the length of the longest
        # document or query in the current batch.
        self.doc = tf.placeholder(tf.int32, [None, max_doc_len], name="doc")  # Document words
        self.qry = tf.placeholder(tf.int32, [None, max_qry_len], name="qry")  # Query words

        # Placeholder for the ground truth answer's index in the document.
        # A tensor of shape [batch_size, 2]
        # The values refer to the answer's index in the document. Can be either the index among tokens or chars.
        # [[answer_start_0, answer_end_0]
        #  [answer_start_1, answer_end_1]
        #  [............................]
        #  [answer_start_n, answer_end_n]] - where n = batch_size
        self.answer = tf.placeholder(
            tf.int32, [None, 2], name="answer")

        # Placeholder for document and query masks.
        # These are the same as the document and query placeholders above, except that they are binary,
        # having 0's where there is no token, and 1 where there is.
        # Example:
        # Assuming max_doc_len = 4 and batch_size = 3
        #              <---4---->                           <---4---->
        # self.doc = [[2, 5, 4, 7]  ----> self.doc_mask = [[1, 1, 1, 1]  <-- document 1
        #             [3, 2, 6, 0]                         [1, 1, 1, 0]  <-- document 2
        #             [2, 1, 0, 0]]                        [1, 1, 0, 0]] <-- document 3
        # The masks are used to calculate the sequence length of each text sample going into
        # the bi-directional RNN.
        self.doc_mask = tf.placeholder(
            tf.int32, [None, max_doc_len], name="doc_mask")
        self.qry_mask = tf.placeholder(
            tf.int32, [None, max_qry_len], name="query_mask")

        # Placeholder for character mask.
        # It's a mask over all the unique words broken into characters in the current batch
        # Example: word1 = [1, 2, 3, 4], word2 = [7, 3, 5], MAX_WORD_LEN = 6
        #
        #               <------6------->
        # char_mask = [[1, 1, 1, 1, 0, 0]  <- word1
        #              [1, 1, 1, 0, 0, 0]] <- word2
        # Used for sequence length in the character bi-directional GRU
        self.char_mask = tf.placeholder(
            tf.int32, [None, MAX_WORD_LEN], name="char_mask")

        # Placeholder for document and query character array.
        # These tensors hold only the index of each word (as characters) in the unique word type dictionary
        # Their shapes are [batch_size, max_length]
        # See utils/MiniBatchLoader.py
        # Example:
        # max_doc_len = 4, batch_size = 3
        #                   <---4---->
        # self.doc_char = [[2, 5, 4, 7]  <-- document 1
        #                  [3, 2, 6, 0]  <-- document 2
        #                  [2, 1, 0, 0]] <-- document 3
        #
        self.doc_char = tf.placeholder(
            tf.int32, [None, None], name="doc_char")
        self.qry_char = tf.placeholder(
            tf.int32, [None, None], name="qry_char")

        # Placeholder for the type character array (unique word dictionary)
        # Its shape is [unique_words_in_batch, max_word_length]
        self.token = tf.placeholder(
            tf.int32, [None, MAX_WORD_LEN], name="token")

        # qe-comm feature (see paper, section 3.1.4)
        self.feat = tf.placeholder(
            tf.int32, [None, None], name="features")

        # The predicted answer span's indices in the document
        # Its shape is [batch_size, 2]
        # self.pred_ans = [[predicted_answer_start_0, predicted_answer_end_0]
        #                  [predicted_answer_start_1, predicted_answer_end_1]
        #                  [................................................]
        #                  [predicted_answer_start_n, predicted_answer_end_n]] where batch_size = n
        self.pred_ans = tf.placeholder(tf.int32, [None, 2], name="predicted_answer")

        # Probabilities of a document word being the start or the end of the answer.
        # Shape is [batch_size, max_document_length], values range from (0-1)
        self.start_probs = tf.placeholder(tf.float32, [None, None], name="answer_start_probs")
        self.end_probs = tf.placeholder(tf.float32, [None, None], name="answer_end_probs")

        # Placeholder for the attention matrices generated during forward passes of a document and a query.
        # At the moment of writing, there are K+1 attention matrices saved in a forward pass:
        # An initial one after embedding the document and query, then K during the subsequent K layers.
        # Each attention matrix is shaped: [batch_size, max_document_length, max_query_length]
        # so the shape of this tensor should be: [K+1, batch_size, max_document_length, max_query_length]
        self.attention_tensors = tf.placeholder(tf.float32, [None], name="attentions")

        # Model parameters
        # Initial learning rate
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        # Keep probability = 1 - dropout probability
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # word embedding
        if embed_init is None:
            word_embedding = tf.get_variable(
                "word_embedding", [self.vocab_size, self.embed_dim],
                initializer=tf.glorot_normal_initializer(seed, tf.float32),
                trainable=self.train_emb)
        else:
            word_embedding = tf.Variable(embed_init, trainable=self.train_emb,
                                         name="word_embedding")

        doc_embed = tf.nn.embedding_lookup(
            word_embedding, self.doc, name="document_embedding")
        qry_embed = tf.nn.embedding_lookup(
            word_embedding, self.qry, name="query_embedding")

        # feature embedding
        feature_embedding = tf.get_variable(
            "feature_embedding", [2, 2],
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=self.train_emb)
        feat_embed = tf.nn.embedding_lookup(
            feature_embedding, self.feat, name="feature_embedding")

        # char embedding
        if self.use_chars:
            char_embedding = tf.get_variable(
                "char_embedding", [self.vocab_size_char, self.n_hidden_char],
                initializer=tf.random_normal_initializer(stddev=0.1))
            token_embed = tf.nn.embedding_lookup(char_embedding, self.token)
            fw_gru = GRU(self.n_hidden_char)
            bk_gru = GRU(self.n_hidden_char)
            # fw_states/bk_states: [batch_size, gru_size]
            # only use final state
            seq_length = tf.reduce_sum(self.char_mask, axis=1)
            _, (fw_final_state, bk_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                    fw_gru, bk_gru, token_embed, sequence_length=seq_length,
                    dtype=tf.float32, scope="char_rnn")
            fw_embed = tf.layers.dense(
                fw_final_state, self.embed_dim // 2)
            bk_embed = tf.layers.dense(
                bk_final_state, self.embed_dim // 2)
            merge_embed = fw_embed + bk_embed
            doc_char_embed = tf.nn.embedding_lookup(
                merge_embed, self.doc_char, name="doc_char_embedding")
            qry_char_embed = tf.nn.embedding_lookup(
                merge_embed, self.qry_char, name="query_char_embedding")

            doc_embed = tf.concat([doc_embed, doc_char_embed], axis=2)
            qry_embed = tf.concat([qry_embed, qry_char_embed], axis=2)

        self.attentions = []  # Saving for debugging reasons
        if self.save_attn:
            inter = pairwise_interaction(doc_embed, qry_embed)
            self.attentions.append(inter)

        # Creating the 'K' hops with Bi-directional GRUs
        # TODO: Document with comments extensively, refer to figures (cf.), name paper, link paper
        for i in range(self.n_layers - 1):
            # DOCUMENT
            fw_doc = GRU(self.n_hidden)
            bk_doc = GRU(self.n_hidden)
            # The actual length of each document in the current batch
            seq_length = tf.reduce_sum(self.doc_mask, axis=1)
            (fw_doc_states, bk_doc_states), _ = \
                tf.nn.bidirectional_dynamic_rnn(
                    fw_doc, bk_doc, doc_embed, sequence_length=seq_length,
                    dtype=tf.float32, scope="layer_{}_doc_rnn".format(i)) # TODO: turn off scope for cleaner tensorboard?
            doc_bi_embed = tf.concat([fw_doc_states, bk_doc_states], axis=2)

            # QUERY
            fw_qry = GRU(self.n_hidden)
            bk_qry = GRU(self.n_hidden)
            seq_length = tf.reduce_sum(self.qry_mask, axis=1)
            (fw_qry_states, bk_qry_states), _ = \
                tf.nn.bidirectional_dynamic_rnn(
                    fw_qry, bk_qry, qry_embed, sequence_length=seq_length,
                    dtype=tf.float32, scope="{}_layer_qry_rnn".format(i))
            qry_bi_embed = tf.concat([fw_qry_states, bk_qry_states], axis=2)

            # Pairwise interaction (matrix multiplication)
            inter = pairwise_interaction(doc_bi_embed, qry_bi_embed)
            # Gated attention layer
            doc_inter_embed = gated_attention(
                doc_bi_embed, qry_bi_embed, inter, self.qry_mask,
                gating_fn=self.gating_fn)
            doc_embed = tf.nn.dropout(doc_inter_embed, self.keep_prob)
            if self.save_attn:
                self.attentions.append(inter)

        if self.use_qe_comm_feature:
            doc_embed = tf.concat([doc_embed, feat_embed], axis=2)

        # Final layer
        # Same as before but there is no gated attention
        fw_doc_final = GRU(self.n_hidden)
        bk_doc_final = GRU(self.n_hidden)
        seq_length = tf.reduce_sum(self.doc_mask, axis=1)
        (fw_doc_states, bk_doc_states), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_doc_final, bk_doc_final, doc_embed, sequence_length=seq_length,
            dtype=tf.float32, scope="final_doc_rnn")
        doc_embed_final = tf.concat([fw_doc_states, bk_doc_states], axis=2)

        fw_qry_final = GRU(self.n_hidden)
        bk_doc_final = GRU(self.n_hidden)
        seq_length = tf.reduce_sum(self.qry_mask, axis=1)
        (fw_qry_states, bk_qry_states), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_qry_final, bk_doc_final, qry_embed, sequence_length=seq_length,
            dtype=tf.float32, scope="final_qry_rnn")
        qry_embed_final = tf.concat([fw_qry_states, bk_qry_states], axis=2)

        inter = pairwise_interaction(doc_embed_final, qry_embed_final)
        if self.save_attn:
            self.attentions.append(inter)

        # TODO: Fix to be tf variable with scope?
        self.attention_tensors = tf.convert_to_tensor(self.attentions, dtype=tf.float32, name="attentions")

        # Attention Sum
        # Transforming the final pairwise interaction matrix (between document and query)
        # The interaction matrix is input into dense layers (1 for answer start- and 1 for end-index)
        # The dense layer output is softmax'd then averaged across query words to obtain predictions.
        self.pred = attention_sum(inter, self.n_hidden_dense)
        self.start_probs = self.pred[0]
        self.end_probs = self.pred[1]
        start_pred_idx = tf.expand_dims(tf.argmax(self.pred[0], axis=1), axis=1)
        end_pred_idx = tf.expand_dims(tf.argmax(self.pred[1], axis=1), axis=1)

        self.pred_ans = tf.concat([start_pred_idx, end_pred_idx], axis=1)

        # TODO: Review if cross entropy is used correctly here
        start_loss = tf.expand_dims(crossentropy(self.pred[0], self.answer[:, 0]), axis=1)
        end_loss = tf.expand_dims(crossentropy(self.pred[1], self.answer[:, 1]), axis=1)
        # TODO: Is it correct to average the losses on answer start- and end-index?
        total_loss = tf.reduce_mean(
            tf.concat([start_loss, end_loss], axis=1), axis=1)
        self.loss = tf.reduce_mean(total_loss)

        # TEST
        self.pred_ans = tf.cast(self.pred_ans, tf.int32)
        self.test = tf.cast(
            tf.equal(self.answer, self.pred_ans), tf.float32)

        self.accuracy = tf.reduce_sum(tf.cast(
            tf.equal(self.answer, self.pred_ans), tf.float32))
        # self.accuracy = tf.reduce_sum(self.accuracy)  # Not necessary since, it's already scalar
        self.accuracy /= 2
        self.acc_metric, self.acc_metric_update = tf.metrics.accuracy(
            self.answer, self.pred_ans, name="accuracy_metric")
        self.valid_acc_metric, self.valid_acc_metric_update = tf.metrics.accuracy(
            self.answer, self.pred_ans, name="valid_accuracy_metric")

        vars_list = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_list), grad_clip)
        self.updates = optimizer.apply_gradients(zip(grads, vars_list))
        self.save_vars()

        # Tensorboard summaries
        self.acc_summ = tf.summary.scalar('acc_metric', self.acc_metric_update)
        self.loss_summ = tf.summary.scalar('loss_metric', self.loss)
        self.merged_summary = tf.summary.merge_all()

        self.valid_acc_summ = tf.summary.scalar('valid_acc_metric', self.valid_acc_metric_update)

    def save_vars(self):
        """
        for restoring model
        """
        tf.add_to_collection('doc', self.doc)
        tf.add_to_collection('qry', self.qry)
        tf.add_to_collection('doc_char', self.doc_char)
        tf.add_to_collection('qry_char', self.qry_char)
        tf.add_to_collection('answer', self.answer)
        tf.add_to_collection('doc_mask', self.doc_mask)
        tf.add_to_collection('qry_mask', self.qry_mask)
        tf.add_to_collection('token', self.token)
        tf.add_to_collection('char_mask', self.char_mask)
        tf.add_to_collection('feat', self.feat)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('accuracy', self.accuracy)
        tf.add_to_collection('updates', self.updates)
        tf.add_to_collection('learning_rate', self.learning_rate)
        tf.add_to_collection('use_chars', self.use_chars)
        tf.add_to_collection('predicted_answer', self.pred_ans)
        tf.add_to_collection('answer_start_probs', self.start_probs)
        tf.add_to_collection('answer_end_probs', self.end_probs)
        tf.add_to_collection('attentions', self.attention_tensors)

    def train(self, sess, training_data, dropout, learning_rate, iteration, writer, epoch, max_it):
        """
        train model
        Args:
        - data: (object) containing training data
        """
        document_array, document_character_array, query_array, query_character_array,\
            answer_array, document_mask_array, query_mask_array, type_character_array,\
            type_character_mask, filenames = training_data

        feed_dict = {self.doc: document_array, self.qry: query_array,
                     self.doc_char: document_character_array, self.qry_char: query_character_array,
                     self.answer: answer_array, self.doc_mask: document_mask_array,
                     self.qry_mask: query_mask_array, self.token: type_character_array,
                     self.char_mask: type_character_mask, self.keep_prob: 1 - dropout,
                     self.learning_rate: learning_rate}

        if self.use_qe_comm_feature:
            feature = prepare_input(document_array, query_array)
            feed_dict += {self.feat: feature}

        if iteration % 10 == 0:  # Get updated summary for Tensorboard every 10th iteration
            loss, accuracy, updates, merged_summ = sess.run([self.loss, self.accuracy,
                                                             self.updates, self.merged_summary], feed_dict)
            # TODO: add merged summaries to writer, once they are done
            writer.add_summary(merged_summ, (epoch * max_it + iteration))
        else:  # Otherwise, get regular updates
            loss, accuracy, updates = \
                sess.run([self.loss, self.accuracy,
                          self.updates], feed_dict)

        return loss, accuracy, updates

    def validate(self, sess, valid_batch_loader,
                 iteration=None, writer=None,
                 epoch=None, max_it=None):
        """
        Validate/Test the model
        """
        loss = accuracy = n_example = 0
        tr = trange(
            len(valid_batch_loader),
            desc="loss: {:.3f}, accuracy: {:.3f}".format(0.0, 0.0),
            leave=False,
            ascii=True)
        start_time = time.time()
        for validation_data in valid_batch_loader:
            document_array, document_character_array, query_array, query_character_array, answer_array,\
                document_mask_array, query_mask_array, type_character_array, type_character_mask,\
                filenames = validation_data

            feed_dict = {self.doc: document_array, self.qry: query_array,
                         self.doc_char: document_character_array, self.qry_char: query_character_array,
                         self.answer: answer_array, self.doc_mask: document_mask_array,
                         self.qry_mask: query_mask_array, self.token: type_character_array,
                         self.char_mask: type_character_mask, self.keep_prob: 1.,
                         self.learning_rate: 0.}

            if self.use_qe_comm_feature:
                feature = prepare_input(document_array, query_array)
                feed_dict += {self.feat: feature}

            _loss, _accuracy, valid_acc_summary = \
                sess.run([self.loss, self.accuracy, self.valid_acc_summ], feed_dict)

            n_example += document_array.shape[0]
            loss += _loss
            accuracy += _accuracy
            tr.set_description("loss: {:.3f}, accuracy: {:.3f}".
                               format(_loss, _accuracy / document_array.shape[0]))
            tr.update()

        tr.close()
        if writer is not None:
            writer.add_summary(valid_acc_summary, (epoch * max_it + iteration))

        loss /= n_example
        accuracy /= n_example
        time_spent = (time.time() - start_time) / 60
        statement = "loss: {:.3f}, accuracy: {:.3f}, time: {:.1f}(m)" \
            .format(loss, accuracy, time_spent)
        logging.info(statement)
        return loss, accuracy

    def predict(self, sess, batch_loader):

        output = []
        for samples in batch_loader:
            document_array, document_character_array, query_array, query_character_array, answer_array,\
                document_mask_array, query_mask_array, type_character_array, type_character_mask,\
                filenames = samples

            feed_dict = {self.doc: document_array, self.qry: query_array,
                         self.doc_char: document_character_array, self.qry_char: query_character_array,
                         self.answer: answer_array, self.doc_mask: document_mask_array,
                         self.qry_mask: query_mask_array, self.token: type_character_array,
                         self.char_mask: type_character_mask, self.keep_prob: 1.,
                         self.learning_rate: 0.}

            document, query, answer,\
                predicted_answer, answer_start_probabilities,\
                answer_end_probabilities, attention_tensors = sess.run([self.doc, self.qry, self.answer,
                                                                        self.pred_ans, self.start_probs,
                                                                        self.end_probs, self.attention_tensors],
                                                                       feed_dict)
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
        self.doc = tf.get_collection('doc')[0]
        self.qry = tf.get_collection('qry')[0]
        self.doc_char = tf.get_collection('doc_char')[0]
        self.qry_char = tf.get_collection('qry_char')[0]
        self.answer = tf.get_collection('answer')[0]
        self.doc_mask = tf.get_collection('doc_mask')[0]
        self.qry_mask = tf.get_collection('qry_mask')[0]
        self.token = tf.get_collection('token')[0]
        self.char_mask = tf.get_collection('char_mask')[0]
        self.feat = tf.get_collection('feat')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]
        self.loss = tf.get_collection('loss')[0]
        self.accuracy = tf.get_collection('accuracy')[0]
        self.updates = tf.get_collection('updates')[0]
        self.learning_rate = tf.get_collection('learning_rate')[0]
        self.pred_ans = tf.get_collection('predicted_answer')[0]
        self.start_probs = tf.get_collection('answer_start_probs')[0]
        self.end_probs = tf.get_collection('answer_end_probs')[0]
        self.use_chars = tf.get_collection('use_chars')[0]
        self.attention_tensors = tf.get_collection('attentions')[0]

    def save(self, sess, saver, checkpoint_dir, epoch):
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch{}.ckpt'.format(epoch))
        saver.save(sess, checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))