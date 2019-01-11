"""
Gated-Attention Reader model, implemented according to the paper:
"Gated-Attention Readers for Text Comprehension" by Bhuwan Dhingra/Hanxiao Liu et al.
    arXiv:  1606.01549v3
    github: https://github.com/bdhingra/ga-reader
Initial tensorflow implementation from: https://github.com/mingdachen/gated-attention-reader
Code adapted for answer span prediction (datasets such as SQuADv1.1,
link: https://rajpurkar.github.io/SQuAD-explorer/).
For the sake of brevity, unless otherwise stated all references saying "see paper" below refers to this paper.
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell as GRU
import time
import os
import logging
from tqdm import trange
from model.layers import gated_attention,\
                               pairwise_interaction,\
                               attention_sum,\
                               crossentropy
from utils.Helpers import calculate_accuracies

# Maximum word length, used for limiting word lengths if using character model
MAX_WORD_LEN = 10


class GAReader:
    def __init__(self, n_layers, dictionaries, vocab_size, vocab_size_char,
                 n_hidden, n_hidden_dense, embed_dim, train_emb, n_hidden_char,
                 use_feat, gating_fn, save_attn=False):
        # Input variables
        self.n_hidden = n_hidden                # The number of hidden units in the GRU cell
        self.n_hidden_dense = n_hidden_dense    # The number of hidden units in the final dense layer
        self.n_layers = n_layers                # The number of layers (or 'hops', see paper fig. 1) in the model
        self.embed_dim = embed_dim              # The size of the initial embedding vectors (e.g. GloVe)
        self.train_emb = train_emb              # Bool: train embeddings or not
        self.use_qe_comm_feature = use_feat     # Bool: use qe-comm feature or not (see paper, section 3.1.4)
        # The "activation" function at the end of the gated attention layer
        # Default is tf.multiply()
        self.gating_fn = eval(gating_fn)
        self.save_attn = save_attn              # Bool: save attention matrices during forward pass or not
        self.word_dictionary = dictionaries[0]  # The word dictionary, used in accuracy
        self.vocab_size = vocab_size            # Size of the word vocabulary (unique word tokens)

        # Input (only for character model)
        self.n_hidden_char = n_hidden_char      # The number of hidden units in the character GRU cell
        self.vocab_size_char = vocab_size_char  # Number of different characters in vocabulary
        self.use_chars = self.n_hidden_char != 0  # Bool: Whether or not to train a character model

        # Graph initialization
        # See their explanation below in the build_graph() method
        self.document = None
        self.query = None
        self.answer = None
        self.document_mask = None
        self.query_mask = None
        self.document_char = None
        self.query_char = None
        self.token = None
        self.char_mask = None
        self.feature = None
        self.learning_rate = None
        self.keep_prob = None
        self.attentions = None
        self.attention_tensors = None
        self.prediction = None
        self.start_probabilities = None
        self.end_probabilities = None
        self.predicted_answer = None
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
        # F1
        # self.F1_acc_metric = None
        # self.F1_acc_metric_update = None
        # self.F1_valid_acc_metric = None
        # self.F1_valid_acc_metric_update = None

        self.loss_summ = None
        self.em_acc_summ = None
        self.em_valid_acc_summ = None
        # self.F1_acc_summ = None
        # self.F1_valid_acc_summ = None
        self.merged_summary = None

    def build_graph(self, grad_clip, embed_init, seed, max_doc_len, max_qry_len):
        # ============================================================================================================
        #                                         DEFINING GRAPH PLACEHOLDERS
        # ============================================================================================================

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

        # Placeholder for character mask.
        # It's a mask over all the unique words broken into characters in the current batch
        # Example: word1 = [1, 2, 3, 4], word2 = [7, 3, 5], MAX_WORD_LEN = 6
        #
        #               <------6------->
        # char_mask = [[1, 1, 1, 1, 0, 0]  <- word1
        #              [1, 1, 1, 0, 0, 0]] <- word2
        #
        # Used for sequence length in the character bi-directional GRU
        self.char_mask = tf.placeholder(
            tf.int32, [None, MAX_WORD_LEN], name="char_mask")

        # Placeholder for document and query character array.
        # These tensors hold only the index of each word (as characters) in the unique word type dictionary
        # Their shapes are [batch_size, max_length]
        # See utils/MiniBatchLoader.py
        # Example:
        # max_doc_len = 4, batch_size = 3
        #
        #                        <---4---->
        # self.document_char = [[2, 5, 4, 7]  <-- document 1
        #                       [3, 2, 6, 0]  <-- document 2
        #                       [2, 1, 0, 0]] <-- document 3
        #
        self.document_char = tf.placeholder(
            tf.int32, [None, None], name="document_char")
        self.query_char = tf.placeholder(
            tf.int32, [None, None], name="query_char")

        # Placeholder for the type character array (unique word dictionary)
        # Its shape is [unique_words_in_batch, max_word_length]
        self.token = tf.placeholder(
            tf.int32, [None, MAX_WORD_LEN], name="token")

        # qe-comm feature (see paper, section 3.1.4)
        self.feature = tf.placeholder(
            tf.int32, [None, max_doc_len], name="features")

        # The predicted answer span's indices in the document
        # Its shape is [batch_size, 2]
        # self.predicted_answer =
        #                 [[predicted_answer_start_0, predicted_answer_end_0]
        #                  [predicted_answer_start_1, predicted_answer_end_1]
        #                  [................................................]
        #                  [predicted_answer_start_n, predicted_answer_end_n]] with batch_size = n
        #
        self.predicted_answer = tf.placeholder(tf.int32, [None, 2], name="predicted_answer")

        # Probabilities of a document word being the start or the end of the answer.
        # Shape is [batch_size, max_document_length], values range from (0-1)
        self.start_probabilities = tf.placeholder(tf.float32, [None, None], name="answer_start_probs")
        self.end_probabilities = tf.placeholder(tf.float32, [None, None], name="answer_end_probs")

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

        # ============================================================================================================
        #                                           BUILDING THE GRAPH
        # ============================================================================================================

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

        # Assert shape change from [batch_size, max_doc_len] to
        # [batch_size, max_doc_len, embed_dim]
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

        # Create embedding for "Question-Evidence Common Word Feature"
        # According to paper: "Dataset and Neural Recurrent Sequence Labeling Model for
        # Open-Domain Factoid Question Answering" by Li et al. 2016
        # arXiv:1607.06275v2
        # This a very simple feature which marks for each wprd in the document if that same word
        # can also be found in the query. This one-hot representation is then embedded using
        # a shape [2, 2] lookup table that is randomly initialized. Later the embeddings get
        # concatenated with the final layer document embedding along the last axis (hidden dim).
        #
        # EXAMPLE, assuming that:
        # batch_size = 2, max_document_length = 3, embedding_dim = 2
        #                             <- 2->
        # feature_embedding_lookup = [[1, 1]  <- if the document word is not in the query
        #                             [2, 2]] <- if the document word is in the query
        #            <---3--->
        # feature = [[1, 0, 0]  < - document 1 (the first word is in the query)
        #            [0, 1, 0]] < - document 2 (the second word is in the query)
        #
        # Then the feature embeddings will be:
        #                      <----------3----------->
        #                       <- 2->  <- 2->  <- 2->
        # feature_embedding = [[[2, 2], [1, 1], [1, 1]]  <- document 1
        #                      [[1, 1], [2, 2], [1, 1]]] <- document 2
        # final shape is [batch_size, max_document_length, 2]
        #
        feature_embedding_lookup = tf.get_variable(
            "feature_embedding_lookup", [2, 2],
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=self.train_emb)
        feature_embedding = tf.nn.embedding_lookup(
            feature_embedding_lookup, self.feature, name="feature_embedding")

        # Assert feature_embedding shape is [None, max_document_length, 2]
        assert feature_embedding.shape.as_list() == [None, max_doc_len, 2],\
            "Expected feature embedding shape [None, {}, {}] but got {}".format(
            max_doc_len, 2, feature_embedding.shape)

        # TODO: Incorporate into the rest
        if self.use_chars:
            char_embedding = tf.get_variable(
                "char_embedding", [self.vocab_size_char, self.n_hidden_char],
                initializer=tf.random_normal_initializer(stddev=0.1))
            token_embed = tf.nn.embedding_lookup(char_embedding, self.token)
            fw_gru = GRU(self.n_hidden_char)
            bk_gru = GRU(self.n_hidden_char)
            # fw_states/bk_states: [batch_size, gru_size]
            # only use final state
            sequence_length = tf.reduce_sum(self.char_mask, axis=1)
            _, (fw_final_state, bk_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                    fw_gru, bk_gru, token_embed, sequence_length=sequence_length,
                    dtype=tf.float32, scope="char_rnn")
            fw_embed = tf.layers.dense(
                fw_final_state, self.embed_dim // 2)
            bk_embed = tf.layers.dense(
                bk_final_state, self.embed_dim // 2)
            merge_embed = fw_embed + bk_embed
            doc_char_embed = tf.nn.embedding_lookup(
                merge_embed, self.document_char, name="doc_char_embedding")
            qry_char_embed = tf.nn.embedding_lookup(
                merge_embed, self.query_char, name="query_char_embedding")

            document_embedding = tf.concat([document_embedding, doc_char_embed], axis=2)
            query_embedding = tf.concat([query_embedding, qry_char_embed], axis=2)

        # Save attention matrices (which are the dot product of document and query embeddings)
        # Here, an initial attention matrix is saved before the embeddings go through the 'K'
        # layers of the model.
        self.attentions = []
        if self.save_attn:
            # Save an initial interaction matrix (dot product of document and query embeddings)
            interaction = pairwise_interaction(document_embedding, query_embedding)
            # Assert interaction matrix shape is [None, max_document_length, max_query_length]
            assert interaction.shape.as_list() == [None, max_doc_len, max_qry_len],\
                "Expected interaction matrix shape [None, {}, {}] but got {}".format(
                max_doc_len, max_qry_len, interaction.shape)

            self.attentions.append(interaction)

        # ----------------------------------------------
        # Creating the 'K' hops with Bi-directional GRUs
        # ----------------------------------------------
        # 'Paper' below refers to "Gated-Attention Readers for Text Comprehension". See link at the top of this file.
        #
        # In this loop the document and query embeddings are passed through the K layers or
        # 'hops' as described in the paper. See figure 1.
        # In each layer document and query embeddings are passed through Gated Recurrent Units (GRUs)
        # x_1 to x_T on the horizontal arrows on figure 1 and eq. 2 for the DOCUMENT.
        # vertical arrows on figure 1 and eq. 3 for the QUERY
        # The embeddings are passed both forward and backward (in reverse sequence) through the GRUs and the outputs
        # are concatenated (see eq. 1 in paper). In Tensorflow this process is done with the function:
        # tf.nn.bidirectional_dynamic_rnn(), from here on refered to as Bi-GRU.
        # After obtaining the concatenated output of the document and query Bi-GRU they are passed through the
        # Gated-Attention module, which consists of taking the following steps:
        #
        # See sections 3.1.2, eq. 5 and 6 in the paper.
        #   "Soft-Attention":
        # 1. Matrix multiply the document and query embeddings to obtain an 'interaction' matrix
        # 2. Apply a softmax function to the interaction matrix
        # 3. Mask the interaction matrix. It will be zero where there are no query words.
        # 4. Normalize the result so that it sums to 1 again across the last dimension (max_query_length)
        # 5. Multiply the result (alpha in eq. 5) with the query embedding to obtain a document token
        #    specific representation of the query (q_i in eq. 5).
        # 6. Finally, element-wise multiply together the query representation with the document embedding
        # The result (after dropout) becomes the input for the next layer in the model.

        # Note: only K-1 layers in loop, because the K'th layer is slightly different (see below)
        for i in range(self.n_layers - 1):
            # -------------------
            # BI-DIRECTIONAL GRUs
            # -------------------

            # DOCUMENT
            # Define the GRU cells used for the forward and backward document sequence
            forward_document = GRU(self.n_hidden)
            backward_document = GRU(self.n_hidden)
            # Get the actual length of documents in the current batch
            sequence_length = tf.reduce_sum(self.document_mask, axis=1)

            # Pass the document through the Bi-GRU (see figure 1 in paper, x_1 to x_T on horizontal arrows)
            (forward_document_states, backward_document_states), _ = \
                tf.nn.bidirectional_dynamic_rnn(
                    forward_document, backward_document, document_embedding, sequence_length=sequence_length,
                    dtype=tf.float32, scope="layer_{}_doc_rnn".format(i))
            # Concatenate the output from the Bi-GRU, see eq. 1 and 2 in paper
            document_bi_embedding = tf.concat([forward_document_states, backward_document_states], axis=2)

            # Assert shape of document_bi_embedding
            assert document_bi_embedding.shape.as_list() == [None, max_doc_len, 2*self.n_hidden],\
                "Expected document_bi_embedding shape [None, {}, {}] but got {}".format(
                max_doc_len, 2*self.n_hidden, document_bi_embedding.shape)

            # QUERY
            # Define the GRU cells used for the forward and backward query sequence
            forward_query = GRU(self.n_hidden)
            backward_query = GRU(self.n_hidden)
            # Get the actual length of queries in the current batch
            sequence_length = tf.reduce_sum(self.query_mask, axis=1)

            # Pass the query through the Bi-GRU (see figure 1 in paper, vertical arrows on top of the figure)
            (forward_query_states, backward_query_states), _ = \
                tf.nn.bidirectional_dynamic_rnn(
                    forward_query, backward_query, query_embedding, sequence_length=sequence_length,
                    dtype=tf.float32, scope="{}_layer_qry_rnn".format(i))
            # Concatenate the output from the Bi-GRU, see eq. 3 in paper
            query_bi_embedding = tf.concat([forward_query_states, backward_query_states], axis=2)

            # Assert shape of document_bi_embedding
            assert query_bi_embedding.shape.as_list() == [None, max_qry_len, 2 * self.n_hidden], \
                "Expected document_bi_embedding shape [None, {}, {}] but got {}".format(
                    max_qry_len, 2 * self.n_hidden, query_bi_embedding.shape)

            # ----------------------
            # GATED-ATTENTION MODULE
            # ----------------------
            # Matrix multiply document and query embeddings to get an 'interaction' matrix
            interaction = pairwise_interaction(document_bi_embedding, query_bi_embedding)

            # Assert interaction matrix shape is [None, max_document_length, max_query_length]
            assert interaction.shape.as_list() == [None, max_doc_len, max_qry_len], \
                "Expected interaction matrix shape [None, {}, {}] but got {}".format(
                    max_doc_len, max_qry_len, interaction.shape)

            # Calculate new document embeddings with Gated-Attention
            doc_inter_embed = gated_attention(
                document_bi_embedding, query_bi_embedding, interaction, self.query_mask,
                self.document_mask, gating_fn=self.gating_fn)

            # Assert gated attention output (new document embedding) shape is [None, max_document_length, 2*n_hidden]
            assert doc_inter_embed.shape.as_list() == [None, max_doc_len, 2*self.n_hidden], \
                "Expected document embedding shape [None, {}, {}] but got {}".format(
                    max_doc_len, 2*self.n_hidden, doc_inter_embed.shape)

            # Apply dropout to the document embedding
            document_embedding = tf.nn.dropout(doc_inter_embed, self.keep_prob)

            # Save the interaction matrix (attention)
            if self.save_attn:
                self.attentions.append(interaction)

        # -----------------------------------
        # The K'th layer ('hop') of the model
        # -----------------------------------

        # Concatenating the Question-Evidence Common Word feature with the last layer's document embedding input
        if self.use_qe_comm_feature:
            document_embedding = tf.concat([document_embedding, feature_embedding], axis=2)

        # Final layer
        # Does exactly the same as the K-1 layers in the loop above, but after getting the concatenated output
        # of the Bi-GRUs, the Gated-Attention module is no longer applied. Instead after calculating another
        # interaction matrix, the attentions are summarized to generate predictions.
        # Note: This implementation differs from the paper after the final layer output.

        fw_doc_final = GRU(self.n_hidden)
        bk_doc_final = GRU(self.n_hidden)
        sequence_length = tf.reduce_sum(self.document_mask, axis=1)
        (forward_document_states, backward_document_states), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_doc_final, bk_doc_final, document_embedding, sequence_length=sequence_length,
            dtype=tf.float32, scope="final_doc_rnn")
        doc_embed_final = tf.concat([forward_document_states, backward_document_states], axis=2)

        fw_qry_final = GRU(self.n_hidden)
        bk_doc_final = GRU(self.n_hidden)
        sequence_length = tf.reduce_sum(self.query_mask, axis=1)
        (forward_query_states, backward_query_states), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_qry_final, bk_doc_final, query_embedding, sequence_length=sequence_length,
            dtype=tf.float32, scope="final_qry_rnn")
        qry_embed_final = tf.concat([forward_query_states, backward_query_states], axis=2)

        # ---------------------
        # End of the K'th layer
        # ---------------------

        # Calculate the final interaction matrix (attention)
        interaction = pairwise_interaction(doc_embed_final, qry_embed_final)
        if self.save_attn:
            self.attentions.append(interaction)

        # Convert the list of saved interaction matrices (attentions) to a tensor, so that it can be retrieved
        # later for visualization purposes and debugging.
        self.attention_tensors = tf.convert_to_tensor(self.attentions, dtype=tf.float32, name="attentions")

        assert self.attention_tensors.shape.as_list() == [self.n_layers+1, None, max_doc_len, max_qry_len],\
            "Expected attention tensors shape [{}, None, {}, {}] but got {}".format(
                    self.n_layers+1, max_doc_len, max_qry_len, self.attention_tensors.shape)

        # -----------
        # PREDICTIONS
        # -----------
        # Transforming the final pairwise interaction matrix (between document and query)
        # The interaction matrix is input into 2 dense layers (1 for answer start- and 1 for end-index)
        # The dense layer output is softmax'd then averaged across the query words to obtain predictions.
        self.prediction = attention_sum(interaction, self.n_hidden_dense, self.keep_prob)
        self.start_probabilities = self.prediction[0]
        self.end_probabilities = self.prediction[1]

        # Assert shape of probabilites:
        assert self.start_probabilities.shape.as_list() == [None, max_doc_len],\
            "Expected start probabilities shape [None, {}] but got {}".format(
            max_doc_len, self.start_probabilities.shape)

        assert self.end_probabilities.shape.as_list() == [None, max_doc_len],\
            "Expected end probabilities shape [None, {}] but got {}".format(
            max_doc_len, self.end_probabilities.shape)

        # Get the index of the predicted answer start and end by taking the maximum of the probabilities
        # of each probability vector.
        start_pred_idx = tf.expand_dims(tf.argmax(self.start_probabilities, axis=1), axis=1)
        end_pred_idx = tf.expand_dims(tf.argmax(self.end_probabilities, axis=1), axis=1)

        # The predicted answer span is defined by these two indices
        self.predicted_answer = tf.concat([start_pred_idx, end_pred_idx],
                                          axis=1, name="prediction")

        # -----------
        #    LOSS
        # -----------
        # Calculating the categorical cross-entropy loss separately for both the answer start and end index
        # Then, an average loss is calculated per answer, and finally an average across the current batch.
        start_loss = tf.expand_dims(crossentropy(self.prediction[0], self.answer[:, 0]), axis=1)
        end_loss = tf.expand_dims(crossentropy(self.prediction[1], self.answer[:, 1]), axis=1)
        # TODO: Is it correct to average the losses on answer start- and end-index?
        total_loss = tf.reduce_mean(
            tf.concat([start_loss, end_loss], axis=1), axis=1)

        self.loss = tf.reduce_mean(total_loss)

        # -------------------
        #      ACCURACY
        # -------------------
        # Exact Match Accuracy
        # Cast prediction as int so it can be compared with the true answers
        self.predicted_answer = tf.cast(self.predicted_answer, tf.int32)
        # Calculate exact match (EM) accuracy, and cast back to float for division
        common_indices = tf.equal(self.answer, self.predicted_answer)
        self.EM_accuracy = tf.reduce_sum(tf.cast(tf.reduce_all(common_indices, axis=1), tf.float32))

        # Exact Match Accuracy, Tensorboard
        # Define the same accuracy to be logged on tensorboard for visual monitoring
        self.em_acc_metric, self.em_acc_metric_update = tf.metrics.accuracy(
            self.answer, self.predicted_answer, name="em_accuracy_metric")
        self.em_valid_acc_metric, self.em_valid_acc_metric_update = tf.metrics.accuracy(
            self.answer, self.predicted_answer, name="em_valid_accuracy_metric")

        vars_list = tf.trainable_variables()

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_list), grad_clip)
        self.updates = optimizer.apply_gradients(zip(grads, vars_list))
        self.save_vars()

        # Tensorboard summaries
        self.em_acc_summ = tf.summary.scalar('exact_match_accuracy', self.em_acc_metric_update)
        self.loss_summ = tf.summary.scalar('cross-entropy_loss', self.loss)
        self.merged_summary = tf.summary.merge_all()

        self.em_valid_acc_summ = tf.summary.scalar('em_valid_acc_metric',
                                                   self.em_valid_acc_metric_update)

    def save_vars(self):
        """
        for restoring model
        """
        tf.add_to_collection('document', self.document)
        tf.add_to_collection('query', self.query)
        tf.add_to_collection('document_char', self.document_char)
        tf.add_to_collection('query_char', self.query_char)
        tf.add_to_collection('answer', self.answer)
        tf.add_to_collection('document_mask', self.document_mask)
        tf.add_to_collection('query_mask', self.query_mask)
        tf.add_to_collection('token', self.token)
        tf.add_to_collection('char_mask', self.char_mask)
        tf.add_to_collection('feature', self.feature)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('accuracy', self.EM_accuracy)
        tf.add_to_collection('updates', self.updates)
        tf.add_to_collection('learning_rate', self.learning_rate)
        tf.add_to_collection('use_chars', self.use_chars)
        tf.add_to_collection('use_qe_comm_feature', self.use_qe_comm_feature)
        tf.add_to_collection('predicted_answer', self.predicted_answer)
        tf.add_to_collection('answer_start_probs', self.start_probabilities)
        tf.add_to_collection('answer_end_probs', self.end_probabilities)
        tf.add_to_collection('attentions', self.attention_tensors)

    def train(self, sess, training_data, dropout, learning_rate, iteration, writer, epoch, max_it):
        """
        Performs one training iteration with input tuple of training data.
        """
        document_array, document_character_array, query_array, query_character_array,\
            answer_array, document_mask_array, query_mask_array, type_character_array,\
            type_character_mask, filenames = training_data

        feed_dict = {self.document: document_array, self.query: query_array,
                     self.document_char: document_character_array,
                     self.query_char: query_character_array,
                     self.answer: answer_array, self.document_mask: document_mask_array,
                     self.query_mask: query_mask_array, self.token: type_character_array,
                     self.char_mask: type_character_mask, self.keep_prob: 1 - dropout,
                     self.learning_rate: learning_rate}

        if self.use_qe_comm_feature:
            feature = np.isin(document_array, query_array)
            feed_dict[self.feature] = feature

        if iteration % 50 == 0:  # Get updated summary for Tensorboard every 10th iteration
            loss, accuracy, predicted_answer_array, updates, merged_summ = \
                sess.run([self.loss, self.EM_accuracy, self.predicted_answer,
                          self.updates, self.merged_summary], feed_dict)

            writer.add_summary(merged_summ, (epoch * max_it + iteration))
        else:  # Otherwise, get regular updates
            loss, accuracy, predicted_answer_array, updates = \
                sess.run([self.loss, self.EM_accuracy, self.predicted_answer,
                          self.updates], feed_dict)

        # Calculate F1 Score and Exact Match accuracy over the batch
        f1_score, exact_match_accuracy = calculate_accuracies(answer_array, predicted_answer_array,
                                                              document_array, self.word_dictionary)

        return loss, f1_score, exact_match_accuracy, updates

    def validate(self, sess, valid_batch_loader,
                 iteration=None, writer=None,
                 epoch=None, max_it=None):
        """
        Validate/Test the model
        """
        it = loss = em_accuracy = f1_score = n_example = 0

        tr = trange(
            len(valid_batch_loader),
            desc="loss: {:.3f}, EM_accuracy: {:.3f}, F1_accuracy: {:.3f}".format(0.0, 0.0, 0.0),
            leave=False,
            ascii=True)
        start_time = time.time()
        for validation_data in valid_batch_loader:
            it += 1
            document_array, document_character_array, query_array, query_character_array, answer_array,\
                document_mask_array, query_mask_array, type_character_array, type_character_mask,\
                filenames = validation_data

            feed_dict = {self.document: document_array, self.query: query_array,
                         self.document_char: document_character_array,
                         self.query_char: query_character_array,
                         self.answer: answer_array, self.document_mask: document_mask_array,
                         self.query_mask: query_mask_array, self.token: type_character_array,
                         self.char_mask: type_character_mask, self.keep_prob: 1.,
                         self.learning_rate: 0.}

            if self.use_qe_comm_feature:
                feature = np.isin(document_array, query_array)
                feed_dict[self.feature] = feature

            loss_, _accuracy, predicted_answer_array, em_valid_acc_summary = \
                sess.run([self.loss, self.EM_accuracy,
                          self.predicted_answer, self.em_valid_acc_summ], feed_dict)

            # Calculate F1 Score and Exact Match accuracy over the batch
            f1_score_, exact_match_accuracy_ = \
                calculate_accuracies(answer_array, predicted_answer_array,
                                     document_array, self.word_dictionary)

            n_example += document_array.shape[0]
            loss += loss_
            em_accuracy += exact_match_accuracy_
            f1_score += f1_score_
            tr.set_description("loss: {:.3f}, EM_accuracy: {:.3f}, F1_Score: {:.3f}".
                               format(loss_, exact_match_accuracy_, f1_score_))
            tr.update()

        tr.close()
        if writer is not None:
            writer.add_summary(em_valid_acc_summary, (epoch * max_it + iteration))

        loss /= n_example
        em_accuracy /= it
        f1_score /= it
        time_spent = (time.time() - start_time) / 60
        statement = "loss: {:.3f}, EM_accuracy: {:.3f}, F1_Score: {:.3f}, time: {:.1f}(m)" \
            .format(loss, em_accuracy, f1_score, time_spent)
        logging.info(statement)
        return loss, em_accuracy, f1_score

    def predict(self, sess, batch_loader):

        output = []
        for samples in batch_loader:
            document_array, document_character_array, query_array, query_character_array, answer_array,\
                document_mask_array, query_mask_array, type_character_array, type_character_mask,\
                filenames = samples

            feed_dict = {self.document: document_array, self.query: query_array,
                         self.document_char: document_character_array,
                         self.query_char: query_character_array,
                         self.answer: answer_array, self.document_mask: document_mask_array,
                         self.query_mask: query_mask_array, self.token: type_character_array,
                         self.char_mask: type_character_mask, self.keep_prob: 1.,
                         self.learning_rate: 0.}

            if self.use_qe_comm_feature:
                feature = np.isin(document_array, query_array)
                feed_dict[self.feature] = feature

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
        self.query = tf.get_collection('query')[0]
        self.document_char = tf.get_collection('document_char')[0]
        self.query_char = tf.get_collection('query_char')[0]
        self.answer = tf.get_collection('answer')[0]
        self.document_mask = tf.get_collection('document_mask')[0]
        self.query_mask = tf.get_collection('query_mask')[0]
        self.token = tf.get_collection('token')[0]
        self.char_mask = tf.get_collection('char_mask')[0]
        self.feature = tf.get_collection('feature')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]
        self.loss = tf.get_collection('loss')[0]
        self.EM_accuracy = tf.get_collection('accuracy')[0]
        self.updates = tf.get_collection('updates')[0]
        self.learning_rate = tf.get_collection('learning_rate')[0]
        self.predicted_answer = tf.get_collection('predicted_answer')[0]
        self.start_probabilities = tf.get_collection('answer_start_probs')[0]
        self.end_probabilities = tf.get_collection('answer_end_probs')[0]
        self.use_chars = tf.get_collection('use_chars')[0]
        self.use_qe_comm_feature = tf.get_collection('use_qe_comm_feature')[0]
        self.attention_tensors = tf.get_collection('attentions')[0]

    def save(self, sess, saver, checkpoint_dir, epoch):
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch{}.ckpt'.format(epoch))
        saver.save(sess, checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))