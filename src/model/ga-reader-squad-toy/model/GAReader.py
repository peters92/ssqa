import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell as GRU
import time
import os
import logging
from tqdm import trange
from model.model_helper import gated_attention,\
                               pairwise_interaction,\
                               attention_sum,\
                               attention_sum_cloze, \
                               crossentropy
from utils.Helpers import prepare_input

MAX_WORD_LEN = 10


class GAReader:
    def __init__(self, n_layers, vocab_size, n_chars,
                 n_hidden, n_hidden_dense, embed_dim, train_emb, char_dim,
                 use_feat, gating_fn, save_attn=False, use_cloze_style=False):
        self.n_hidden = n_hidden
        self.n_hidden_dense = n_hidden_dense
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.train_emb = train_emb
        self.char_dim = char_dim
        self.n_chars = n_chars
        self.use_feat = use_feat
        self.save_attn = save_attn
        self.gating_fn = gating_fn
        self.vocab_size = vocab_size
        self.use_chars = self.char_dim != 0
        self.use_cloze_style = use_cloze_style
        # Graph initialization
        self.doc = None
        self.qry = None
        if use_cloze_style:  # These are only needed for cloze-style QA
            self.cand = None
            self.cand_mask = None
            self.cloze = None
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

        self.pred = None
        self.loss = None
        self.pred_ans = None
        self.test = None
        self.accuracy = None
        self.updates = None

        self.acc_metric = None
        self.acc_metric_update = None
        self.acc_sum = None

        self.merged_summary = None

    def build_graph(self, grad_clip, embed_init, seed, max_doc_len, max_qry_len, use_cloze_style=False):
        # Defining inputs
        with tf.name_scope("Inputs"):
            self.doc = tf.placeholder(
                tf.int32, [None, max_doc_len], name="doc")  # Document words
            self.qry = tf.placeholder(
                tf.int32, [None, max_qry_len], name="qry")  # Query words

            if use_cloze_style:  # Cloze-style data
                self.answer = tf.placeholder(
                    tf.int32, [None, ], name="answer")              # Answer
                self.cand = tf.placeholder(
                    tf.int32, [None, None, None], name="cand_ans")  # Candidate answers
                self.cloze = tf.placeholder(
                    tf.int32, [None, ], name="cloze")               # Cloze
                self.cand_mask = tf.placeholder(
                    tf.int32, [None, None], name="cand_mask")
            else:  # Span-style data, with start- and end-index
                # Answer looks like -> Batch_size * [answer_start_index, answer_end_index]
                self.answer = tf.placeholder(
                    tf.int32, [None, 2], name="answer")

            # word mask TODO: dtype could be changed to int8 or bool
            self.doc_mask = tf.placeholder(
                tf.int32, [None, None], name="doc_mask")
            self.qry_mask = tf.placeholder(
                tf.int32, [None, None], name="query_mask")
            # character mask
            self.char_mask = tf.placeholder(
                tf.int32, [None, MAX_WORD_LEN], name="char_mask")
            # character input
            self.doc_char = tf.placeholder(
                tf.int32, [None, None], name="doc_char")
            self.qry_char = tf.placeholder(
                tf.int32, [None, None], name="qry_char")
            self.token = tf.placeholder(
                tf.int32, [None, MAX_WORD_LEN], name="token")
            # extra features, see GA Reader (Dhingra et al.) paper, "question evidence common word feature"
            self.feat = tf.placeholder(
                tf.int32, [None, None], name="features")

        # model parameters
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        with tf.name_scope("Embeddings"):
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
            with tf.name_scope("Character_Embeddings"):
                if self.use_chars:
                    char_embedding = tf.get_variable(
                        "char_embedding", [self.n_chars, self.char_dim],
                        initializer=tf.random_normal_initializer(stddev=0.1))
                    token_embed = tf.nn.embedding_lookup(char_embedding, self.token)
                    fw_gru = GRU(self.char_dim)
                    bk_gru = GRU(self.char_dim)
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
        for i in range(self.n_layers - 1):
            # DOCUMENT
            with tf.name_scope("Document"):
                fw_doc = GRU(self.n_hidden)
                bk_doc = GRU(self.n_hidden)
                seq_length = tf.reduce_sum(self.doc_mask, axis=1)  # actual length of each document
                (fw_doc_states, bk_doc_states), _ = \
                    tf.nn.bidirectional_dynamic_rnn(
                        fw_doc, bk_doc, doc_embed, sequence_length=seq_length,
                        dtype=tf.float32, scope="layer_{}_doc_rnn".format(i)) # TODO: turn off scope for cleaner tensorboard?
                doc_bi_embed = tf.concat([fw_doc_states, bk_doc_states], axis=2)

            # QUERY
            with tf.name_scope("Query"):
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

        if self.use_feat:
            doc_embed = tf.concat([doc_embed, feat_embed], axis=2)

        # Final layer
        # Same as before but there is no gated attention
        with tf.name_scope("Final_Layer_Document"):
            fw_doc_final = GRU(self.n_hidden)
            bk_doc_final = GRU(self.n_hidden)
            seq_length = tf.reduce_sum(self.doc_mask, axis=1)
            (fw_doc_states, bk_doc_states), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_doc_final, bk_doc_final, doc_embed, sequence_length=seq_length,
                dtype=tf.float32, scope="final_doc_rnn")
            doc_embed_final = tf.concat([fw_doc_states, bk_doc_states], axis=2)

        with tf.name_scope("Final_Layer_Query"):
            fw_qry_final = GRU(self.n_hidden)
            bk_doc_final = GRU(self.n_hidden)
            seq_length = tf.reduce_sum(self.qry_mask, axis=1)
            (fw_qry_states, bk_qry_states), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_qry_final, bk_doc_final, qry_embed, sequence_length=seq_length,
                dtype=tf.float32, scope="final_qry_rnn")
            qry_embed_final = tf.concat([fw_qry_states, bk_qry_states], axis=2)

        if not self.use_cloze_style:
            # Final interaction matrix
            inter = pairwise_interaction(doc_embed_final, qry_embed_final)
        if self.save_attn:
            self.attentions.append(inter)

        # Attention Sum
        with tf.name_scope("Prediction"):
            if self.use_cloze_style:
                self.pred = attention_sum_cloze(
                    doc_embed_final, qry_embed_final, self.cand,
                    self.cloze, self.cand_mask)
                # Making the prediction by taking the max. probability among candidates
                self.pred_ans = tf.cast(tf.argmax(self.pred, axis=1), tf.int32)
            else:  # Span-style
                # Transforming the final pairwise interaction matrix (between document and query)
                # The interaction matrix is input into dense layers (1 for answer start- and 1 for end-index)
                # The dense layer output is softmax'd then averaged across query words to obtain predictions.
                self.pred = attention_sum(inter, self.n_hidden_dense, name="attention_sum")
                start_pred_idx = tf.expand_dims(tf.argmax(self.pred[0], axis=1), axis=1)
                end_pred_idx = tf.expand_dims(tf.argmax(self.pred[1], axis=1), axis=1)

                self.pred_ans = tf.concat([start_pred_idx, end_pred_idx], axis=1)

        with tf.name_scope("Loss"):
            if self.use_cloze_style:
                self.loss = tf.reduce_mean(crossentropy(self.pred, self.answer))
            else:  # Span-style
                # TODO: Review if cross entropy is used correctly here
                start_loss = tf.expand_dims(crossentropy(self.pred[0], self.answer[:, 0]), axis=1)
                end_loss = tf.expand_dims(crossentropy(self.pred[1], self.answer[:, 1]), axis=1)
                # TODO: Is it correct to average the losses on answer start- and end-index?
                total_loss = tf.reduce_mean(
                    tf.concat([start_loss, end_loss], axis=1), axis=1)
                self.loss = tf.reduce_mean(total_loss)
        with tf.name_scope("Test"):
            if self.use_cloze_style:
                self.test = tf.cast(
                    tf.equal(self.answer, self.pred_ans), tf.float32)
            else:  # Span-style
                self.pred_ans = tf.cast(self.pred_ans, tf.int32)
                self.test = tf.cast(
                    tf.equal(self.answer, self.pred_ans), tf.float32)

        with tf.name_scope("Accuracy"):
            if self.use_cloze_style:
                self.accuracy = tf.reduce_sum(
                    tf.cast(tf.equal(self.answer, self.pred_ans), tf.float32))
                self.acc_metric, self.acc_metric_update = tf.metrics.accuracy(
                    self.answer, self.pred_ans)
            else:  # Span-style
                accuracy = tf.reduce_sum(tf.cast(
                    tf.equal(self.answer, self.pred_ans), tf.float32))
                self.accuracy = tf.reduce_sum(accuracy)  # TODO: Might be wrong, seems low on tensorboard
                self.acc_metric, self.acc_metric_update = tf.metrics.accuracy(
                    self.answer, self.pred_ans)

        vars_list = tf.trainable_variables()

        with tf.name_scope("Train"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # gradient clipping
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, vars_list), grad_clip)
        self.updates = optimizer.apply_gradients(zip(grads, vars_list))
        self.save_vars()

        # Tensorboard summaries
        self.acc_summ = tf.summary.scalar('acc_metric', self.acc_metric)
        self.merged_summary = tf.summary.merge_all()

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
        if self.use_cloze_style:
            tf.add_to_collection('cand', self.cand)
            tf.add_to_collection('cand_mask', self.cand_mask)
            tf.add_to_collection('cloze', self.cloze)
        tf.add_to_collection('feat', self.feat)
        tf.add_to_collection('keep_prob', self.keep_prob)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('accuracy', self.accuracy)
        tf.add_to_collection('updates', self.updates)
        tf.add_to_collection('learning_rate', self.learning_rate)

# Replace train inputs with one input, then unpack the tuple input within the definition.
    def train(self, sess, training_data, dropout, learning_rate, iteration, writer, epoch, max_it):
        """
        train model
        Args:
        - data: (object) containing training data
        """
        if self.use_cloze_style:
            dw, dt, qw, qt, a, m_dw, m_qw,\
                tt, tm, c, m_c, cl, fnames = training_data

            feed_dict = {self.doc: dw, self.qry: qw,
                         self.doc_char: dt, self.qry_char: qt, self.answer: a,
                         self.doc_mask: m_dw, self.qry_mask: m_qw,
                         self.token: tt, self.char_mask: tm,
                         self.cand: c, self.cand_mask: m_c,
                         self.cloze: cl, self.keep_prob: 1 - dropout,
                         self.learning_rate: learning_rate}
        else:  # Span-style data
            dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, fnames = training_data

            # Original answer contains character and token indices both
            # Here, I mask out the unnecessary part
            if self.use_chars:  # Use the character indices
                a = a[:, :2]
            else:  # Use the word-token indices
                a = a[:, 2:]

            feed_dict = {self.doc: dw, self.qry: qw,
                         self.doc_char: dt, self.qry_char: qt,
                         self.answer: a, self.doc_mask: m_dw,
                         self.qry_mask: m_qw, self.token: tt,
                         self.char_mask: tm, self.keep_prob: 1 - dropout,
                         self.learning_rate: learning_rate}
        if self.use_feat:
            feat = prepare_input(dw, qw)
            feed_dict += {self.feat: feat}

        loss, acc, updates, attentions, pred, answer = \
            sess.run([self.loss, self.accuracy,
                      self.updates, self.attentions,
                      self.pred, self.answer], feed_dict)

        # Adding TensorBoard summary
        if iteration % 10 == 0:
            sess.run(self.acc_metric_update, feed_dict)
            writer.add_summary(self.acc_summ.eval(), (epoch * max_it + iteration))

        return loss, acc, updates, attentions, pred, answer

    def validate(self, sess, valid_batch_loader):
        """
        test the model
        """
        loss = acc = n_example = 0
        tr = trange(
            len(valid_batch_loader),
            desc="loss: {:.3f}, acc: {:.3f}".format(0.0, 0.0),
            leave=False)
        # SQUAD_MOD
        start = time.time()
        for validation_data in valid_batch_loader:
            if self.use_cloze_style:
                dw, dt, qw, qt, a, m_dw, m_qw, tt, \
                    tm, c, m_c, cl, fnames = validation_data
                feed_dict = {self.doc: dw, self.qry: qw,
                             self.doc_char: dt, self.qry_char: qt, self.answer: a,
                             self.doc_mask: m_dw, self.qry_mask: m_qw,
                             self.token: tt, self.char_mask: tm,
                             self.cand: c, self.cand_mask: m_c,
                             self.cloze: cl, self.keep_prob: 1.,
                             self.learning_rate: 0.}
            else:  # Span-style
                dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, fnames = validation_data

                if self.use_chars:  # Use the character indices
                    a = a[:, :2]
                else:  # Use the word-token indices
                    a = a[:, 2:]

                feed_dict = {self.doc: dw, self.qry: qw,
                             self.doc_char: dt, self.qry_char: qt,
                             self.answer: a, self.doc_mask: m_dw,
                             self.qry_mask: m_qw, self.token: tt,
                             self.char_mask: tm, self.keep_prob: 1.,
                             self.learning_rate: 0.}

            if self.use_feat:
                feat = prepare_input(dw, qw)
                feed_dict += {self.feat: feat}
            _loss, _acc = sess.run([self.loss, self.accuracy], feed_dict)
            n_example += dw.shape[0]
            loss += _loss
            acc += _acc
            tr.set_description("loss: {:.3f}, acc: {:.3f}".
                               format(_loss, _acc / dw.shape[0]))
            tr.update()
        # SQUAD_MOD
        tr.close()
        loss /= n_example
        acc /= n_example
        spend = (time.time() - start) / 60
        statement = "loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)" \
            .format(loss, acc, spend)
        logging.info(statement)
        return loss, acc

    def restore(self, sess, checkpoint_dir):
        """
        restore model
        """
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
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
        if self.use_cloze_style:
            self.cand = tf.get_collection('cand')[0]
            self.cand_mask = tf.get_collection('cand_mask')[0]
            self.cloze = tf.get_collection('cloze')[0]
        self.feat = tf.get_collection('feat')[0]
        self.keep_prob = tf.get_collection('keep_prob')[0]
        self.loss = tf.get_collection('loss')[0]
        self.accuracy = tf.get_collection('accuracy')[0]
        self.updates = tf.get_collection('updates')[0]
        self.learning_rate = tf.get_collection('learning_rate')[0]

    def save(self, sess, saver, checkpoint_dir, epoch):
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch{}.ckpt'.format(epoch))
        saver.save(sess, checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))