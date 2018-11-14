import numpy as np
import random

MAX_WORD_LEN = 10


class MiniBatchLoader:
    def __init__(self, questions, batch_size,
                 shuffle=True, sample=1.0, prediction_only=None):
        self.batch_size = batch_size
        if sample == 1.0:
            self.questions = questions
        else:
            self.questions = random.sample(
                questions, int(sample * len(questions)))

        self.max_qry_len = max(list(map(lambda x: len(x[1]), self.questions)))
        # TEMP DEBUGGING TODO: Change this once, dense layer problem is fixed
        self.max_doc_len = 1024
        self.max_qry_len = 62
        # TEMP DEBUGGING

        # Normal behaviour, build bins according to the defined method
        # Otherwise if we are predicting only, use only one bin
        if prediction_only is None:
            self.bins = self.build_bins(self.questions)
        else:  # Return a dict with the max doc length and all question indices
            indices = [index for index, value in enumerate(self.questions)]
            self.bins = {self.max_doc_len: indices}

        self.max_word_len = MAX_WORD_LEN
        self.shuffle = shuffle
        self.reset()

    def __len__(self):
        return len(self.batch_pool)

    def __iter__(self):
        """make the object iterable"""
        return self

    def build_bins(self, questions):
        """
        Returns a dictionary
            key: document length (rounded to the powers of two)
            value: indexes of questions with document length equal to key
        """

        # round the input to the nearest power of two
        def round_to_power(x):
            return 2 ** (int(np.log2(x - 1)) + 1)

        doc_len = list(map(lambda x: round_to_power(len(x[0])), questions))
        max_doc_len = max(doc_len)

        bins = {}
        for i, l in enumerate(doc_len):
            if l not in bins:
                bins[l] = []
            bins[l].append(i)

        return bins  # ,max_doc_len

    def reset(self):
        """new iteration"""
        self.ptr = 0

        # randomly shuffle the question indices in each bin
        if self.shuffle:
            for ixs in self.bins.values():
                random.shuffle(ixs)

        # construct a list of mini-batches where each batch
        # is a list of question indices
        # questions within the same batch have identical max
        # document length
        self.batch_pool = []
        for l, ixs in self.bins.items():
            n = len(ixs)
            k = n / self.batch_size if \
                n % self.batch_size == 0 else n / self.batch_size + 1
            ixs_list = [(ixs[self.batch_size * i:
                             min(n, self.batch_size * (i + 1))], l)
                        for i in range(int(k))]
            self.batch_pool += ixs_list

        # randomly shuffle the mini-batches
        if self.shuffle:
            random.shuffle(self.batch_pool)

    def __next__(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr][0]
        curr_batch_size = len(ixs)

        curr_max_doc_len = self.max_doc_len

        # document words
        dw = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # query words
        qw = np.zeros(
            (curr_batch_size, self.max_qry_len),
            dtype='int32')
        # document word mask
        m_dw = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # query word mask
        m_qw = np.zeros(
            (curr_batch_size, self.max_qry_len),
            dtype='int32')

        # The answer contains start- and end-index
        a = np.zeros((curr_batch_size, 2), dtype='int32')

        fnames = [''] * curr_batch_size

        types = {}

        for n, ix in enumerate(ixs):
            doc_w, qry_w, ans, doc_c, qry_c, \
                ans_start, ans_end, fname = self.questions[ix]

            # document, query and candidates
            dw[n, : len(doc_w)] = np.array(doc_w)
            qw[n, : len(qry_w)] = np.array(qry_w)
            m_dw[n, : len(doc_w)] = 1
            m_qw[n, : len(qry_w)] = 1
            for it, word in enumerate(doc_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((0, n, it))
            for it, word in enumerate(qry_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((1, n, it))

            # Store the indices of the answer in the paragraph
            a[n, :] = np.array([ans_start, ans_end])

            fnames[n] = fname

        # create type character matrix and indices for doc, qry
        # document token index
        dt = np.zeros(
            (curr_batch_size, curr_max_doc_len),
            dtype='int32')
        # query token index
        qt = np.zeros(
            (curr_batch_size, self.max_qry_len),
            dtype='int32')
        # type characters
        tt = np.zeros(
            (len(types), self.max_word_len),
            dtype='int32')
        # type mask
        tm = np.zeros(
            (len(types), self.max_word_len),
            dtype='int32')
        n = 0
        for k, v in types.items():
            tt[n, : len(k)] = np.array(k)
            tm[n, : len(k)] = 1
            for (sw, bn, sn) in v:
                if sw == 0:
                    dt[bn, sn] = n
                else:
                    qt[bn, sn] = n

            n += 1

        self.ptr += 1

        return dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, fnames
