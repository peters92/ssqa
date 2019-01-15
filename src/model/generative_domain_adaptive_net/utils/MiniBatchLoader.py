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

        self.max_query_length = max(list(map(lambda x: len(x[1]), self.questions)))
        # TEMP DEBUGGING TODO: Change this once, dense layer problem is fixed
        self.max_document_length = 1024
        self.max_query_length = 62
        # TEMP DEBUGGING

        # Normal behaviour, build bins according to the defined method
        # Otherwise if we are predicting only, use only one bin
        if prediction_only is None:
            self.bins = self.build_bins(self.questions)
        else:  # Return a dict with the max document length and all question indices
            indices = [index for index, value in enumerate(self.questions)]
            self.bins = {self.max_document_length: indices}

        self.max_word_length = MAX_WORD_LEN
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

        document_lengths = list(map(lambda x: round_to_power(len(x[0])), questions))
        max_document_length = max(document_lengths)

        bins = {}
        for i, l in enumerate(document_lengths):
            if l not in bins:
                bins[l] = []
            bins[l].append(i)

        return bins  # ,max_document_length

    def reset(self):
        """new iteration"""
        self.batch_index = 0

        # randomly shuffle the question indices in each bin
        if self.shuffle:
            for question_indices in self.bins.values():
                random.shuffle(question_indices)

        # construct a list of mini-batches where each batch
        # is a list of question indices
        # questions within the same batch have identical max
        # document length
        self.batch_pool = []
        for max_document_length, question_indices in self.bins.items():
            number_of_questions = len(question_indices)
            k = number_of_questions / self.batch_size if \
                number_of_questions % self.batch_size == 0 else number_of_questions / self.batch_size + 1
            question_index_list = [(question_indices[self.batch_size * i:
                                   min(number_of_questions, self.batch_size * (i + 1))], max_document_length)
                                   for i in range(int(k))]
            self.batch_pool += question_index_list

        # randomly shuffle the mini-batches
        if self.shuffle:
            random.shuffle(self.batch_pool)

    def __next__(self):
        """load the next batch"""
        if self.batch_index == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        question_indices = self.batch_pool[self.batch_index][0]
        current_batch_size = len(question_indices)

        current_max_document_length = self.max_document_length

        # document words
        document_array = np.zeros(
            (current_batch_size, current_max_document_length),
            dtype='int32')
        # query words
        query_array = np.zeros(
            (current_batch_size, self.max_query_length),
            dtype='int32')
        # document word mask
        document_mask_array = np.zeros(
            (current_batch_size, current_max_document_length),
            dtype='int32')
        # query word mask
        query_mask_array = np.zeros(
            (current_batch_size, self.max_query_length),
            dtype='int32')
        # Answer mask array
        answer_mask_array = np.zeros(
            (current_batch_size, self.max_document_length),
            dtype='int32')
        # The answer contains start- and end-index
        answer_array = np.zeros((current_batch_size, 2), dtype='int32')

        filenames = [''] * current_batch_size

        types = {}

        for n, question_index in enumerate(question_indices):
            current_document, current_query, current_answer,\
                current_document_chars, current_query_chars, current_answer_start,\
                current_answer_end, current_filename = self.questions[question_index]

            # document, query
            document_array[n, : len(current_document)] = np.array(current_document)
            query_array[n, : len(current_query)] = np.array(current_query)

            document_mask_array[n, : len(current_document)] = 1
            query_mask_array[n, : len(current_query)] = 1
            answer_mask_array[n, int(current_answer_start):int(current_answer_end)] = 1

            # Collecting unique document words (in characters)
            # marking which question they came from in the batch
            # and which position they take in the document.
            for index, word in enumerate(current_document_chars):
                word_tuple = tuple(word)
                if word_tuple not in types:
                    types[word_tuple] = []
                types[word_tuple].append((0, n, index))

            # Collecting unique query words (in characters)
            # marking which question they came from in the batch
            # and which position they take in the query.
            for index, word in enumerate(current_query_chars):
                word_tuple = tuple(word)
                if word_tuple not in types:
                    types[word_tuple] = []
                types[word_tuple].append((1, n, index))

            # Store the indices of the answer in the paragraph
            answer_array[n, :] = np.array([current_answer_start, current_answer_end])

            filenames[n] = current_filename

        # create type character matrix and indices for document, query
        # document token index
        document_character_array = np.zeros(
            (current_batch_size, current_max_document_length),
            dtype='int32')
        # query token index
        query_character_array = np.zeros(
            (current_batch_size, self.max_query_length),
            dtype='int32')
        # type characters
        type_character_array = np.zeros(
            (len(types), self.max_word_length),
            dtype='int32')
        # type mask
        type_character_mask = np.zeros(
            (len(types), self.max_word_length),
            dtype='int32')

        type_index = 0  # The index of the word in the unique word dictionary (types)
        for word, markers in types.items():
            type_character_array[type_index, : len(word)] = np.array(word)
            type_character_mask[type_index, : len(word)] = 1
            # batch_index -> which example in the current batch the word was first found in
            for (is_query, batch_index, index_in_text) in markers:
                if is_query == 0:  # If it is a word found in a document
                    document_character_array[batch_index, index_in_text] = type_index
                else:              # Else, it is a word from a query
                    query_character_array[batch_index, index_in_text] = type_index

            type_index += 1

        self.batch_index += 1

        return document_array, document_character_array, query_array, query_character_array,\
            answer_array, document_mask_array, query_mask_array, answer_mask_array,\
            type_character_array, type_character_mask, filenames
