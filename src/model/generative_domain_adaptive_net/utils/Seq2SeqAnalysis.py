"""
This class is used for restoring Seq2Seq models and printing predictions.
"""
import tensorflow as tf
import os

from utils.DataPreprocessor import DataPreprocessor
from utils.UnlabeledDataPreprocessor import UnlabeledDataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader
from model.seq2seq_model import Seq2Seq
from nltk.tokenize.moses import MosesDetokenizer


class Analyser:
    def __init__(self):
        self.use_chars = False
        self.answer_injection = True

    def get_predictions(self, max_example, model_name, training_set, epoch, test_run=True,
                        unlabeled=True, save_dir=None):
        """
        This method takes some question indices as input and prints out
        the paragraphs, questions, ground truth and predicted questions as text.
        """

        # Process data and create batch_loader for prediction
        data_dir = "/scratch/s161027/ga_reader_data/ssqa_processed"
        vocab_size = 10000

        if unlabeled:
            unlabeled_set = "small"
            dp = UnlabeledDataPreprocessor()
            data = dp.preprocess(data_dir, unlabeled_set, training_set, vocab_size,
                                 max_example=max_example, use_chars=False,
                                 only_test_run=test_run)
            batch_loader_input = data.training
        else:
            dp = DataPreprocessor()
            data = dp.preprocess(
                data_dir,
                training_set,
                vocab_size,
                max_example=max_example,
                use_chars=False,
                only_test_run=test_run)
            batch_loader_input = data.test

        dictionary = data.dictionary

        batch_loader = MiniBatchLoader(batch_loader_input, 32, dictionary[0], shuffle=True,
                                       prediction_only=True)
        batch_number = batch_loader.__len__()

        # Logging
        if test_run:
            print("Loaded and processed data with {} examples. Number of batches is {}"
                  .format(max_example, batch_number))
        else:
            print("Loaded and processed data with all examples. Number of batches is {}"
                  .format(max_example, batch_number))

        # Restore Model
        default_path = "/scratch/s161027/run_data/SSQA/COMBINED_RUNS/"
        if save_dir is None:
            save_dir = os.path.join(default_path, "saved_models")

        # Restore model and graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        Seq2Seq.restore(self, sess, save_dir, model_name, epoch)

        # Numeric output from the prediction is a list of tuples like:
        # (document, query, answer, predicted_question)
        # Check the 'predict' method in seq2seq_model.py for exact output.
        with sess:
            # Logging
            print("Generating predictions...")
            numeric_output = Seq2Seq.predict(self, sess, batch_loader, unlabeled)

            # Get the first index of the output
            # since there should be only one batch in case of prediction
            if not unlabeled:
                numeric_output = [numeric_output[0]]

        if unlabeled:
            return numeric_output, data
        # Based on the dictionary, turn the embeddings back into text
        # Text output is a tuple such as:
        # (document, query, ground_truth, predicted_answer)
        # Where the elements of this tuple are lists of strings.
        text_output = []
        for numeric_batch in numeric_output:
            text_output.append(self.inverse_dictionary(numeric_batch,
                                                       dictionary))

        return text_output, numeric_output

    def inverse_dictionary(self, numeric_input, dictionary):
        """
        This method will find the string
        representation of the numeric values of the input tuple in the input
        dictionary
        """
        detokenizer = MosesDetokenizer()

        word_dictionary = dictionary[0]
        inv_word_dictionary = dict(zip(word_dictionary.values(),
                                       word_dictionary.keys()))

        # Masking doc rows [:-1] to get rid of @end markers
        doc = [[inv_word_dictionary[num] for num in doc_row[doc_row != 0][:-1]]
               for doc_row in numeric_input[0]]
        # Masking qry_rows [1:] to get rid of @begin markers
        qry = [[inv_word_dictionary[num] for num in qry_row[qry_row != 0][1:]]
               for qry_row in numeric_input[1]]
        # Predicted question - masking out the padding (index=0) and end of seq. markers (index=2)
        # predicted_question = []
        # for pred_qry_row in numeric_input[3]:
        #     for num in pred_qry_row:
        #         if num != 0 and num != 2:
        #             predicted_question.append(inv_word_dictionary[num])

        predicted_question = \
            [[inv_word_dictionary[num] for num in pred_qry_row if num != 0 and num != 2]
             for pred_qry_row in numeric_input[3]]

        ans = numeric_input[2]
        answer = [doc_row[ans[index][0]:ans[index][1]+1] for index, doc_row in enumerate(doc)]

        # Mark answer with capitalization in document
        doc_marked = []
        for index, row in enumerate(doc):
            answer_start = ans[index][0]
            answer_end = ans[index][1]
            current_answer = row[answer_start:answer_end+1]
            doc_marked_row = row[:answer_start] + [words.upper() for words in current_answer] +\
                row[answer_end+1:]
            doc_marked.append(doc_marked_row)

        # Detokenize the lists of words into strings
        doc_text = [detokenizer.detokenize(text, return_str=True) for text in doc_marked]
        qry_text = [detokenizer.detokenize(text, return_str=True) for text in qry]
        answer_text = [detokenizer.detokenize(text, return_str=True) for text in answer]
        predicted_question = [detokenizer.detokenize(text, return_str=True) for text in predicted_question]

        return doc_text, qry_text, answer_text, predicted_question
