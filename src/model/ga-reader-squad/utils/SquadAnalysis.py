"""
This class is used for visualizing model performance on the SQuAD v1.1 data set.
"""
import tensorflow as tf
import os

from utils.DataPreprocessor import DataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader
from model.GAReader import GAReader
from nltk.tokenize.moses import MosesDetokenizer


class Analyser:
    def __init__(self):
        self.use_cloze_style = False
        self.use_chars = False

    def print_predictions(self, max_example, epoch, save_dir=None):
        """
        This method takes some question indices as input and prints out
        the paragraphs, questions, ground truth and predicted answers as text.
        """

        # Process data and create batch_loader for prediction
        data_dir = "/scratch/s161027/ga_reader_data/squad"

        # Multiplying by 5, since each paragraph has 5 questions, so this way there will
        # be as many examples as the max_example input says.
        max_example = max_example * 5
        dp = DataPreprocessor()
        data = dp.preprocess(
            question_dir=data_dir,
            max_example=max_example,
            use_chars=False,
            use_cloze_style=False,
            only_test_run=False)

        batch_loader = MiniBatchLoader(data.test, 32, shuffle=False,
                                       use_cloze_style=False)
        batch_number = batch_loader.__len__()

        # Restore Model
        # default_path = "/scratch/s161027/run_data/first_working_squad_model"
        default_path = "/scratch/s161027/run_data/ga_reader_squad_reserve"
        if save_dir is None:
            save_dir = os.path.join(default_path, "saved_models")

        sess = tf.Session()
        GAReader.restore(self, sess, save_dir, epoch)  # Restore model and graph

        with sess:
            numeric_output = GAReader.predict(self, sess, batch_loader)

        # Based on the dictionary, turn the embeddings back into text
        text_output = self.inverse_dictionary(numeric_output,
                                              data.dictionary)
        # output is a list of tuples like:
        # (doc, qry, answer, pred_ans, start_probs, end_probs)
        return text_output, batch_number

    def inverse_dictionary(self, data, dictionary):
        """
        For each tuple in the input data, this method will find the string
        representation of the numeric values in the tuple in the passed
        dictionary
        """
        detokenizer = MosesDetokenizer()

        word_dictionary = dictionary[0]
        inv_word_dictionary = dict(zip(word_dictionary.values(),
                                       word_dictionary.keys()))

        text_output = []
        for batches in data:
            # Create
            doc = [[inv_word_dictionary[num] for num in doc_row[doc_row != 0]] for doc_row in batches[0]]
            # Masking qry_rows [1:-1] to get rid of @begin and @end markers
            qry = [[inv_word_dictionary[num] for num in qry_row[qry_row != 0][1:-1]] for qry_row in batches[1]]

            ans = batches[2]
            pred_ans = batches[3]

            ground_truth = [doc_row[ans[index][0]:ans[index][1]+1] for index, doc_row in enumerate(doc)]
            predicted_answer = [doc_row[pred_ans[index][0]:pred_ans[index][1]+1] for index, doc_row in enumerate(doc)]

            # Detokenize the lists of words into strings
            doc = [detokenizer.detokenize(text, return_str=True) for text in doc]
            qry = [detokenizer.detokenize(text, return_str=True) for text in qry]
            ground_truth = [detokenizer.detokenize(text, return_str=True) for text in ground_truth]
            predicted_answer = [detokenizer.detokenize(text, return_str=True) for text in predicted_answer]

            text_output.append((doc, qry, ground_truth, predicted_answer))

        return text_output
