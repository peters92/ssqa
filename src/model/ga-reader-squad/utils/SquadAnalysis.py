"""
This class is used for visualizing model performance on the SQuAD v1.1 data set.
"""
import tensorflow as tf
import numpy as np
import svgwrite
import os

from utils.DataPreprocessor import DataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader
from model.GAReader import GAReader
from nltk.tokenize.moses import MosesDetokenizer


class Analyser:
    def __init__(self):
        self.use_chars = False

    def visualize_attention(self, num_examples, file_suffix=None):
        """
        This method generates colored svg's of paragraphs/questions according to the attention
        matrix values generated by the GAReader during prediction (forward-pass).
        """
        # Generate predictions
        text_outputs, _, attentions_and_probs, _ = \
            self.get_predictions(max_example=num_examples, epoch=14)

        ground_truth = text_outputs[2]
        predicted_answer = text_outputs[3]
        doc = text_outputs[4]  # Document text split into tokens
        qry = text_outputs[5]  # Query text split into tokens
        attentions = attentions_and_probs[2]  # Attention matrices (initial plus over K "hops")

        # Generate drawings
        for i in range(num_examples):
            curr_ground_truth = ground_truth[i].split()
            curr_predicted_answer = predicted_answer[i].split()
            curr_doc = doc[i]
            curr_qry = qry[i]
            curr_attentions = attentions[i]  # 4 attention matrices
            final_attention = curr_attentions[-1]
            # Get the average attention across query words
            final_attention_mean = np.mean(final_attention, axis=1)

            curr_file_suffix = file_suffix + "_{}".format(i)
            self.draw_svg(curr_doc, curr_qry, curr_ground_truth,
                          curr_predicted_answer,
                          final_attention_mean,
                          suffix=curr_file_suffix)

        return None

    def draw_svg(self, doc, qry, ground_truth, predicted_answer, attention, suffix):
        """
        This method prints the input doc, qry into a .svg file. The document paragraph is
        colored according to the input attention.
        The supplied svg file will be saved with the given suffix (default=None).
        """
        # Set parameters
        # Starting coordinates of text (from upper left corner) and gap between lines
        x_start, y_start, line_gap = (20, 70, 30)
        line_break_num = 20  # Text will be broken after this amount of tokens
        resolution = 100

        # Dark Style 1
        bg_color = (33, 33, 33)
        # gradient_start = (187, 222, 251)
        # gradient_end = (173, 20, 87)
        # Dark Style 2
        gradient_start = (174, 213, 129)
        gradient_end = (239, 108, 0)

        color_scale = self.generate_gradient(gradient_start, gradient_end, resolution)
        attention_color_index = self.generate_attention_color_index(attention, resolution)

        # Create svg drawing
        default_path = "/scratch/s161027/analysis/svg"
        filename = "visual_attention_"+suffix+".svg"
        filename = os.path.join(default_path, filename)
        dwg = svgwrite.Drawing(filename, profile='tiny')
        # Create background rectangle
        dwg.add(dwg.rect((0, 0), (1300, 700), fill=self.rgb_to_string(bg_color)))

        # Adding the paragraph to the image
        paragraph = dwg.text('', insert=(x_start, y_start - line_gap))
        paragraph.add(dwg.tspan("Paragraph:", font_size='1.3em', fill=self.rgb_to_string(gradient_start)))
        dwg.add(paragraph)

        paragraph_list = [(dwg.text('', insert=(x_start, y_start)))]

        line_count = 0
        line_y_coordinate = y_start
        for index, words in enumerate(doc):
            # Getting fill color based on attention value of tokens
            fill_color = (color_scale[0][attention_color_index[index]],
                          color_scale[1][attention_color_index[index]],
                          color_scale[2][attention_color_index[index]])
            fill_color = self.rgb_to_string(fill_color)

            # current_text = words + " ({:.2f}) ".format(attention[index])
            paragraph_list[line_count].add(dwg.tspan(words+" ", font_size='1.2em', fill=fill_color))
            if index % line_break_num == 0 and index > 0:
                line_count += 1
                line_y_coordinate = y_start + line_count * line_gap
                paragraph_list.append(dwg.text('', insert=(x_start, line_y_coordinate)))
        for lines in paragraph_list:
            dwg.add(lines)

        # Adding the query to the image
        # y-coordinate dynmically adjusts to the last line of paragraph
        last_y_coordinate_query = self.generate_text_lines(dwg, qry, x_start, line_y_coordinate,
                                                     line_gap, line_break_num, gradient_start,
                                                     title="Query:")
        # Adding ground truth to the image
        last_y_coordinate_gtruth = self.generate_text_lines(dwg, ground_truth, x_start, last_y_coordinate_query,
                                                     line_gap, line_break_num, gradient_start,
                                                     title="Ground truth:")
        # Adding predicted answer to the image
        _ = self.generate_text_lines(dwg, predicted_answer, x_start, last_y_coordinate_gtruth,
                                     line_gap, line_break_num, gradient_start,
                                     title="Predicted answer:")

        # Gradient testing
        # Create a vertical linear gradient and add it the svg's definitions
        vert_grad = svgwrite.gradients.LinearGradient(start=(0, 1), end=(0, 0), id="vert_lin_grad")
        vert_grad.add_stop_color(offset='0%', color=self.rgb_to_string(gradient_start), opacity=None)
        # vert_grad.add_stop_color(offset='50%', color='green', opacity=None)
        vert_grad.add_stop_color(offset='100%', color=self.rgb_to_string(gradient_end), opacity=None)
        dwg.defs.add(vert_grad)

        # draw a box and reference the above gradient definition by #id
        dwg.add(dwg.rect((1200, 50), (75, 600),
                         stroke=svgwrite.rgb(10, 10, 16, '%'),
                         fill="url(#vert_lin_grad)"
                         ))

        dwg.save()

        return None

    def get_predictions(self, max_example, epoch, save_dir=None):
        """
        This method takes some question indices as input and prints out
        the paragraphs, questions, ground truth and predicted answers as text.
        """

        # Process data and create batch_loader for prediction
        data_dir = "/scratch/s161027/ga_reader_data/squad"

        dp = DataPreprocessor()
        data = dp.preprocess(
            question_dir=data_dir,
            max_example=max_example,
            use_chars=False,
            only_test_run=False)

        batch_loader = MiniBatchLoader(data.test, 128, shuffle=False, prediction_only=True)
        batch_number = batch_loader.__len__()

        # Logging
        print("Loaded and processed data with {} examples. Number of batches is {}"
              .format(max_example, batch_number))

        # Restore Model
        # default_path = "/scratch/s161027/run_data/first_working_squad_model"
        # default_path = "/scratch/s161027/run_data/temp_test_delete"
        # default_path = "/scratch/s161027/run_data/ga_reader_squad_reserve"
        default_path = "/scratch/s161027/run_data/visualization_test"
        if save_dir is None:
            save_dir = os.path.join(default_path, "saved_models")

        sess = tf.Session()
        GAReader.restore(self, sess, save_dir, epoch)  # Restore model and graph

        # Numeric output from the prediction is a list of tuples like:
        # (doc, qry, answer, pred_ans, start_probs, end_probs, attention_tensors)
        # Check the 'predict' method in GAReader.py for exact output.
        with sess:
            # Logging
            print("Generating predictions...")
            numeric_output = GAReader.predict(self, sess, batch_loader)
            # Get the first (and only) index of the output
            # since there should be only one batch in case of prediction
            numeric_output = numeric_output[0]

        # Based on the dictionary, turn the embeddings back into text
        # Text output is a tuple such as:
        # (doc, qry, ground_truth, predicted_answer)
        # Where the elements of this tuple are lists of strings.
        text_output = self.inverse_dictionary(numeric_output,
                                              data.dictionary)

        # Attentions_and_probs is a tuple such as:
        # (start_probs_list, end_probs_list, attentions_list)
        # Where each element of the tuple is masked to be the same shape as its respective document and query.
        # As opposed to having the shape: [max_document_length] or [max_document_length x max_query_length] in case
        # of the attention matrices
        attentions_and_probs = self.mask_numeric_output(numeric_output)

        return text_output, numeric_output, attentions_and_probs, batch_number

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

        doc = [[inv_word_dictionary[num] for num in doc_row[doc_row != 0]] for doc_row in numeric_input[0]]
        # Masking qry_rows [1:-1] to get rid of @begin and @end markers
        qry = [[inv_word_dictionary[num] for num in qry_row[qry_row != 0][1:-1]] for qry_row in numeric_input[1]]
        # The tokenized document and query is saved separately
        doc_tokenized = doc
        qry_tokenized = qry

        ans = numeric_input[2]
        pred_ans = numeric_input[3]

        ground_truth = [doc_row[ans[index][0]:ans[index][1]+1] for index, doc_row in enumerate(doc)]
        predicted_answer = [doc_row[pred_ans[index][0]:pred_ans[index][1]+1] for index, doc_row in enumerate(doc)]

        # Detokenize the lists of words into strings
        doc = [detokenizer.detokenize(text, return_str=True) for text in doc]
        qry = [detokenizer.detokenize(text, return_str=True) for text in qry]
        ground_truth = [detokenizer.detokenize(text, return_str=True) for text in ground_truth]
        predicted_answer = [detokenizer.detokenize(text, return_str=True) for text in predicted_answer]

        return doc, qry, ground_truth, predicted_answer, doc_tokenized, qry_tokenized

    def mask_numeric_output(self, numeric_input):
        """
        This method takes the numeric output from a GAReader prediction containing doc, qry,
        answer start and end index probabilities and attention matrices.
        According to the document and query size (the non-zero part),
        it masks the attention matrices to be in shape: [doc_size x query_size]
        Whereas the original attention matrix shape is: [N x Q] (max. doc. by max. query length)
        The answer start and end index probabilities are also masked the same way by the actual
        document length.
        """
        # for batch in numeric_input:
        doc = numeric_input[0]          # Embedded documents, shape: [batch_size x max_document_length]
        qry = numeric_input[1]          # Embedded queries, shape: [batch_size x max_query_length]
        start_probs = numeric_input[4]  # Vectors of answer start probabilities, shape [batch_size x max_doc_length]
        end_probs = numeric_input[5]    # Vectors of answer end probabilities, shape [batch_size x max_doc_length]
        # Collection of pairwise attentions, shape [batch_size x 4]
        # Where the second dimension (4) contains matrices of shape [max_doc_length x max_qry_length]
        attentions = numeric_input[6]

        # Gather actual document and query lengths then mask the attention
        # and probability matrices according to those lengths
        start_probs_list = []
        end_probs_list = []
        attentions_list = []
        # Looping through examples in the current batch
        for index, row in enumerate(doc):
            doc_row = row
            qry_row = qry[index, :]

            # Get the non-zero indices of documents and queries
            doc_mask = doc_row != 0
            qry_mask = qry_row != 0

            # Mask probability vectors to be the same length as their respective documents
            start_probs_list.append(start_probs[index, doc_mask])
            end_probs_list.append(end_probs[index, doc_mask])

            # Mask attention matrices to be the same size as their respective documents and queries
            current_attentions = attentions[:, index, :, :]  # This is now an array of 4 attention matrices

            attention_matrices = []
            # Loop through the 4 attention matrices and mask them
            for i in range(current_attentions.shape[0]):
                matrix = current_attentions[i, :, :]
                matrix = matrix[doc_mask, :]
                matrix = matrix[:, qry_mask]
                attention_matrices.append(matrix)

            attentions_list.append(attention_matrices)

        return start_probs_list, end_probs_list, attentions_list

    # Helper functions of draw_svg
    def rgb_to_string(self, rgb_tuple):
        """
        Transform input tuple of rgb values (0-255) into a string in the form:
        "rgb(red_val, green_val, blue_val)"
        """
        red_val = int(round(rgb_tuple[0]))
        green_val = int(round(rgb_tuple[1]))
        blue_val = int(round(rgb_tuple[2]))
        rgb_str = "rgb(" + str(red_val) + "," + str(green_val) + "," + str(blue_val) + ")"
        return rgb_str

    def generate_gradient(self, grad_start, grad_end, resolution):
        """
        Based on input colors, generates a gradient color scale between the start and end color with
        as many elements as defined by resolution.
        """
        red_grad = np.linspace(grad_start[0], grad_end[0], resolution)
        green_grad = np.linspace(grad_start[1], grad_end[1], resolution)
        blue_grad = np.linspace(grad_start[2], grad_end[2], resolution)
        return np.array((red_grad, green_grad, blue_grad))

    def generate_attention_color_index(self, attention, resolution):
        """
        Given an input attention vector, creates a linear space between min and max values of the vector.
        Then finds the index of the attention values on the created scale. This is used to help index a gradient
        color scale.
        """
        min_attention = np.min(attention)
        max_attention = np.max(attention)
        attention_scale = np.linspace(min_attention, max_attention, resolution)

        attention_color_index = \
            [np.argmin(np.abs(attention_scale-attention_value)) for attention_value in attention]

        return attention_color_index

    def generate_text_lines(self, drawing, word_list, x_start, y_start, line_gap,
                            line_breaks=16, text_color=(255, 255, 255),
                            title="default_title"):

        y_coordinate = y_start + line_gap * 1.25
        text_lines = drawing.text('', insert=(x_start, y_coordinate))
        text_lines.add(drawing.tspan(title, font_size='1.3em', fill=self.rgb_to_string(text_color)))
        drawing.add(text_lines)

        text_list = [(drawing.text('', insert=(x_start, y_coordinate + line_gap)))]
        line_count = 0
        last_y_coordinate = y_coordinate + line_gap
        for index, words in enumerate(word_list):
            fill_color = self.rgb_to_string(text_color)

            text_list[line_count].add(drawing.tspan(words + ' ',
                                      font_size='1.2em',
                                      fill=fill_color))
            if index % line_breaks == 0 and index > 0:
                line_count += 1
                last_y_coordinate = y_coordinate + (line_count * line_gap)
                text_list.append(drawing.text('',
                                 insert=(x_start, last_y_coordinate)))

        for lines in text_list:
            drawing.add(lines)

        return last_y_coordinate