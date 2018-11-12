import numpy as np
import nltk
import json
from tqdm import trange
import os
import argparse

'''
This script reads the SQuAD 1.1 dataset as JSON then outputs each sample as a separate .question file.
These files will each contain a paragraph, question and an answer, which can then be used by the Gated Attention Reader.

'''


def squad_parser(test_run=False):
    data_path = "data"
    data_dest = "squad"

    data_path_full = os.path.join(os.getcwd(), data_path, "SQuAD-v1.1.json")
    data_file = open(data_path_full, encoding="utf8")
    json_file = json.loads(data_file.read())
    data = json_file[1][1:]  # The list containing all the sample paragraphs and question/answer pairs

    # Directory for the separated question files
    data_dir = os.path.join(data_path, data_dest)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    N = len(data)

    test_ratio = 0.1  # This portion of the training data will be used for testing
    np.random.seed(42)
    test_array = np.random.rand(N) < test_ratio

    # Progress bar
    if test_run:
        N = 900
    tr = trange(N, desc="Parsing dataset", leave=True, ncols=100, ascii=True)
    for index, sample in enumerate(data):


        paragraph = sample[1][2].replace("'", "")  # The paragraph text
        paragraph = paragraph.replace("\n", " ")  # Replacing new lines with space to have paragraph as a one-line string
        qa_set = sample[2][2][1:]                # The set of all (5) question/answer pairs relating to the paragraph
        validation_role = \
            sample[-1][2].replace("'", "").lower()   # text marking if the sample is training or validation data
        if test_array[index] == 1:  # Overwriting role if index is in test set
            validation_role = "test"

        # Replace validation role during test run to limit number of samples
        if test_run:
            if index == N:
                break
            if index < N/3:
                validation_role = "training"
            elif index < 2*N/3:
                validation_role = "validation"
            else:
                validation_role = "test"

        # Creating/Using the folder for training/validation samples
        file_dir = os.path.join(data_dir, validation_role)
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        for qa_pair in qa_set:  # for each question/answer pair we create a separate ".question" file
            question = qa_pair[1][2].replace("'", "")   # question text
            answer = qa_pair[2][2][1].replace("'", "")  # answer text
            answer_start = qa_pair[3][2][1]               # position of answer in paragraph, integer

            # Heuristic method for finding answer end positions in char space
            # We look for the answer in the paragraph starting from the answer start
            # character minus a window (to take care of cases where answer start was wrong)

            # Fix cases where answer_start was chosen to be too high
            try:
                paragraph[answer_start:].index(answer)
            except ValueError:
                answer_start = paragraph.index(answer)

            answer_end = answer_start + paragraph[answer_start:].index(answer) + len(answer)

            # Heuristic method for finding answer start and end index in token space
            # We look for the span of answer words in the paragraph that matches the
            # length of the actual answer span
            doc_tokens = nltk.word_tokenize(paragraph)
            answer_tokens = nltk.word_tokenize(answer)
            ans_mask = [1 if doc_tokens[index] in answer_tokens else 0
                        for index, value in enumerate(doc_tokens)]
            answer_start_token = 0
            answer_end_token = 0
            span = 0
            answer_span = len(answer_tokens)

            for idx, value in enumerate(ans_mask):
                if value == 1:
                    span += 1
                else:
                    span = 0
                if span == answer_span:
                    answer_end_token = idx
                    answer_start_token = idx - answer_span + 1
                    break

            question_id = qa_pair[4][2].replace("'", "")  # unique id for question

            # create .question file with the given question id as name
            file_name = file_dir + "/" + question_id + ".question"
            f = open(file_name, "w+", encoding="utf-8")
            f.write("\n\n")
            f.write(paragraph + "\n\n")
            f.write(question + "\n\n")
            f.write(answer + "\n\n")
            f.write(str(answer_start) + " " + str(answer_end) + "\n\n")
            f.write(str(answer_start_token) + " " + str(answer_end_token) + "\n")
            f.close()

        tr.update()
    tr.close()

    print("Dataset was parsed into {}".format(os.path.join(os.getcwd(), data_dir)))

