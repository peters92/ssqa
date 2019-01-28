import numpy as np
import os
import collections
import string
import re

EMBED_DIM = 128
MAX_WORD_LEN = 10
SYMB_PLACEHOLDER = "@placeholder"
SYMB_BEGIN = "@begin"
SYMB_END = "@end"
SYMB_PAD = "@pad"
SYMB_UNK = "@unk"

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []

    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def calculate_accuracies(answer_array, predicted_answer_array, document_array, word_dictionary):
    """
    This method calculates the F1 score and the Exact Match (EM) accuracy over the input batch.
    Inputs:
    answer_array - The start and end indices of the ground truth answers
    predicted_answer_array - The start and end indices of the predicted answers
    document_array - Document tokens for each example in the batch
    Output:
    f1_score - The harmonic mean of precision and recall over the current batch, where:
    Precision = True Positives / (True Positives + False Positives)
    Recall    = True Positives / (True Positives + False negatives)
    f1_score  = 2 * (Precision * Recall) / (Precision + Recall)

    Exact Match Accuracy, which is:
    1 - if answer == prediction
    0 otherwise

    Both measures of accuracy disregard:
    Capitalization              - "CAT"     == "cat"
    Punctuation                 - ".cat,"   == "cat"
    Articles ("a", "an", "the") - "the cat" == "cat"
    White space                 - "   cat " == "cat"
    """

    # Create the inverse dictionary, which maps numbers to (string) words
    inv_dictionary = dict(zip(word_dictionary.values(), word_dictionary.keys()))

    # Initialize batch f1 score
    f1_score_batch = 0
    exact_match_batch = 0

    # Loop over examples in the batch, calculate f1 score for each and accumulate it
    for i in range(answer_array.shape[0]):
        current_answer_indices = answer_array[i]
        current_prediction_indices = predicted_answer_array[i]
        current_document = document_array[i]

        current_answer = current_document[current_answer_indices[0]:current_answer_indices[1]+1]
        current_answer_string = " ".join([inv_dictionary[words] for words in current_answer])

        current_prediction = \
            current_document[current_prediction_indices[0]:current_prediction_indices[1] + 1]
        current_prediction_string = " ".join([inv_dictionary[words] for words in
                                              current_prediction])

        f1_score_batch += compute_f1(current_answer_string, current_prediction_string)
        exact_match_batch += compute_exact(current_answer_string, current_prediction_string)

        # if i == 0:
        #     inv_document = " ".join([inv_dictionary[words] for words in current_document])
        #     print("\n\n{:-^100}".format(" DEBUG {} ".format(i)))
        #     print("doc, answer and prediction numbers: {}, {}, {}".format(current_document,
        #                                                                   current_answer,
        #                                                                   current_prediction))
        #     # print("Document: {}".format(inv_document))
        #     print("Answer string: {}".format(current_answer_string))
        #     print("Prediction string: {}\n".format(current_prediction_string))
        #     print("F1 and EM accuracy: {}, {}".format(compute_f1(current_answer_string,
        #                                                          current_prediction_string),
        #                                               compute_exact(current_answer_string,
        #                                                             current_prediction_string)))

    # Calculate averages for exact match accuracy and f1 score by dividing with the
    # number of examples in the current batch

    f1_score_batch /= document_array.shape[0]
    exact_match_batch /= document_array.shape[0]

    return f1_score_batch, exact_match_batch


def load_word2vec_embeddings(dictionary, vocab_embed_file):
    if vocab_embed_file is None:
        return None, EMBED_DIM

    fp = open(vocab_embed_file, encoding='utf-8')

    info = fp.readline().split()
    embed_dim = int(info[1])
    # vocab_embed: word --> vector
    vocab_embed = {}
    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(
            list(map(float, line[1:])), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for word, value in dictionary.items():
        if word in vocab_embed:
            W[value, :] = vocab_embed[word]
            n += 1
    print("{}/{} vocabs are initialized with word2vec embeddings."
          .format(n, vocab_size))
    return W, embed_dim


def check_dir(*args, exit_function=False):
    """
    check the existence of directories
    Args:
    - args: (list) paths of directories
    - exit_function: (bool) action to take
    """
    for dir_ in args:
        if not os.path.exists(dir_):
            if not exit_function:
                os.makedirs(dir_)
            else:
                raise ValueError("{} does not exist!".format(dir_))


def batch_splitter(batched_data):
    """
    This method splits the input batch in two and returns the split as a tuple.
    The purpose of the method is to help the training/validation process when it
    exhausts the GPU memory with too large information. The split batch can be fed
    to the model again.
    :param batched_data: This can be a batch of training/validation/test data
    :return: tuple of split batched data
    """
    batch_split_1, batch_split_2 = [], []
    batch_size_split = batched_data[0].shape[0]//2
    for arrays in batched_data:
        batch_split_1.append(arrays[0:batch_size_split])
        batch_split_2.append(arrays[batch_size_split:])

    return batch_split_1, batch_split_2
