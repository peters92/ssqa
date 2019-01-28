import nltk
from tqdm import trange
import os
import random
import time
import json
import pickle
import msgpack
from utils.Helpers import SYMB_PLACEHOLDER, \
                          SYMB_BEGIN,\
                          SYMB_END, \
                          SYMB_PAD, \
                          SYMB_UNK
MAX_WORD_LEN = 10
MWETokenizer = nltk.tokenize.MWETokenizer


class UnlabeledData:

    def __init__(self, dictionary, num_entities, training):
        self.dictionary = dictionary
        self.training = training
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v: k for k, v in dictionary[0].items()}


class UnlabeledDataPreprocessor:
    def __init__(self):
        self.removed_questions = []
        self.num_removed_questions = 0
        # Multi-word tokenizer for fixing placeholders in cloze-style data
        # This is to join "@" and "placeholder" as one token.
        self.tokenizer = MWETokenizer([('@', 'placeholder')], separator='')

    def preprocess(self, question_dir, unlabeled_set, training_set, vocab_size, max_example=100,
                   use_chars=True, only_test_run=False):
        """
        preprocess all data into a standalone Data object.
        """
        # Define files to be used
        dataset_files = {"small": "unlabeled-data.small",
                         "large": "unlabeled-data.large"}
        # Define vocabulary source file
        vocab_filename = "vocab_training" + training_set + "+" +\
                         "unlabeled_" + unlabeled_set + ".bin"
        vocab_f = os.path.join(question_dir, vocab_filename)

        # Generate or load dictionaries
        raw_file_dir = '/scratch/s161027/ga_reader_data/ssqa'
        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(dataset_files, raw_file_dir, unlabeled_set, vocab_size,
                                 vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)

        # Check for the existence of parsed data in binary files
        # Load them if they exist. Otherwise generate and save new ones.
        print("Preparing data...")

        path = os.path.join(question_dir, dataset_files[unlabeled_set])+".bin"

        # Parse original json file and save results in binary, then load with pickle
        if not os.path.exists(path):
            print("Can't find binary data for {} set. Parsing raw data...".format(unlabeled_set))
            self.json_parser(raw_file_dir, dataset_files[unlabeled_set], dictionary, use_chars)

        start_time = time.time()
        print("Loading binary data for {} set".format(unlabeled_set))
        infile = open(path, "rb")
        # training = pickle.load(infile)
        training = msgpack.unpack(infile)
        unpack_time = time.time() - start_time

        print("Unpacking took {}".format(unpack_time))

        if only_test_run:
            training = random.sample(training, max_example)
            # training = training[0:max_example]

        data = UnlabeledData(dictionary, num_entities, training)

        # Message about bad samples being removed.
        print("{} questions were removed due to bad formatting."
              .format(self.num_removed_questions))

        return data

    def make_dictionary(self, dataset_files, raw_file_dir, unlabeled_set, vocab_size, vocab_file):

        # First we check for a saved vocabulary binary file, that contains a list of the unique
        # words in the dataset sorted from most frequent to last. If it exists, then load it.
        # If it doesn't, then generate a new one.

        if os.path.exists(vocab_file):
            print("Loading vocabularies from " + vocab_file + " ...")

            infile = open(vocab_file, 'rb')
            vocab_list = pickle.load(infile)
        else:
            print("No vocab file found on the following path:\n" + vocab_file)
            # New dictionary generation
            # Initialize list containing all tokens from all files
            text = []
            # Collect all tokens in json files into one list
            raw_file_path = os.path.join(raw_file_dir, dataset_files[unlabeled_set]) + ".json"
            data_file = open(raw_file_path, encoding="utf8")
            json_file = json.loads(data_file.read())
            # List of all the paragraph/question/answer triplets in the data
            all_topics = json_file["data"]

            tr = trange(len(all_topics),
                        desc="Collecting tokens from {} set for vocabulary".format(unlabeled_set),
                        leave=True, ncols=100, ascii=True)

            for topic in all_topics:
                topic = topic["paragraphs"]
                for index, sample in enumerate(topic):
                    paragraph = sample["context"].replace("'", "")  # The paragraph text
                    # Replacing new lines with space to have paragraph as a one-line string
                    paragraph = paragraph.replace("\n", " ")
                    # Extend token list with tokens from the current paragraph
                    doc_tokens = nltk.word_tokenize(paragraph)

                    # Limiting doc_tokens to 512 max length to fit GPU memory with higher
                    # batch sizes. Comment this part out if memory is not an issue
                    if len(doc_tokens) > 512:
                        doc_tokens = doc_tokens[0:512]
                    text.extend(doc_tokens)
                    # The set of all (5) question/answer pairs relating to the paragraph
                    qa_set = sample["qas"]
                    for qa_pair in qa_set:
                        question = qa_pair["question"].replace("'", "")  # question text
                        answer = qa_pair["answers"][0]["text"].replace("'", "")  # answer text
                        # Extend token list with tokens from the current question and answer
                        text.extend(nltk.word_tokenize(question))
                        text.extend(nltk.word_tokenize(answer))

                tr.update()
            tr.close()

            # Initialize word frequency dictionary as {"word": (frequency count)}
            word_frequency = {}

            tr = trange(len(text), desc="Constructing token frequency list",
                        leave=True, ncols=120, ascii=True)
            # Loop through list, store unique tokens and count their frequency
            for token in text:
                try:  # If the token is already in the dictionary, increment frequency
                    word_frequency[token] += 1
                except KeyError:  # Else, it's a new token, create key/value pair.
                    word_frequency[token] = 1
                tr.update()
            tr.close()

            # Sort dictionary by token frequency
            sorted_word_frequency_dict = sorted(word_frequency, key=word_frequency.get, reverse=True)
            # Get the N-5 most frequent tokens (N = vocab_size)
            # -5 is to leave space for special symbols such as @pad, @unk etc.,
            # while keeping the vocab size exactly as defined by the vocab_size argument.
            vocab_list = sorted_word_frequency_dict[0:vocab_size-5]

            special_vocab = [SYMB_PAD, SYMB_BEGIN, SYMB_END, SYMB_PLACEHOLDER, SYMB_UNK]

            # Adding special symbols together with the rest of the tokens
            vocab_list = special_vocab + vocab_list

            print("Dumping vocabulary to binary file: {}".format(vocab_file))
            outfile = open(vocab_file, "wb")
            pickle.dump(vocab_list, outfile)
            outfile.close()

        # Check vocab_size
        vocab_size_check = len(vocab_list)
        assert vocab_size_check == vocab_size, \
            "Mismatch between defined and actual vocabulary size.\
            \nDefined vocab size: {}\nActual:{}".format(vocab_size, vocab_size_check)

        word_dictionary = dict(zip(vocab_list, range(vocab_size)))
        char_set = set([c for w in vocab_list for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocab_list if v.startswith('@entity')])
        print("vocab_size = {}".format(vocab_size))
        print("num characters = {}".format(len(char_set)))
        print("{} anonymized entities".format(num_entities))
        print("{} other tokens (including @placeholder, {}, {}, {} and {})".format(
            vocab_size - num_entities, SYMB_BEGIN, SYMB_END, SYMB_PAD, SYMB_UNK))

        return word_dictionary, char_dictionary, num_entities

    def json_parser(self, data_path, file_name, dictionaries, use_chars):
        """
        This method opens the raw json dataset files and parses them into a list of tuples as:
        [(document_words, query_words, answer, document_characters, query_character_list, \
            answer_start, answer_end)]
        :param data_path: The path to the raw data files (.json)
        :param file_name: The name of the file to parse
        :param dictionaries: Word/character dictionaries to parse text into numbers
        :return: Nothing, but saves a list of tuples in a binary file
        """
        data_path = os.path.join(data_path, file_name+".json")
        data_file = open(data_path, encoding="utf8")
        json_file = json.loads(data_file.read())
        # List of all the paragraph/question/answer triplets in the data
        all_topics = json_file["data"]

        # Initialize variables for holding documents, questions, answers
        # document_words, query_words, answer, document_characters, query_character_list, \
        # answer_start, answer_end
        dataset_list = []
        word_dictionary, character_dictionary = dictionaries[0], dictionaries[1]

        debug_slice_count = 0
        doc_count = 0
        # Progress bar
        tr = trange(len(all_topics), desc="Parsing dataset", leave=True, ncols=100, ascii=True)
        for topic in all_topics:
            topic = topic["paragraphs"]
            for index, sample in enumerate(topic):
                paragraph = sample["context"].replace("'", "")  # The paragraph text
                # Replacing new lines with space to have paragraph as a one-line string
                paragraph = paragraph.replace("\n", " ")
                qa_set = sample[
                    "qas"]  # The set of all (5) question/answer pairs relating to the paragraph
                doc_tokens = nltk.word_tokenize(paragraph)

                # Limiting doc_tokens to 512 max length to fit GPU memory with higher batch sizes
                # Comment this part out if memory is not an issue
                doc_count += 1
                if len(doc_tokens) > 256:
                    # Limit to 511 to leave space for wrapping symbol
                    doc_tokens = doc_tokens[0:255]
                    debug_slice_count += 1
                # Wrapping end of document with end of sequence symbol
                doc_tokens.append(SYMB_END)

                # for each question/answer pair we create a separate ".question" file
                for qa_pair in qa_set:
                    question = qa_pair["question"].replace("'", "")  # question text
                    answer = qa_pair["answers"][0]["text"].replace("'", "")  # answer text
                    answer_start = qa_pair["answers"][0][
                        "answer_start"]  # position of answer in paragraph, integer

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
                    query_tokens = nltk.word_tokenize(question)
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

                    question_id = qa_pair["id"]  # unique id for question

                    # Wrapping sequences with special symbols
                    # (see discussion at: https://github.com/tensorflow/nmt/issues/3)
                    # Query tokens will be used as input to the decoder in a seq2seq model and
                    # they have a begin symbol inserted at the start.
                    # Target query tokens are used to calculate losses based on the decoder output
                    # and are appended with the end symbol.
                    target_query_tokens = query_tokens.copy()
                    target_query_tokens.append(SYMB_END)
                    query_tokens.insert(0, SYMB_BEGIN)

                    # Construct lists of numeric token representations.
                    # Look up tokens in dictionary, if it's not there then use the value for
                    # @unk (unknown tokens)

                    document_words = [word_dictionary.get(token, word_dictionary[SYMB_UNK]) for
                                      token in doc_tokens]
                    query_words = [word_dictionary.get(token, word_dictionary[SYMB_UNK]) for
                                   token in query_tokens]
                    target_query_words = [word_dictionary.get(token, word_dictionary[SYMB_UNK]) for
                                          token in target_query_tokens]
                    if use_chars:
                        document_characters = [
                            [character_dictionary.get(c, character_dictionary[' '])
                             for c in list(w)[:MAX_WORD_LEN]] for w in doc_tokens]
                        query_characters = [
                            [character_dictionary.get(c, character_dictionary[' '])
                             for c in list(w)[:MAX_WORD_LEN]] for w in query_tokens]

                    else:
                        document_characters, query_characters = [], []

                    answer = [word_dictionary.get(token, word_dictionary[SYMB_UNK])
                              for token in answer_tokens]

                    # Append to predefined lists
                    question_list = [document_words, query_words, target_query_words, answer,
                                     document_characters, query_characters, answer_start_token,
                                     answer_end_token, question_id]
                    dataset_list.append(question_list)
            tr.update()
        tr.close()

        print("{:-^100}".format("DEBUG"))
        print("During parsing, {} documents were shortened to fit the 256 seq. length limit"
              ", and there were {} documents in total"
              .format(debug_slice_count, doc_count))

        # Create a binary pickle dump of the dataset tuple
        # Directory for the generated binary files (file_name index is to remove .json extension)
        binary_path = "/scratch/s161027/ga_reader_data/ssqa_processed/" + file_name + ".bin"
        outfile = open(binary_path, "wb")

        start_time = time.time()
        # pickle.dump(dataset_list, outfile)
        msgpack.pack(dataset_list, outfile, use_bin_type=True)
        outfile.close()
        unpack_time = time.time() - start_time

        print("Packing took {}".format(unpack_time))

        print("Dataset was parsed into {}".format(binary_path))

    def make_combined_dictionary(self, dataset_files, raw_file_dir, training_set,
                                 unlabeled_set, vocab_size, vocab_file):
        # This method creates a combined vocabulary between a labeled and an unlabeled dataset.
        # For example: training_set=0.1, unlabeled_set="small" will get the word frequency list
        # for both 10% of the training set and the small unlabeled dataset.

        # First we check for a saved vocabulary binary file, that contains a list of the unique
        # words in the dataset sorted from most frequent to last. If it exists, then load it.
        # If it doesn't, then generate a new one.

        if os.path.exists(vocab_file):
            print("Loading vocabularies from " + vocab_file + " ...")

            infile = open(vocab_file, 'rb')
            vocab_list = pickle.load(infile)
        else:
            print("No vocab file found on the following path:\n" + vocab_file)
            # New dictionary generation
            # Initialize list containing all tokens from all files
            text = []
            # Collect all tokens in json files into one list
            raw_file_paths = [os.path.join(raw_file_dir, dataset_files[unlabeled_set]) + ".json",
                              os.path.join(raw_file_dir, "train-p" + training_set) + ".json",
                              os.path.join(raw_file_dir, "dev-v1.1.json"),
                              os.path.join(raw_file_dir, "test-p0.1.json")]

            for dataset in raw_file_paths:
                data_file = open(dataset, encoding="utf8")
                json_file = json.loads(data_file.read())
                # List of all the paragraph/question/answer triplets in the data
                all_topics = json_file["data"]

                tr = trange(len(all_topics),
                            desc="Collecting tokens from {} for vocabulary"
                                 .format(dataset),
                            leave=True, ncols=100, ascii=True)

                for topic in all_topics:
                    topic = topic["paragraphs"]
                    for index, sample in enumerate(topic):
                        paragraph = sample["context"].replace("'", "")  # The paragraph text
                        # Replacing new lines with space to have paragraph as a one-line string
                        paragraph = paragraph.replace("\n", " ")
                        # Extend token list with tokens from the current paragraph
                        doc_tokens = nltk.word_tokenize(paragraph)

                        # Limiting doc_tokens to 512 max length to fit GPU memory with higher
                        # batch sizes. Comment this part out if memory is not an issue
                        if len(doc_tokens) > 512:
                            doc_tokens = doc_tokens[0:512]
                        text.extend(doc_tokens)
                        # The set of all (5) question/answer pairs relating to the paragraph
                        qa_set = sample["qas"]
                        for qa_pair in qa_set:
                            question = qa_pair["question"].replace("'", "")  # question text
                            answer = qa_pair["answers"][0]["text"].replace("'", "")  # answer text
                            # Extend token list with tokens from the current question and answer
                            text.extend(nltk.word_tokenize(question))
                            text.extend(nltk.word_tokenize(answer))

                    tr.update()
                tr.close()

            # Initialize word frequency dictionary as {"word": (frequency count)}
            word_frequency = {}

            tr = trange(len(text), desc="Constructing token frequency list",
                        leave=True, ncols=120, ascii=True)
            # Loop through list, store unique tokens and count their frequency
            for token in text:
                try:  # If the token is already in the dictionary, increment frequency
                    word_frequency[token] += 1
                except KeyError:  # Else, it's a new token, create key/value pair.
                    word_frequency[token] = 1
                tr.update()
            tr.close()

            # Sort dictionary by token frequency
            sorted_word_frequency_dict = sorted(word_frequency, key=word_frequency.get, reverse=True)
            # Get the N-5 most frequent tokens (N = vocab_size)
            # -5 is to leave space for special symbols such as @pad, @unk etc.,
            # while keeping the vocab size exactly as defined by the vocab_size argument.
            vocab_list = sorted_word_frequency_dict[0:vocab_size-5]

            special_vocab = [SYMB_PAD, SYMB_BEGIN, SYMB_END, SYMB_PLACEHOLDER, SYMB_UNK]

            # Adding special symbols together with the rest of the tokens
            vocab_list = special_vocab + vocab_list

            print("Dumping vocabulary to binary file: {}".format(vocab_file))
            outfile = open(vocab_file, "wb")
            pickle.dump(vocab_list, outfile)
            outfile.close()

        # Check vocab_size
        vocab_size_check = len(vocab_list)
        assert vocab_size_check == vocab_size, \
            "Mismatch between defined and actual vocabulary size.\
            \nDefined vocab size: {}\nActual:{}".format(vocab_size, vocab_size_check)

        word_dictionary = dict(zip(vocab_list, range(vocab_size)))
        char_set = set([c for w in vocab_list for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocab_list if v.startswith('@entity')])
        print("vocab_size = {}".format(vocab_size))
        print("num characters = {}".format(len(char_set)))
        print("{} anonymized entities".format(num_entities))
        print("{} other tokens (including @placeholder, {}, {}, {} and {})".format(
            vocab_size - num_entities, SYMB_BEGIN, SYMB_END, SYMB_PAD, SYMB_UNK))

        return word_dictionary, char_dictionary, num_entities
