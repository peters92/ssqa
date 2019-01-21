import nltk
from tqdm import trange
import os
import json
import pickle
from utils.Helpers import SYMB_PLACEHOLDER, \
                          SYMB_BEGIN,\
                          SYMB_END, \
                          SYMB_PAD, \
                          SYMB_UNK
MAX_WORD_LEN = 10
MWETokenizer = nltk.tokenize.MWETokenizer


class Data:

    def __init__(self, dictionary, num_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v: k for k, v in dictionary[0].items()}


class DataPreprocessor:
    def __init__(self):
        self.removed_questions = []
        self.num_removed_questions = 0
        # Multi-word tokenizer for fixing placeholders in cloze-style data
        # This is to join "@" and "placeholder" as one token.
        self.tokenizer = MWETokenizer([('@', 'placeholder')], separator='')

    def preprocess(self, question_dir, training_set, vocab_size, max_example=100,
                   use_chars=True, only_test_run=False):
        """
        preprocess all data into a standalone Data object.
        """
        # Define files to be used
        dataset_files = {"training": "train-p" + training_set,
                         "validation": "dev-v1.1",
                         "test": "test-p0.1"}
        # Define vocabulary source file
        vocab_f = os.path.join(question_dir, "vocab_" + dataset_files["training"][-3:] + ".bin")

        # Generate or load dictionaries
        raw_file_dir = '/scratch/s161027/ga_reader_data/ssqa'
        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(dataset_files, raw_file_dir, vocab_size,
                                 vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)

        # Check for the existence of parsed data in binary files
        # Load them if they exist. Otherwise generate and save new ones.
        print("Preparing data...")

        loaded_dataset = []
        for data_role in dataset_files.keys():
            path = os.path.join(question_dir, dataset_files[data_role])+".bin"

            # Parse original json file and save results in binary, then load with pickle
            if not os.path.exists(path):
                print("Can't find binary data for {} set. Parsing raw data...".format(data_role))
                self.json_parser(raw_file_dir, dataset_files[data_role], dictionary, use_chars)

            print("Loading binary data for {} set".format(data_role))
            infile = open(path, "rb")
            loaded_data = pickle.load(infile)

            loaded_dataset.append(loaded_data)

        training, validation, test = loaded_dataset

        if only_test_run:
            training = training[0:max_example]
            validation = validation[0:max_example]
            test = test[0:max_example]

        data = Data(dictionary, num_entities, training, validation, test)

        # Message about bad samples being removed.
        print("{} questions were removed due to bad formatting."
              .format(self.num_removed_questions))

        return data

    def make_dictionary(self, dataset_files, raw_file_dir, vocab_size, vocab_file):

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
            # DEBUG NEW METHOD reading from json
            # Collect all tokens in json files into one list
            for data_role in dataset_files.keys():
                raw_file_path = os.path.join(raw_file_dir, dataset_files[data_role]) + ".json"
                data_file = open(raw_file_path, encoding="utf8")
                json_file = json.loads(data_file.read())
                # List of all the paragraph/question/answer triplets in the data
                all_topics = json_file["data"]

                tr = trange(len(all_topics),
                            desc="Collecting tokens from {} set for vocabulary".format(data_role),
                            leave=True, ncols=100, ascii=True)

                for topic in all_topics:
                    topic = topic["paragraphs"]
                    for index, sample in enumerate(topic):
                        paragraph = sample["context"].replace("'", "")  # The paragraph text
                        # Replacing new lines with space to have paragraph as a one-line string
                        paragraph = paragraph.replace("\n", " ")
                        # Extend token list with tokens from the current paragraph
                        text.extend(nltk.word_tokenize(paragraph))
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

            # Debug, dump complete word frequency list for visualizing.
            # outfile = open(vocab_file, "wb")
            # pickle.dump([vocab_list, sorted_word_frequency_dict, word_frequency], outfile)
            # outfile.close()

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
                    doc_tokens = nltk.word_tokenize(paragraph)
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

                    # Construct lists of numeric token representations.
                    # Look up tokens in dictionary, if it's not there then use the value for
                    # @unk (unknown tokens)

                    document_words = [word_dictionary.get(token, word_dictionary[SYMB_UNK]) for
                                      token in doc_tokens]
                    query_words = [word_dictionary.get(token, word_dictionary[SYMB_UNK]) for
                                   token in query_tokens]
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
                    question_tuple = (document_words, query_words, answer, document_characters,
                                      query_characters, answer_start_token, answer_end_token,
                                      question_id)
                    dataset_list.append(question_tuple)
            tr.update()
        tr.close()

        # Create a binary pickle dump of the dataset tuple
        # Directory for the generated binary files (file_name index is to remove .json extension)
        binary_path = "/scratch/s161027/ga_reader_data/ssqa_processed/" + file_name + ".bin"
        outfile = open(binary_path, "wb")
        pickle.dump(dataset_list, outfile)
        outfile.close()

        print("Dataset was parsed into {}".format(binary_path))
