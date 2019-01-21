import nltk
from tqdm import tqdm, trange
import glob
import os
import random
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

    def preprocess(self, question_dir, vocab_size, max_example=None,
                   use_chars=True, only_test_run=False):
        """
        preprocess all data into a standalone Data object.
        """
        vocab_f = os.path.join(question_dir, "vocab.bin")

        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(question_dir, vocab_size, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)

        print("preparing training data ...")
        training = self.parse_all_files(question_dir + "/train-p0.1.json",
                                        dictionary, max_example,
                                        use_chars, only_test_run)

        print("preparing validation data ...")
        validation = self.parse_all_files(question_dir + "/dev-v1.1.json",
                                          dictionary, max_example,
                                          use_chars, only_test_run)

        print("preparing test data ...")
        test = self.parse_all_files(question_dir + "/test-p0.1.json",
                                    dictionary, max_example,
                                    use_chars, only_test_run)

        data = Data(dictionary, num_entities, training, validation, test)

        # Message about bad samples being removed.
        print("{} questions were removed due to bad formatting."
              .format(self.num_removed_questions))

        return data

    def make_dictionary(self, question_dir, vocab_size, vocab_file):

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
            # Gather all files containing text
            fnames = []
            fnames += glob.glob(question_dir + "/test-p0.1.json/*.question")
            fnames += glob.glob(question_dir + "/dev-v1.1.json/*.question")
            fnames += glob.glob(question_dir + "/train-p0.1.json/*.question")

            # Progress bar
            tr = trange(len(fnames), desc="Collecting tokens for vocabulary",
                        leave=True, ncols=120, ascii=True)
            # Initialize list containing all tokens from all files
            text = []
            # Collect all tokens in data into one list
            for fname in fnames:
                fp = open(fname)
                fp.readline()
                fp.readline()
                # Document
                text.extend(nltk.word_tokenize(fp.readline()))
                fp.readline()
                # Query
                text.extend(nltk.word_tokenize(fp.readline()))
                fp.readline()
                # Answer
                text.extend(nltk.word_tokenize(fp.readline()))
                fp.close()
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
            # -5 is to leave space for special symbols such as @pad, @unk etc., while keeping the vocab
            # size exactly as defined by the vocab_size argument.
            vocab_list = sorted_word_frequency_dict[0:vocab_size-5]

            special_vocab = [SYMB_PAD, SYMB_BEGIN, SYMB_END, SYMB_PLACEHOLDER, SYMB_UNK]

            # Adding special symbols together with the rest of the tokens
            vocab_list = special_vocab + vocab_list

            # TODO: replace these parts with pickle dump
            print("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocab_list))
            vocab_fp.close()

            # REPLACEMENT
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

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        ###############################################################################################################
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()

        # Using nltk to tokenize input
        document_raw = nltk.word_tokenize(raw[2])  # document
        query_raw = nltk.word_tokenize(raw[4])  # query
        answer_raw = nltk.word_tokenize(raw[6])  # answer

        # In case a paragraph has extra new lines ("\n") breaking the indexing,
        # we remove them from the dataset.
        # But this shouldn't happen anymore, since NLTK tokenizer takes care of those.
        try:
            ans_start_char = raw[8].split()[0]
            ans_end_char = raw[8].split()[1]
            ans_start_token = raw[10].split()[0]
            ans_end_token = raw[10].split()[1]
        except IndexError:
            self.removed_questions.append(fname)
            self.num_removed_questions += 1
            return (0,)

        # Wrap the query with special symbols
        query_raw.insert(0, SYMB_BEGIN)
        query_raw.append(SYMB_END)

        # Construct lists of numeric token representations.
        # Look up tokens in dictionary, if it's not there, use the value for @unk (unknown tokens)

        document_words = [w_dict.get(token, w_dict[SYMB_UNK]) for token in document_raw]
        query_words = [w_dict.get(token, w_dict[SYMB_UNK]) for token in query_raw]
        if use_chars:
            document_characters = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in document_raw]
            query_characters = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in query_raw]

        else:
            document_characters, query_characters = [], []

        answer_start = ans_start_token
        answer_end = ans_end_token
        
        answer = [w_dict.get(token, w_dict[SYMB_UNK]) for token in answer_raw]

        return document_words, query_words, answer, document_characters, query_characters, \
            answer_start, answer_end

    def parse_all_files(self, directory, dictionary,
                        max_example, use_chars, test_run=False):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """

        all_files = glob.glob(directory + '/*.question')
        if max_example is not None:
            all_files = random.sample(all_files, max_example)
        # If it's a test run we limit the amount of samples in the batch
        if test_run:
            all_files = random.sample(all_files, 100)

        # Wrap iterable for progress bar
        all_files = tqdm(all_files, leave=True, ascii=True, ncols=100)

        questions = [self.parse_one_file(f, dictionary, use_chars) +
                     (f,) for f in all_files]

        # In case of broken paragraphs, we remove those samples from the data
        if self.num_removed_questions != 0:
            for index, value in enumerate(questions):
                if len(value) == 2:
                    del questions[index]

        return questions
