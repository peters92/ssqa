import nltk
from tqdm import tqdm, trange
import glob
import os
import random
from utils.Helpers import SYMB_PLACEHOLDER, \
                          SYMB_BEGIN,\
                          SYMB_END, \
                          SYMB_PAD
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

    def preprocess(self, question_dir, max_example=None,
                   use_chars=True, only_test_run=False):
        """
        preprocess all data into a standalone Data object.
        """
        vocab_f = os.path.join(question_dir, "vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(question_dir, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)

        print("preparing training data ...")
        training = self.parse_all_files(question_dir + "/train-p0.2.json",
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

    def make_dictionary(self, question_dir, vocab_file):
        if os.path.exists(vocab_file):
            print("Loading vocabularies from " + vocab_file + " ...")
            vocabularies = [lines.strip() for lines in open(vocab_file).readlines()]
        else:
            print("No vocab file found on the following path:\n" + vocab_file)

            fnames = []
            fnames += glob.glob(question_dir + "/test-p0.1.json/*.question")
            fnames += glob.glob(question_dir + "/dev-v1.1.json/*.question")
            fnames += glob.glob(question_dir + "/train-p0.9.json/*.question")
            vocab_set = set()

            # Progress bar
            tr = trange(len(fnames), desc="Constructing vocabulary",
                        leave=True, ncols=100, ascii=True)
            for fname in fnames:

                fp = open(fname)
                fp.readline()
                fp.readline()
                document = nltk.word_tokenize(fp.readline())
                fp.readline()
                query = nltk.word_tokenize(fp.readline())
                fp.readline()
                answer = nltk.word_tokenize(fp.readline())
                fp.close()

                vocab_set |= set(document) | set(query) | set(answer)

                tr.update()
            tr.close()
            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin, @end and @pad are included in the
            # vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_PLACEHOLDER)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)
            tokens.add(SYMB_PAD)

            vocabularies = list(entities)+list(tokens)
            print("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)

        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print("vocab_size = %d" % vocab_size)
        print("num characters = %d" % len(char_set))
        print("%d anonymoused entities" % num_entities)
        print("%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_BEGIN, SYMB_END))

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
        # we remove them from the dataset
        try:
            ans_start_char = raw[8].split()[0]
            ans_end_char = raw[8].split()[1]
            ans_start_token = raw[10].split()[0]
            ans_end_token = raw[10].split()[1]
        except IndexError:
            self.removed_questions.append(fname)
            self.num_removed_questions += 1
            return (0,)

        # wrap the query with special symbols
        query_raw.insert(0, SYMB_BEGIN)
        query_raw.append(SYMB_END)

        # tokens/entities --> indexes
        document_words = [w_dict[w] for w in document_raw]
        query_words = [w_dict[w] for w in query_raw]
        if use_chars:
            document_characters = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in document_raw]
            query_characters = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in query_raw]

            # TODO: commented out because it is not needed
            # answer_start = ans_start_char
            # answer_end = ans_end_char
        else:
            document_characters, query_characters = [], []
            # TODO: commented out because it is not needed, answer index is the same
            # answer_start = ans_start_token
            # answer_end = ans_end_token

        answer_start = ans_start_token
        answer_end = ans_end_token
        
        answer = [w_dict[w] for w in answer_raw]

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
