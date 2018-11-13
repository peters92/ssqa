import nltk
from tqdm import tqdm, trange
import glob
import os

MAX_WORD_LEN = 10
MWETokenizer = nltk.tokenize.MWETokenizer

SYMB_PLACEHOLDER = "@placeholder"
SYMB_BEGIN = "@begin"
SYMB_END = "@end"

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

    def preprocess(self, question_dir, max_example=None, use_chars=True,
                   use_cloze_style=False, only_test_run=False):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir, "vocab.txt")
        word_dictionary, char_dictionary, num_entities = \
            self.make_dictionary(question_dir, vocab_file=vocab_f)
        dictionary = (word_dictionary, char_dictionary)

        print("preparing training data ...")
        training = self.parse_all_files(question_dir + "/training",
                                        dictionary, max_example,
                                        use_chars, use_cloze_style,
                                        only_test_run)

        print("preparing validation data ...")
        validation = self.parse_all_files(question_dir + "/validation",
                                          dictionary, max_example,
                                          use_chars, use_cloze_style,
                                          only_test_run)

        print("preparing test data ...")
        test = self.parse_all_files(question_dir + "/test",
                                    dictionary, max_example,
                                    use_chars, use_cloze_style,
                                    only_test_run)

        data = Data(dictionary, num_entities, training, validation, test)

        # Message about bad samples being removed. (SQuAD only)
        if not use_cloze_style:
            print("{} questions were removed due to bad formatting.".format(self.num_removed_questions))

        return data

    def make_dictionary(self, question_dir, vocab_file):
        if os.path.exists(vocab_file):
            print("Loading vocabularies from " + vocab_file + " ...")
            vocabularies = [lines.strip() for lines in open(vocab_file).readlines()]
        else:
            print("No vocab file found on the following path:\n" + vocab_file)

            fnames = []
            fnames += glob.glob(question_dir + "/test/*.question")
            fnames += glob.glob(question_dir + "/validation/*.question")
            fnames += glob.glob(question_dir + "/training/*.question")
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

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_PLACEHOLDER)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities)+list(tokens)
            print("writing vocabularies to " + vocab_file + " ...")
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

        vocab_size = len(vocabularies)
        # TODO: Shift dictionary values by 1, to not have 0 represent any word
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

    def parse_one_file_span(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        ###############################################################################################################
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        # SQUAD_MOD
        # Use nltk to tokenize input
        doc_raw = nltk.word_tokenize(raw[2])  # document
        qry_raw = nltk.word_tokenize(raw[4])  # query
        # SQUAD_MOD
        ans_raw = nltk.word_tokenize(raw[6])  # answer

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
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)

        # tokens/entities --> indexes
        doc_words = [w_dict[w] for w in doc_raw]
        qry_words = [w_dict[w] for w in qry_raw]
        if use_chars:
            doc_chars = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in doc_raw]
            qry_chars = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in qry_raw]
        else:
            doc_chars, qry_chars = [], []

        ans = [w_dict[w] for w in ans_raw]
        return doc_words, qry_words, ans, doc_chars, qry_chars, \
            ans_start_char, ans_end_char, ans_start_token, ans_end_token

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        ###############################################################################################################
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        # SQUAD_MOD
        # Use nltk to tokenize input
        doc_raw = nltk.word_tokenize(raw[2])  # document
        qry_raw = nltk.word_tokenize(raw[4])  # query
        qry_raw = self.tokenizer.tokenize(qry_raw)
        # SQUAD_MOD

        ans_raw = raw[6].strip()  # answer
        cand_raw = [cand.strip().split(":")[0].split() for cand in raw[8:]]
        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)

        cloze = qry_raw.index('@placeholder')

        # tokens/entities --> indexes
        doc_words = [w_dict[w] for w in doc_raw]
        qry_words = [w_dict[w] for w in qry_raw]
        if use_chars:
            doc_chars = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in doc_raw]
            qry_chars = [[c_dict.get(c, c_dict[' ']) for c in list(w)[:MAX_WORD_LEN]] for w in qry_raw]
        else:
            doc_chars, qry_chars = [], []

        ans = w_dict.get(ans_raw, 0)
        cand = [w_dict.get(w[0], 0) for w in cand_raw]
        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze

    def parse_all_files(self, directory, dictionary, max_example, use_chars, use_cloze_style, test_run=False):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        if use_cloze_style:
            parsing_function = self.parse_one_file
        else:
            parsing_function = self.parse_one_file_span

        all_files = glob.glob(directory + '/*.question')[:max_example]
        # Wrap iterable for progress bar
        if test_run:
            all_files = all_files[:100]
        all_files = tqdm(all_files, leave=True, ascii=True, ncols=100)

        questions = [parsing_function(f, dictionary, use_chars) + (f,) for f in all_files]

        # In case of broken paragraphs, we remove those samples from the data
        if self.num_removed_questions != 0:
            for index, value in questions:
                if len(value) == 2:
                    del questions[index]

        return questions

    def gen_text_for_word2vec(self, question_dir, text_file):

            fnames = []
            fnames += glob.glob(question_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:

                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()

                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()

if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.gen_text_for_word2vec("cnn/questions", "/tmp/cnn_questions.txt")
