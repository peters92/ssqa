import os
import numpy as np

EMBED_DIM = 128


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
    for w, i in dictionary.items():
        if w in vocab_embed:
            # W row index is 'i-1', because the dictionary values
            # start at one. TODO: re-enable
            W[i, :] = vocab_embed[w]
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


def prepare_input(d, q):
    f = np.zeros(d.shape[:2]).astype('int32')
    for i in range(d.shape[0]):
        f[i, :] = np.in1d(d[i, :, 0], q[i, :, 0])
    return f
