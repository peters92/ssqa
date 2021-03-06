import os
# import logging
import numpy as np

EMBED_DIM = 128


def show_predicted_vs_ground_truth(probs, a, inv_dict):
    predicted_ans = map(lambda i:inv_dict[i], list(np.argmax(probs, axis=1)))
    true_ans = map(lambda i:inv_dict[i], list(a))
    print(zip(predicted_ans, true_ans))


def count_candidates(probs, c, m_c):
    hits = 0
    predicted_ans = list(np.argmax(probs, axis=1))
    for i, x in enumerate(predicted_ans):
        for j, y in enumerate(c[i,:]):
            if x == y and m_c[i,j] > 0:
                hits += 1
                break
    return hits


def show_question(d, q, a, m_d, m_q, c, m_c, inv_dict):
    i = 0

    def inv_vocab(x):
        return inv_dict[x]
    print(list(map(inv_vocab, list(d[i, m_d[i] > 0, 0]))))
    print(list(map(inv_vocab, list(q[i, m_q[i] > 0, 0]))))
    print(list(map(inv_vocab, list(c[i, m_c[i] > 0]))))
    print(inv_vocab(a[i]))


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