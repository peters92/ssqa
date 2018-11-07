from utils.DataPreprocessor import DataPreprocessor
from utils.MiniBatchLoader import MiniBatchLoader

dp = DataPreprocessor()

data = dp.preprocess(
    question_dir="/scratch/s161027/ga_reader_data/squad", no_training_set=True,
        use_cloze_style=False, only_test_run=True)

test_batch_loader = MiniBatchLoader(
        data.test, 32, shuffle=False, use_cloze_style=False)
index = 0
for samples in test_batch_loader:
    dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, fnames = samples

    doc_len = len(dw)
    a_len = len(a)
    if not doc_len == a_len:
        print("something's wrong at index: {}".format(index))
    # if index == 0:
    #     print("Doc words: {}".format(dw))
    #     print("Answer: {}\nAnswer shape: {}".format(a, a.shape))
    index += 1
