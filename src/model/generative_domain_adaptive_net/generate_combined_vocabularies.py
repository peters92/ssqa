from utils.UnlabeledDataPreprocessor import UnlabeledDataPreprocessor

dp = UnlabeledDataPreprocessor()

vocab_size = 10000
raw_file_dir = "/scratch/s161027/ga_reader_data/ssqa"
data_dir = "/scratch/s161027/ga_reader_data/ssqa_processed"
unlabeled_dataset = "small"
dataset_files = {"small": "unlabeled-data.small",
                 "large": "unlabeled-data.large"}
training_sets = ["0.1", "0.2", "0.5", "0.9"]

for training_set in training_sets:
    vocab_filename = "vocab_training"+training_set+"+"+"unlabeled_"+unlabeled_dataset+".bin"
    vocab_filepath = data_dir+"/"+vocab_filename
    dp.make_combined_dictionary(dataset_files, raw_file_dir, training_set,
                                unlabeled_dataset, vocab_size, vocab_filepath)

