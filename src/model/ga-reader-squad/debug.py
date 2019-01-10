from data.ssqa_processing import ssqa_parser

data_path = "/scratch/s161027/ga_reader_data/ssqa"
filenames = ["dev-v1.1.json", "test-p0.1.json",
             "train-p0.1.json", "train-p0.2.json",
             "train-p0.5.json", "train-p0.9.json",
             "unlabeled-data.small.json", "unlabeled-data.large.json"]

for filename in filenames:
    print("Parsing {}...".format(filename))
    ssqa_parser(data_path, filename)
