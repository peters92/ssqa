from utils.DataPreprocessor import DataPreprocessor

question_dir = "/scratch/s161027/ga_reader_data/ssqa_processed"

dp = DataPreprocessor()
data = dp.preprocess(question_dir)

print(type(data))