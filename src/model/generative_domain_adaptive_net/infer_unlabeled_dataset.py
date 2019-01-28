import os
import msgpack
from tqdm import trange
from utils.Seq2SeqAnalysis import Analyser

"""
This script generates predictions (questions) based on a trained model and an unlabeled dataset that consists of
tuples such as (document, answer).
After generating predictions, the script uses the ID of the prediction to find the original document
and answer from the dataset and inserts the generated question next to them.
Finally, the modified unlabeled data (no longer unlabeled) is packed in binary for later use with a
supervised model.
"""

analyser = Analyser()

training_set = "0.9"
model_name = "combined_train0.9_dropout0.1_nlayer_1"
epoch_number = 6
numeric_output, original_data = \
    analyser.get_predictions(max_example=100, model_name=model_name, epoch=epoch_number,
                             training_set=training_set, test_run=False, unlabeled=True,
                             save_dir=None)

# For each prediction in the batches, find the ID in the original dataset
# and create tuples of (prediction, ID).
predictions_list = []

print("Building list of (predictions, question_ids)")
for (predictions, filenames) in numeric_output:
    for index, value in enumerate(predictions):
        # Get the ID of the prediction
        current_id = filenames[index]
        current_prediction = value
        predictions_list.append((current_prediction, current_id))

# For each prediction and ID, find the respective sample in the original data
# and insert the prediction as the question.
print("Inserting generated questions into unlabeled dataset...")
tr = trange(len(predictions_list), desc="", leave=True, ncols=100, ascii=True)
for prediction, question_index in predictions_list:
    for index, sample in enumerate(original_data.training):
        if question_index == sample[-1]:
            # Insert current prediction into query list of original data
            # Get rid of padding (index=0)
            current_prediction = prediction[prediction != 0]
            # Get rid of end of seq. symbol (index=2)
            current_prediction = current_prediction[current_prediction != 2]
            # Convert to list from numpy ndarray
            current_prediction = list(current_prediction)
            # Convert elements to python int, so that it can be serialized later
            current_prediction = [int(num) for num in current_prediction]
            # Create copies of list for query words and target query words separately
            query_words = current_prediction.copy()
            target_query_words = current_prediction.copy()
            # Pad with @begin and @end symbols
            query_words.insert(0, 1)
            target_query_words.append(2)
            # Insert into original data at the right index
            original_data.training[index][1] = query_words
            original_data.training[index][2] = target_query_words
    tr.update()
tr.close()

# Save the modified dataset with a specific name
output_dir = "/scratch/s161027/ga_reader_data/ssqa_processed/inferred_data/"
filename = "inferred_unlabeled_small_train" + training_set + ".bin"
output_path = os.path.join(output_dir, filename)
outfile = open(output_path, "wb")
print("Packing modified dataset to path: {}".format(output_path))
msgpack.pack(original_data.training, outfile, use_bin_type=True)

# DEBUG
# text_outputs, numeric_output, original_data, original_dict = \
#     analyser.get_predictions(max_example=500, model_name=model_name, epoch=epoch_number,
#                              training_set=training_set, test_run=True, unlabeled=True,
#                              save_dir=None)
#
# word_dict = original_dict[0]
# inv_dict = dict(zip(word_dict.values(), word_dict.keys()))
#
# # Check that generated prediction fits the original document/answer
# print("{:-^100}".format("Output from prediction"))
# pred_qry = text_outputs[0][3][0]
# pred_id = numeric_output[0][-1][0]
# pred_doc = text_outputs[0][0][0]
# print("Prediction: {}\nID: {}\nDocument: {}".format(pred_qry, pred_id, pred_doc))
#
# print("{:-^100}".format("Original data"))
# for index, sample in enumerate(original_data):
#     if sample[-1] == pred_id:
#         print("Found the question id!")
#         doc = [inv_dict[word] for word in sample[0]]
#         qry = [inv_dict[word] for word in sample[1]]
#         ans = [inv_dict[word] for word in sample[3]]
#         print("Doc: {}\nQry: {}\nAns: {}".format(doc, qry, ans))
