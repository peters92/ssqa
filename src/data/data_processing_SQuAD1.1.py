'''
This script reads the SQuAD 1.1 dataset in JSON then outputs each sample within as a separate .question file.
These files will each contain a paragraph, question and an answer, which can then be used by the Gated Attention Reader model.
'''
import json
import os
import sys

data_path = "../../data/SQuAD-v1.1.json"
data_file = open(data_path, encoding="utf8")
json_file = json.loads(data_file.read())
data = json_file[1][1:] # The list containing all the sample paragraphs and question/answer pairs

# Directory for the separated question files
data_dir = "../../data/SQuADv1.1"
if not os.path.exists(data_dir): os.mkdir(data_dir)

n = 0
N = len(data)

for sample in data:
    paragraph = sample[1][2].replace("'","") # The paragraph text
    qa_set = sample[2][2][1:]                # The set of all (5) question/answer pairs relating to the paragraph
    validation_role = sample[-1][2].replace("'","")   # text marking if the sample is training or validation data

    # Creating/Using the folder for training/validation samples
    file_dir = os.path.join(data_dir,validation_role)
    if not os.path.exists(file_dir): os.mkdir(file_dir)

    for qa_pair in qa_set:                          # for each question/answer pair we create a separate ".question" file
        question = qa_pair[1][2].replace("'","")    # question text
        answer = qa_pair[2][2][1].replace("'","")   # answer text
        answer_pos = qa_pair[3][2][1]               # position of answer in paragraph, integer
        question_id = qa_pair[4][2].replace("'","") # unique id for question

        file_name = file_dir + "\\" + question_id + ".question" # create .question file with the given question id as name
        f = open(file_name, "w+", encoding="utf-8")
        f.write(paragraph + "\n")
        f.write(question + "\n")
        f.write(answer + "\n")
        f.write(str(answer_pos) + "\n")
        f.close()

    if n % 10 == 0: # print progress
        sys.stdout.write("\rParsing dataset into separate files...{} done out of {}".format(n,N))
    n+=1
