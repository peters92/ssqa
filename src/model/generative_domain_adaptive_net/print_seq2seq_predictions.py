from utils.Seq2SeqAnalysis import Analyser

analyser = Analyser()

model_name = "gen_test_lr1e-3_dropout0.25_new"
epoch_number = 6
text_outputs, numeric_output, batch_number = \
    analyser.get_predictions(max_example=100, model_name=model_name, epoch=epoch_number)

previous_doc = []
doc = text_outputs[0]
qry = text_outputs[1]
ans = text_outputs[2]
pred_qry = text_outputs[3]

qry_count = 1
for j in range(len(doc)):
    if doc[j] != previous_doc:
        print(100 * "-")
        print("\n\nParagraph {}: '{}'".format(j+1, doc[j]))
        qry_count = 1
    print("\nQuery {}: '{}'".format(qry_count, qry[j]))
    print("\nAnswer: '{}'".format(ans[j]))
    print("\nPredicted query: '{}'".format(pred_qry[j]))
    previous_doc = doc[j]
    qry_count += 1
