from utils.GAReaderAnalysis import Analyser

analyser = Analyser()

text_outputs, numeric_output, attentions_and_probs, batch_number = \
    analyser.get_predictions(max_example=10, epoch=3)

previous_doc = []
doc = text_outputs[0]
qry = text_outputs[1]
ans = text_outputs[2]
pred_ans = text_outputs[3]

qry_count = 1
for j in range(len(doc)):
    if doc[j] != previous_doc:
        print(100 * "-")
        print("\n\nParagraph {}: '{}'".format(j+1, doc[j]))
        qry_count = 1
    print("\nQuery {}: '{}'".format(qry_count, qry[j]))
    print("\nGround Truth: '{}'".format(ans[j]))
    print("\nPredicted answer: '{}'".format(pred_ans[j]))
    previous_doc = doc[j]
    qry_count += 1
