from utils.SquadAnalysis import Analyser

analyser = Analyser()

text_outputs, numeric_output, attentions_and_probs, batch_number = \
    analyser.get_predictions(max_example=1, epoch=14)

for i in range(batch_number):
    previous_doc = []
    doc = text_outputs[i][0]
    qry = text_outputs[i][1]
    ans = text_outputs[i][2]
    pred_ans = text_outputs[i][3]

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
