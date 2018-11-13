from utils.SquadAnalysis import Analyser

analyser = Analyser()

text_outputs, batch_number = analyser.print_predictions(max_example=10, epoch=1)

for j in range(batch_number):
        doc = text_outputs[j][0][0]
        qry = text_outputs[j][1]
        ans = text_outputs[j][2]
        pred_ans = text_outputs[j][3]

        print(100*"-")
        print("\n\nParagraph {}: '{}'".format(j, doc))
        for i in range(len(qry)):
            print("\nQuery {}: '{}'".format(i, qry[i]))
            print("\nGround Truth {}: '{}'".format(i, ans[i]))
            print("\nPredicted answer {}: '{}'".format(i, pred_ans[i]))
