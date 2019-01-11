from utils.SquadAnalysis import Analyser

analyser = Analyser()

# This will create num_examples amount of .svg files with paragraph
# colored according to attention. The files will also contain the query,
# ground truth, predicted answer and also gradient color scale for reference.
analyser.visualize_attention(num_examples=20, file_suffix="TEMP")
