from utils.SquadAnalysis import Analyser
import numpy as np

analyser = Analyser()

np.set_printoptions(precision=3, suppress=True)

analyser.visualize_attention(num_examples=10, file_suffix="green")
