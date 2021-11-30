from model import Model

import numpy as np


trues = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1])
preds = np.array([0, 0, 0, 0.9, 0.1, 0.6, 1, 1, 0, 0, 0.4, 1, 0, 1, 0, 1])

"""
Matrix should be 
[
4, 2
5, 5
]
"""

accuracy = Model.confusion_matrix(trues, preds)
print(accuracy)
