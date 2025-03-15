import numpy as np

''' code is implemented form "Amplifying-membership-exposure-via-data-poisoning"
    github:https://github.com/yfchen1994/poisoning_membership/blob/main/attack/attack_utils.py
'''

def _Mentr(preds, y):

    fy = np.sum(preds * y, axis=1)
    fi = preds * (1 - y)
    score = -(1 - fy) * np.log(fy + 1e-30) - np.sum(fi * np.log(1 - fi + 1e-30), axis=1)
    return score
