from math import log
from math_utils.type import naive_is_num

import numpy as np


minimum = 10**-7
maximum = 1 - 10**-7
cross_entropy_minimum = -log(minimum)
cross_entropy_maximum = -log(maximum)


def naive_categorical_cross_entropy(actual, expected):
    """
    -Σy(i,j)*ln(_y(i,j))

    y is expected and _y is actual
    :param expected: One hot encoding or index list
    :param actual:
    :return: mean loss
    """
    if naive_is_num(expected[0]):
        if naive_is_num(actual[0]):
            return -1 * log(actual[expected[0]])
        else:
            r = 0
            for i, l in enumerate(actual):
                a = l[expected[i]]
                if a < minimum:
                    r += cross_entropy_minimum
                elif a > maximum:
                    r += cross_entropy_maximum
                else:
                    r += -log(a)
            return r/len(actual)
    else:
        if naive_is_num(actual[0]):
            r = 0
            for i, e in enumerate(expected[0]):
                a = actual[i]
                if a < minimum:
                    r += cross_entropy_minimum * e
                elif a > maximum:
                    r += cross_entropy_maximum * e
                else:
                    r += -log(a) * e
            return r
        else:
            v = []
            for i, l in enumerate(expected):
                r = 0
                for j, e in enumerate(expected[0]):
                    a = actual[i]
                    if a < minimum:
                        r += cross_entropy_minimum * e
                    elif a > maximum:
                        r += cross_entropy_maximum * e
                    else:
                        r += -log(a) * e
                v.append(r)
            return sum(v)/len(v)


def numpy_categorical_cross_entropy(actual, expected):
    # Number of samples in a batch
    samples = len(actual)
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    actual_clipped = np.clip(actual, 1e-7, 1 - 1e-7)

    correct_confidences = None

    if len(expected.shape) == 1:
        correct_confidences = actual_clipped[
            range(samples),
            expected
        ]
    # Mask values - only for one-hot encoded labels
    elif len(expected.shape) == 2:
        correct_confidences = np.sum(
            actual_clipped * expected,
            axis=1
        )
    # Losses
    negative_log_likelihoods = -np.log(correct_confidences)
    return np.mean(negative_log_likelihoods)
