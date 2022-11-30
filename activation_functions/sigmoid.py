from math import e
from math_utils.type import naive_is_num


def naive_sigmoid(a):
    return 1/(1+(e**(a*-1)))


def naive_sigmoid_activation_function(x):
    if naive_is_num(x):
        return naive_sigmoid(x)
    elif naive_is_num(x[0]):
        res = []
        for i in x:
            res.append(naive_sigmoid(i))
        return res
    else:
        res = []
        for i in x:
            res.append(naive_sigmoid_activation_function(i))
        return res
