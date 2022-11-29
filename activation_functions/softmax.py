from math import e
from math_utils.type import is_num


def normalize(a):
    if is_num(a):
        return 0
    elif is_num(a[0]):
        m = max(a)
        res = []
        for i in a:
            res.append(i-m)
        return res
    else:
        res = []
        for i in a:
            res.append(normalize(i))
        return res


def exponentiate(a):
    if is_num(a):
        return e**a
    elif is_num(a[0]):
        res = []
        for i in a:
            res.append(e**i)
        return res
    else:
        res = []
        for i in a:
            res.append(exponentiate(i))
        return res


def softmax_activation_function(x):
    if is_num(x):
        return 1/x
    elif is_num(x[0]):
        x = exponentiate(normalize(x))
        res = []
        s = sum(x)
        for i in x:
            res.append(i/s)
        return res
    else:
        res = []
        for i in x:
            res.append(softmax_activation_function(i))
        return res
