from math_utils.type import naive_is_num
import numpy


# relu stands for rectified linear unit
def naive_relu_activation_function(x, threshold=0):
    if naive_is_num(x):
        if x <= threshold:
            return 0
        else:
            return x
    elif naive_is_num(x[0]):
        res = []
        for i in x:
            if i <= threshold:
                res.append(0)
            else:
                res.append(i)
        return res
    else:
        res = []
        for i in x:
            res.append(naive_relu_activation_function(i, threshold))
        return res


def numpy_relu_activation_function(x):
    return numpy.maximum(0, x)
