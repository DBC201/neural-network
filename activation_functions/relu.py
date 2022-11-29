from math_utils.type import is_num


def relu_activation_function(x, threshold=0):
    if is_num(x):
        if x <= threshold:
            return 0
        else:
            return x
    elif is_num(x[0]):
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
            res.append(relu_activation_function(i, threshold))
        return res
