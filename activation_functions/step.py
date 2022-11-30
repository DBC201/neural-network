from math_utils.type import naive_is_num


def naive_step_activation_function(x, threshold=0):
    if naive_is_num(x):
        if x <= threshold:
            return 0
        else:
            return 1
    elif naive_is_num(x[0]):
        res = []
        for i in x:
            if i <= threshold:
                res.append(0)
            else:
                res.append(1)
        return res
    else:
        res = []
        for i in x:
            res.append(naive_step_activation_function(i, threshold))
        return res
