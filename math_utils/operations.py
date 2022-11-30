from math_utils.type import naive_is_num


def naive_dot_product(a, b):
    # print(a, b)
    if naive_is_num(a) and naive_is_num(b):
        return a * b
    elif naive_is_num(a[0]) and naive_is_num(b[0]):
        res = 0
        for i in range(len(a)):
            res += a[i] * b[i]
        return res
    elif naive_is_num(a[0]) and naive_is_num(b[0][0]):
        res = []
        for i in range(len(b[0])):
            rr = 0
            for j in range(len(a)):
                rr += a[j] * b[j][i]
            res.append(rr)
        return res
    elif naive_is_num(a[0]):
        res = []
        for eb in b:
            res.append(naive_dot_product(a, eb))
        return res
    else:
        res = []
        for ea in a:
            res.append(naive_dot_product(ea, b))
        return res


def naive_transpose(a):
    if naive_is_num(a[0]):
        return
    else:
        res = [[e] for e in a[0]]
        for e in a[1:]:
            for i, eb in enumerate(e):
                res[i].append(eb)
        return res
