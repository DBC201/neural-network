from math_utils.type import is_num


def dot_product(a, b):
    # print(a, b)
    if is_num(a) and is_num(b):
        return a * b
    elif is_num(a[0]) and is_num(b[0]):
        res = 0
        for i in range(len(a)):
            res += a[i] * b[i]
        return res
    elif is_num(a[0]) and is_num(b[0][0]):
        res = []
        for i in range(len(b[0])):
            rr = 0
            for j in range(len(a)):
                rr += a[j] * b[j][i]
            res.append(rr)
        return res
    elif is_num(a[0]):
        res = []
        for eb in b:
            res.append(dot_product(a, eb))
        return res
    else:
        res = []
        for ea in a:
            res.append(dot_product(ea, b))
        return res


def transpose(a):
    if is_num(a[0]):
        return
    else:
        res = [[e] for e in a[0]]
        for e in a[1:]:
            for i, eb in enumerate(e):
                res[i].append(eb)
        return res
