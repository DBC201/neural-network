import random


class DenseLayer:
    def __init__(self, input_size, neurons):
        self.input_size = input_size
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(neurons)] for _ in range(input_size)]
        self.bias = [0 for _ in range(neurons)]

    @staticmethod
    def is_num(n):
        return isinstance(n, float) or isinstance(n, int)

    @staticmethod
    def dot_product(a, b):

        #print(a, b)

        if DenseLayer.is_num(a) and DenseLayer.is_num(b):
            return a*b
        elif DenseLayer.is_num(a[0]) and DenseLayer.is_num(b[0]):
            res = 0
            for i in range(len(a)):
                res += a[i] * b[i]
            return res
        elif DenseLayer.is_num(a[0]) and DenseLayer.is_num(b[0][0]):
            res = []
            for i in range(len(b[0])):
                rr = 0
                for j in range(len(a)):
                    rr += a[j] * b[j][i]
                res.append(rr)
            return res
        elif DenseLayer.is_num(a[0]):
            res = []
            for eb in b:
                res.append(DenseLayer.dot_product(a, eb))
            return res
        else:
            res = []
            for ea in a:
                res.append(DenseLayer.dot_product(ea, b))
            return res

    @staticmethod
    def transpose(a):
        if DenseLayer.is_num(a[0]):
            return
        else:
            res = [[e] for e in a[0]]
            for e in a[1:]:
                for i, eb in enumerate(e):
                    res[i].append(eb)
            return res

    def execute(self, x):
        products = DenseLayer.dot_product(x, self.weights)
        if DenseLayer.is_num(products):
            return products + self.bias[0]
        else:
            for e in products:
                for i, _ in enumerate(e):
                    e[i] += self.bias[i]
            return products
