class Layer:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @staticmethod
    def is_num(n):
        return isinstance(n, float) or isinstance(n, int)

    @staticmethod
    def dot_product(a, b):

        #print(a, b)

        if Layer.is_num(a) and Layer.is_num(b):
            return a*b
        elif Layer.is_num(a[0]) and Layer.is_num(b[0]):
            res = 0
            for i in range(len(a)):
                res += a[i] * b[i]
            return res
        elif Layer.is_num(a[0]) and Layer.is_num(b[0][0]):
            res = []
            for i in range(len(b[0])):
                rr = 0
                for j in range(len(a)):
                    rr += a[j] * b[j][i]
                res.append(rr)
            return res
        elif Layer.is_num(a[0]):
            res = []
            for eb in b:
                res.append(Layer.dot_product(a, eb))
            return res
        else:
            res = []
            for ea in a:
                res.append(Layer.dot_product(ea, b))
            return res

    def execute(self, values):
        products = Layer.dot_product(self.weights, values)
        if Layer.is_num(products):
            return products
        else:
            return [x+self.bias for x in Layer.dot_product(self.weights, values)]
