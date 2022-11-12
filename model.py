from layer import Layer

if __name__ == '__main__':
    inp = [1, 1, 1]
    l1 = Layer([[1, 2, 3], [4, 5, 6]], 31)
    l2 = Layer([3, 4], 69)
    print(l2.execute(l1.execute(inp)))
