from denseLayer import DenseLayer

if __name__ == '__main__':
    layer1 = DenseLayer(4, 5)
    layer2 = DenseLayer(5, 2)

    X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

    l1 = layer1.execute(X)
    l2 = layer2.execute(l1)
    print(l2)
