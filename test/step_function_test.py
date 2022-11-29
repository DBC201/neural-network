from activation_functions.step import step_activation_function

if __name__ == '__main__':
    assert step_activation_function(1) == 1
    assert step_activation_function(0) == 0
    assert step_activation_function(-1) == 0
    X = [1, -1, 2, 3]
    x = [1, 0, 1, 1]
    assert step_activation_function(X) == x
    X = [[1, 2, 3, -1], [1, 5, -1, -2]]
    x = [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert step_activation_function(X) == x
    X = [[[1, 2, 3, -1], [1, 5, -1, -2]], [[1, 2, 3, -1], [1, 5, -1, -2]]]
    x = [[[1, 1, 1, 0], [1, 1, 0, 0]], [[1, 1, 1, 0], [1, 1, 0, 0]]]
    assert step_activation_function(X) == x
