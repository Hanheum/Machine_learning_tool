from machine_learing_library import *

x = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
]

y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
]

w1 = random_matrix(2, 3, (1, 100))
b1 = random_matrix(1, 3, (0, 0))

w2 = random_matrix(3, 3, (1, 100))
b2 = random_matrix(1, 3, (0, 0))

def model(X):
    hidden1 = add(matmul(X, w1, activation=ReLU), b1)
    hidden2 = add(matmul(hidden1, w2, activation=ReLU), b2)
    return hidden2

optimize(model, [w1, b1, w2, b2], cost, x, y, 0.001, 5000)

for a in range(len(x)):
    prediction = model([x[a]])
    new_prediction = []
    for i in prediction[0]:
        element = round(i, 3)
        new_prediction.append(element)
    print(new_prediction)