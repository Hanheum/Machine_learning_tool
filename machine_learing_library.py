import random

def matmul(A, B, activation):
    A_Column = len(A[0])
    B_Column = len(B[0])
    new_matrix = []
    for aRow in A:
        new_matrix_row = []
        for bCol in range(B_Column):
            new_matrix_element = 0
            for Element_count in range(A_Column):
                one_part = aRow[Element_count]*B[Element_count][bCol]
                new_matrix_element += one_part
            if activation == None:
                new_matrix_row.append(new_matrix_element)
            else:
                new_matrix_row.append(activation(new_matrix_element))
        new_matrix.append(new_matrix_row)

    return new_matrix

def add(A, B):
    A_Column = len(A[0])
    A_Row = len(A)
    new_matrix = []
    for aRow in range(A_Row):
        new_matrix_row = []
        for aColumn in range(A_Column):
            new_matrix_element = A[aRow][aColumn]+B[aRow][aColumn]
            new_matrix_row.append(new_matrix_element)
        new_matrix.append(new_matrix_row)

    return new_matrix

def ReLU(x):
    if x<0:
        return 0
    elif x>=0:
        return x

def random_matrix(row_size, column_size, random_range):
    new_matrix = []
    for a in range(row_size):
        new_matrix_row = []
        for b in range(column_size):
            new_matrix_row.append(random.randint(random_range[0], random_range[1])/100)
        new_matrix.append(new_matrix_row)

    return new_matrix

def cost(prediction, answer):
    how_many = len(prediction)
    Cost = 0
    for i in range(how_many):
        target_prediction = prediction[i]
        target_answer = answer[i]
        for a in range(len(target_answer)):
            loss = (target_answer[a]-target_prediction[a])**2
            Cost += loss

    return Cost

def differential(function, x):
    a_very_small_unit = 0.000001
    first_y = function(x)
    second_y = function(x+a_very_small_unit)
    steep = (second_y-first_y)/a_very_small_unit

    return steep

def gradient_descent(model, target_variable, target_index, loss, x, y, learning_rate, max_iter):
    a_very_small_unit = 0.000001

    predictions_first = []
    for i in x:
        prediction = model(i)[0]
        predictions_first.append(prediction)
    first_loss = loss(predictions_first, y)

    target_variable[target_index[0]][target_index[1]] = target_variable[target_index[0]][target_index[1]] + a_very_small_unit
    predictions_second = []
    for i in x:
        prediction = model(i)[0]
        predictions_second.append(prediction)
    second_loss = loss(predictions_second, y)

    gradient = (second_loss-first_loss)/a_very_small_unit
    iter = 0

    while gradient > 0.01 and iter<max_iter:
        target_variable[target_index[0]][target_index[1]] = target_variable[target_index[0]][target_index[1]] - learning_rate*gradient
        predictions_first = []
        for i in x:
            prediction = model(i)[0]
            predictions_first.append(prediction)
        first_loss = loss(predictions_first, y)

        target_variable[target_index[0]][target_index[1]] = target_variable[target_index[0]][
                                                                target_index[1]] + a_very_small_unit
        predictions_second = []
        for i in x:
            prediction = model(i)[0]
            predictions_second.append(prediction)
        second_loss = loss(predictions_second, y)

        gradient = (second_loss - first_loss) / a_very_small_unit
        iter += 1

def optimize(model, layer_variables, loss, x, y, learning_rate, epochs, max_iter):
    for i in range(epochs):
        print(i)
        for variable in layer_variables:
            for a, row in enumerate(variable):
                for b, element in enumerate(row):
                    gradient_descent(model, variable, (a, b), loss, x, y, learning_rate, max_iter)

def flatten(array):
    flatten_array = []
    for i in array:
        flatten_array.append(i[0])
    return [flatten_array]