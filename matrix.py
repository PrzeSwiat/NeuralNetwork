import random

import numpy as np


def fromArray(array):
    # to Matrix (obj) converter
    m = Matrix(len(array), 1)
    iterator = 0
    for i in array:
        m.matrix[iterator][0] = i
        iterator += 1
    return m


def mapping(matrix, func):
    result = Matrix(matrix.rows, matrix.columns)
    for i in range(matrix.rows):
        for j in range(matrix.columns):
            val = matrix.matrix[i][j]
            result.matrix[i][j] = func(val)

    return result


def multiplication(a, b):
    if a.columns != b.rows:
        print("Error! columns of A must match rows in B")
        return 0
    result = Matrix(a.rows, b.columns)
    for i in range(result.rows):
        for j in range(result.columns):
            sum = 0
            for k in range(a.columns):
                sum += a.matrix[i][k] * b.matrix[k][j]
            result.matrix[i][j] = sum
    return result


class Matrix:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        s = (rows, columns)
        self.matrix = np.zeros(s, dtype=float)

    def toArray(self):
        # from Matrix (obj) converter
        array = []
        for i in range(self.rows):
            for j in range(self.columns):
                array.append(self.matrix[i][j])
        return array

    def multiply(self, n):
        # hadamard product
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.columns):
                    self.matrix[i][j] *= n.matrix[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    self.matrix[i][j] *= n

    def randomize(self):
        for i in range(self.rows):
            for j in range(self.columns):
                value = random.uniform(-1, 1)
                self.matrix[i][j] = value

    def add(self, n):
        if isinstance(n, Matrix):
            for i in range(self.rows):
                for j in range(self.columns):
                    self.matrix[i][j] += n.matrix[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    self.matrix[i][j] += n

    def subtract(self, n):
        result = Matrix(self.rows, n.columns)
        if isinstance(n, Matrix):
            for i in range(result.rows):
                for j in range(result.columns):
                    result.matrix[i][j] = self.matrix[i][j] - n.matrix[i][j]
            return result
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    result.matrix[i][j] = self.matrix[i][j] - n

    def transpose(self, matrix):
        result = Matrix(self.columns, self.rows)
        for i in range(matrix.rows):
            for j in range(matrix.columns):
                result.matrix[j][i] = matrix.matrix[i][j]
        return result

    def map(self, func):
        for i in range(self.rows):
            for j in range(self.columns):
                val = self.matrix[i][j]
                self.matrix[i][j] = func(val)
