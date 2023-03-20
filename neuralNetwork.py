from matrix import Matrix, fromArray, mapping, multiplication
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_layers, hidden_nodes, output_nodes, bias_switch):
        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.hidden_container = []
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = 0.25
        self.momentum = 0.0
        self.rounds = 0
        # default settings for bias switch
        self.bias_switch = True
        # switch off (if necessary)
        if bias_switch == 0:
            self.bias_switch = False
        # Multi hidden layer setup
        # /////////////////////////////
        for i in range(hidden_layers - 1):
            weights_hh = Matrix(self.hidden_nodes, self.hidden_nodes)
            bias_hh = Matrix(self.hidden_nodes, 1)
            weights_hh.randomize()
            bias_hh.randomize()
            container = [weights_hh, bias_hh]
            self.hidden_container.append(container)
        # /////////////////////////////

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()
        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

        # previous weights deltas of individual layer
        self.prev_weights_ih_deltas = Matrix(self.hidden_nodes, self.input_nodes)
        self.prev_weights_ho_deltas = Matrix(self.output_nodes, self.hidden_nodes)
        first_hh_weight_delta = Matrix(self.hidden_nodes, self.hidden_nodes)
        self.prev_weights_hh_deltas = []
        self.prev_weights_hh_deltas.append(first_hh_weight_delta)
        open("Errors.txt", "w").close()

    def feedforward(self, input_array):
        hiddens_container = []
        # hidden layer
        inputs = fromArray(input_array)
        hidden = multiplication(self.weights_ih, inputs)
        if self.bias_switch:
            hidden.add(self.bias_h)
        hidden.map(sigmoid)
        # activation function
        # Multi hidden layer feedforward concept (only if has to)
        # /////////////////////////////
        if self.hidden_layers > 1:
            next_h_input = hidden
            for n in self.hidden_container:
                hiddens = multiplication(n[0], next_h_input)
                if self.bias_switch:
                    hiddens.add(n[1])
                hiddens.map(sigmoid)
                hiddens_container.append(hiddens)
                next_h_input = hiddens
            hidden = next_h_input
        # /////////////////////////////
        # output layer
        output = multiplication(self.weights_ho, hidden)
        if self.bias_switch:
            output.add(self.bias_o)
        output.map(sigmoid)
        result = [output, hidden, hiddens_container, inputs]

        return result

    def train(self, input_array, target_array):
        product = self.feedforward(input_array)
        outputs = product[0]
        hidden = product[1]
        hiddens_container = product[2]
        inputs = product[3]

        targets = fromArray(target_array)

        # CALCULATE OUTPUT-HIDDEN LAYER ERRORS
        # error = targets - outputs
        output_error = targets.subtract(outputs)
        if self.rounds % 500 == 0:
            f = open("Errors.txt", "a")
            f.write(str(self.rounds) + " : \n" + str(output_error.matrix) + "\n")
        self.rounds += 1
        gradients = mapping(outputs, dsigmoid)
        gradients.multiply(output_error)
        gradients.multiply(self.learning_rate)
        # deltas calculations
        hidden_T = hidden.transpose(hidden)
        weight_ho_deltas = multiplication(gradients, hidden_T)
        # Consideration of the momentum factor in backprop for O-H layer
        # ////////////////////////////////////
        if self.momentum != 0:
            self.prev_weights_ho_deltas.multiply(self.momentum)
            weight_ho_deltas.add(self.prev_weights_ho_deltas)
        self.prev_weights_ho_deltas = weight_ho_deltas
        # ////////////////////////////////////
        # Adjust the weights by deltas
        self.weights_ho.add(weight_ho_deltas)
        # adjust bias by its deltas... (by gradients)[if necessary]
        if self.bias_switch:
            self.bias_o.add(gradients)

        # Multi hidden layer (if exists)
        # HIDDEN-HIDDEN ERRORS
        # /////////////////////////////
        if self.hidden_layers > 1:
            iterator = 0
            previous_hidden = hiddens_container[0]
            for n in self.hidden_container:
                hidden_hidden = hiddens_container[iterator]
                # MAKING THE RIGHT ORDER OF THE LAYERS (I->H->H_H->O) [backwards] (0->H_H->H->I)
                if iterator == len(self.hidden_container) - 1:
                    hidden_hidden = hidden
                weight_hh_t = n[0].transpose(n[0])
                hidden_layer_errors = multiplication(weight_hh_t, hidden_hidden)
                hidden_layer_gradient = mapping(hidden_hidden, dsigmoid)
                hidden_layer_gradient.multiply(hidden_layer_errors)
                hidden_layer_gradient.multiply(self.learning_rate)
                next_inputs_T = previous_hidden.transpose(previous_hidden)
                weight_hh_deltas = multiplication(hidden_layer_gradient, next_inputs_T)
                # Momentum factor for H-H layer
                # ////////////////////////////////////
                if self.momentum != 0:
                    self.prev_weights_hh_deltas[iterator].multiply(self.momentum)
                    weight_hh_deltas.add(self.prev_weights_hh_deltas[iterator])
                self.prev_weights_hh_deltas.append(weight_hh_deltas)
                # ////////////////////////////////////
                n[0].add(weight_hh_deltas)
                if self.bias_switch:
                    n[1].add(hidden_layer_gradient)
                previous_hidden = hidden_hidden
                iterator += 1

        # /////////////////////////////

        # CALCULATE HIDDEN-INPUT LAYER ERRORS
        weights_ho_t = self.weights_ho.transpose(self.weights_ho)
        hidden_errors = multiplication(weights_ho_t, output_error)
        hidden_gradient = mapping(hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.learning_rate)
        # deltas calculations
        inputs_T = inputs.transpose(inputs)
        weights_ih_deltas = multiplication(hidden_gradient, inputs_T)
        # Momentum factor for I-H layer
        # ////////////////////////////////////
        if self.momentum != 0:
            self.prev_weights_ih_deltas.multiply(self.momentum)
            weights_ih_deltas.add(self.prev_weights_ih_deltas)
        self.prev_weights_ih_deltas = weights_ih_deltas
        # ////////////////////////////////////
        # Adjust the weights by deltas
        self.weights_ih.add(weights_ih_deltas)
        # adjust bias by its deltas... (by gradients)
        if self.bias_switch:
            self.bias_h.add(hidden_gradient)

        whole_error = targets.subtract(outputs)
        sum_error = 0
        for i in range(whole_error.rows):
            for j in range(whole_error.columns):
                sum_error += abs(whole_error.matrix[i][j])
        return abs(sum_error/self.bias_o.rows)

    def feedforward_result(self, input_array, target):
        result = self.feedforward(input_array)
        outputs = result[0]
        hidden = result[1]
        inputs = result[3]
        target = fromArray(target)
        f = open("Testing stats.txt", "w")
        print("Input array given to network:")
        f.write("Input array given to network:\n")
        print(inputs.matrix)
        f.write(str(inputs.matrix) + "\n")
        print()
        print("Error of network for the entire pattern:")
        f.write("Error of network for the entire pattern:\n")
        whole_error = target.subtract(outputs)
        sum_error = 0
        for i in range(whole_error.rows):
            for j in range(whole_error.columns):
                sum_error += abs(whole_error.matrix[i][j])
        print(abs(sum_error/self.bias_o.rows))
        f.write(str(abs(sum_error)) + "\n")
        print()
        print("Target array:")
        f.write("Target array:\n")
        print(target.matrix)
        f.write(str(target.matrix) + "\n")
        print()
        print("Errors at individual outputs:")
        f.write("Errors at individual outputs:\n")
        print(target.subtract(outputs).matrix)
        f.write(str(target.subtract(outputs).matrix) + "\n")
        print()
        print("Weighted sum of output layer (result):")
        f.write("Weighted sum of output layer (result):\n")
        print(outputs.matrix)
        f.write(str(outputs.matrix) + "\n")
        print()
        print("Weights of Hidden-Output layer:")
        f.write("Weights of Hidden-Output layer:\n")
        print(self.weights_ho.matrix)
        f.write(str(self.weights_ho.matrix) + "\n")
        print()
        print("Weighted sum of hidden layer:")
        f.write("Weighted sum of hidden layer:\n")
        print(hidden.matrix)
        f.write(str(hidden.matrix) + "\n")
        print()
        print("Weights of Input-Hidden layer:")
        f.write("Weights of Input-Hidden layer:\n")
        print(self.weights_ih.matrix)
        f.write(str(self.weights_ih.matrix) + "\n")
        print()

    def write_network_to_file(self, filename):
        f = open(filename + ".txt", "w")

        for i in range(self.weights_ih.rows):
            for j in range(self.weights_ih.columns):
                f.write(str(self.weights_ih.matrix[i][j]))
                if j + 1 != self.weights_ih.columns:
                    f.write(" ")
            if i + 1 != self.weights_ih.rows:
                f.write("\n")
        f.write("endline")

        for i in range(self.bias_h.rows):
            for j in range(self.bias_h.columns):
                f.write(str(self.bias_h.matrix[i][j]))
                if j + 1 != self.bias_h.columns:
                    f.write(" ")
            if i + 1 != self.bias_h.rows:
                f.write("\n")
        f.write("endline")

        for i in range(self.weights_ho.rows):
            for j in range(self.weights_ho.columns):
                f.write(str(self.weights_ho.matrix[i][j]))
                if j + 1 != self.weights_ho.columns:
                    f.write(" ")
            if i + 1 != self.weights_ho.rows:
                f.write("\n")
        f.write("endline")

        for i in range(self.bias_o.rows):
            for j in range(self.bias_o.columns):
                f.write(str(self.bias_o.matrix[i][j]))
                if j + 1 != self.bias_o.columns:
                    f.write(" ")
            if i + 1 != self.bias_o.rows:
                f.write("\n")
        f.write("endline")

        for each in self.hidden_container:
            for i in range(each[0].rows):
                for j in range(each[0].columns):
                    f.write(str(each[0].matrix[i][j]))
                    if j + 1 != each[0].columns:
                        f.write(" ")
                if i + 1 != each[0].rows:
                    f.write("\n")
            f.write("endline")
            for i in range(each[1].rows):
                for j in range(each[1].columns):
                    f.write(str(each[1].matrix[i][j]))
                    if j + 1 != each[1].columns:
                        f.write(" ")
                if i + 1 != each[1].rows:
                    f.write("\n")
            if len(self.hidden_container) > 1 and self.hidden_container[len(self.hidden_container) - 1] != each:
                f.write("endline")

    def read_network_from_file(self, filename):
        everything = []
        f = open(filename + ".txt")
        lines = f.read()
        f.close()
        len_row = 0
        len_col = 0
        iterator_i = 0
        iterator_j = 0
        lines = lines.split("endline")
        for each in lines:
            rows = each.split("\n")
            len_row = len(rows)
            for row in rows:
                elements = row.split(" ")
                len_col = len(elements)
            new_matrix = Matrix(len_row, len_col)
            iterator_i = 0
            for row in rows:
                elements = row.split(" ")
                for element in elements:
                    new_element = float(element)
                    new_matrix.matrix[iterator_i][iterator_j] = new_element
                    iterator_j += 1
                iterator_j = 0
                iterator_i += 1
            everything.append(new_matrix)

        self.weights_ih = everything[0]
        self.bias_h = everything[1]
        self.weights_ho = everything[2]
        self.bias_o = everything[3]
        self.hidden_container.clear()
        trashcan = []
        for i in range(len(everything)):
            if i > 3:
                trashcan.append(everything[i])
        len_of_hidden_hidden = int(len(trashcan) / 2)

        iterator = 0
        for i in range(len_of_hidden_hidden):
            new = []
            for j in range(2):
                new.append(trashcan[iterator])
                iterator += 1
            self.hidden_container.append(new)
