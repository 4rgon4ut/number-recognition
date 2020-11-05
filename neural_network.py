import numpy

from numpy.random import normal
from scipy.special import expit


class NeuralNetwork:
    """
    Neural network class with 3 layers.
    Starting weights initializing from a Gaussian distribution.
    Activation function is a sigmoid.
    Backpropogation training algorithm.
    """
    
    def __init__(self, input_nodes, hidden_nodes,
                 output_nodes, learning_rate):
        
        # set number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.learning_rate = learning_rate
        
        # initialize weights from a normal (Gaussian) distribution
        # i - input layer, h - hidden layer, o - output layer
        # weights on i --> h 
        self.w_i_h = normal(
            0.0,
            pow(self.hidden_nodes, -0.5),
            (self.hidden_nodes, self.input_nodes)
        )
        # weights on h --> o
        self.w_h_o = normal(
            0.0,
            pow(self.output_nodes, -0.5),
            (self.output_nodes, self.hidden_nodes)
        )
        # activation sigmoid function
        self.activation_function = lambda x: expit(x) 
        
        pass    
    
    def calculate(self, inputs):
        """
        Calculating input and output signals on every layer
        """
        # in and out signals for hidden layer
        hidden_inputs = numpy.dot(self.w_i_h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # in and out signals for output layer
        final_inputs = numpy.dot(self.w_h_o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        result = {
            'hidden_inputs': hidden_inputs, 
            'hidden_outputs': hidden_outputs,
            'final_inputs': final_inputs,
            'final_outputs': final_outputs
        }
        return result
    
    def train(self, inputs_list, targets_list):
        """
        Updates weights on connections between neurons
        """
        # convert data to two-dimensional array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T        
        
        # in and out values for layers
        model_values = self.calculate(inputs)
        final_output = model_values['final_output']
        hidden_output = model_values['hidden_output']
        
        # calculating errors
        output_errors = targets - final_output
        hidden_errors = numpy.dot(self.w_h_o.T, output_errors)
        
        # update weights on hidden --> output layers
        self.w_h_o += self.learning_rate * numpy.dot(
            (output_errors * final_output * (1.0 - final_output)),
            numpy.transpose(hidden_output)
        )
        # update weights on input --> hidden layers
        self.w_i_h += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_output * (1.0 - hidden_output)),
            numpy.transpose(inputs)
        )
        
        pass
