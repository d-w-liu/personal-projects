# This code is mainly just a proof of concept to see if I can write a neural network that will
# separate background from noise for two separate variables, while keeping the variables uncorrelated.

import numpy as np
import math

seed()

class NeuralNetwork:
	def __init__(self, input_data, output_var1, output_var2):
		self.input = input_data
		self.weights1 = np.random.rand(self.input.shape[1],4)
		self.weights2 = np.random.rand(4, 1)
		self.weights3 = np.random.rand(4, 1)
		self.var1 = output_var1
		self.var2 = output_var2
		self.output1 = np.zeros(var1.shape)
		self.output2 = np.zeros(var2.shape)

	def feedforward(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.output1 = sigmoid(np.dot(self.layer1, self.weights2))
		self.output2 = sigmoid(np.dot(self.layer1, self.weights3))

	def check_correlation(self):
		covariance = np.cov(self.output1, self.output2)
		pearson_correlation = covariance[0][1] / (self.output1.std() * self.output2.std())
		return pearson_correlation
	
	def backpropagator(self):
		# this backpropagator needs to be the derivative of the loss function.
		# so I need to figure out what the loss function is in order to make this work.
		
		self.weights1 += d_weights1
		self.weights2 += d_weights2
		self.weights3 += d_weights3