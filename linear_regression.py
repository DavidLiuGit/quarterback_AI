import pdb
import numpy as np
import tensorflow as tf

from ml import L2_loss, weight_decay_loss



class LinearRegressionModel:
	def __init__(self, dims):
		weight_matrix_shape = (dims, 1)
		self.W = tf.Variable(tf.zeros(weight_matrix_shape))
		self.b = tf.Variable(0.0)


	def __call__(self, X):
		return self.predict(X)


	def predict(self, X):
		"""
		Use the model to make a prediction. y = XW + b
		"""
		# return self.W * X + self.b
		return tf.matmul(X, self.W) + self.b


	def train(self, X, Y, loss_func, learn_rate=0.005):
		"""
		Define a single iteration (or batch) of training on this model
		"""

		# define the steps to obtaining loss; `GradientTape` will be used to compute gradient
		with tf.GradientTape() as t:
			y_pred = self.predict(X)
			loss = loss_func(y_pred, Y)

		# compute gradients wrt Weights and bias
		dW, db = t.gradient(loss, [self.W, self.b])

		# pdb.set_trace()

		# apply the gradients by subtracting from the current weight & bias, in proportion to the learning rate
		self.W.assign_sub(learn_rate * dW)
		self.b.assign_sub(learn_rate * db)
	
		# return the loss computed in this iteration
		return loss





def linreg_train():
	# define learning parameters
	epochs			= range(500000)
	learning_rate	= 0.000013
	batch_size		= 10

	# load data
	X = tf.constant(np.load('nfl_passing_2018.npy'), dtype=float)
	Y = tf.constant(np.load('nfl_label_2018.npy'), dtype=float)

	# initialize a linear regression model to train
	model = LinearRegressionModel(X.shape[1])

	# collect loss values as training progresses
	loss_history = []
	
	# define training loop
	for epoch in epochs:
		curr_loss = model.train(X, Y, L2_loss, learning_rate)
		loss_history.append(curr_loss)

	print(f"Final loss: {loss_history[-1]}")

	# breakpoint for inspection
	pdb.set_trace()



if __name__=='__main__':
	linreg_train()