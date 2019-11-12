import tensorflow as tf
import numpy as np
from tensorflow import keras
# from tensorflow.keras import layers



# INPUT_FILE_PREFIX



def build_logistic_regression_model(input_dims, output_dims):
	# define the input layer
	inputs = keras.Input(shape=(input_dims,))

	# define a dense layer and connect input to it
	dense = keras.layers.Dense(output_dims, activation="sigmoid")
	outputs = dense(inputs)

	# construct the model with the layers defined above
	return keras.Model(inputs=inputs, outputs=outputs, name="logistic_model")
	


def logreg_train(model, X, Y, batch=32, epochs=50, valid_split=0.2):
	"""
	Fit a keras model
	"""
	history = model.fit(
		X, Y,				# inputs & labels
		batch_size=batch,
		epochs=epochs,
		validation_split=0.2
	)

	return history



def logistic_regression():
	model = build_logistic_regression_model(7, 1)

	# show the model
	model.summary()

	# compile the model; try out different loss functions, optimizers, and metrics here
	model.compile (
		loss=tf.losses.BinaryCrossentropy(),
		optimizer=tf.keras.optimizers.Adam(),
		metrics=['accuracy']
	)

	# train the model
	X = tf.constant(np.load('data/nfl_passing_2018.npy'), dtype=float)
	Y = tf.constant(np.load('data/nfl_label_2018.npy'), dtype=float)
	history = model.fit(X,Y, batch_size=32, epochs=2000, validation_split=0.2)

	# evaluate the accuracy of the model
	X = tf.constant(np.load('data/nfl_passing_2017.npy'), dtype=float)
	Y = tf.constant(np.load('data/nfl_label_2017.npy'), dtype=float)
	scores = model.evaluate(X, Y)
	print(scores)

	# examine the model here
	breakpoint()




if __name__=='__main__':
	logistic_regression()