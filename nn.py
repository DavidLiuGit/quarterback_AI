import tensorflow as tf
import numpy as np
from tensorflow import keras


def build_neural_network_model(input_dims, hidden_layer_dims, output_dims):
	"""
	Define a neural network with a single hidden layer.
	"""
	# define input layer
	inputs = keras.Input(shape=(input_dims,))

	# define hidden layers
	dense = keras.layers.Dense(hidden_layer_dims, activation="relu")(inputs)

	# define output layer
	outputs = keras.layers.Dense(output_dims, activation="sigmoid")(dense)

	return keras.Model(inputs=inputs, outputs=outputs, name="nn")



def nn_train(model, X, Y, batch=32, epochs=500, valid_split=0.2):
	"""
	Train neural network model
	"""
	return model.fit (
		X, Y,
		batch_size=batch,
		epochs=epochs,
		validation_split=0.2
	)



def neural_network():
	# define the model
	model = build_neural_network_model(7, 21, 1)

	# summarize model
	model.summary()

	# compile the model
	model.compile (
		loss=tf.losses.BinaryCrossentropy(),
		optimizer=tf.keras.optimizers.Adam(),
		metrics=['accuracy']
	)

	# train the model
	X = tf.constant(np.load('data/nfl_passing_6yrs.npy'), dtype=float)
	Y = tf.constant(np.load('data/nfl_label_6yrs.npy'), dtype=float)
	history = nn_train(model, X, Y, epochs=1000)

	# evaluate acc
	X = tf.constant(np.load('data/nfl_passing_2017.npy'), dtype=float)
	Y = tf.constant(np.load('data/nfl_label_2017.npy'), dtype=float)
	scores = model.evaluate(X, Y)
	print(scores)

	breakpoint()




if __name__=='__main__':
	neural_network()