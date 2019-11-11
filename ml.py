import tensorflow as tf
import numpy as np




###############################################################################
###### LOSS FUNCTIONS
###############################################################################

def L2_loss(y_pred, y_label):
	"""
	Given a set of predictions and set of corresponding labels, compute the mean square error:  
	loss = (y_label - y_pred)^2 / N  
	where N is the number of predictions
	"""
	# print(f"y_label shape {y_label.shape}, y_pred shape {y_pred.shape}, {tf.reduce_mean(tf.square(y_label - y_pred))}")
	return tf.reduce_mean(tf.square(y_label - y_pred))



def weight_decay_loss(W, coefficient):
	"""
	Given a set of weights and a lambda coefficient, compute the loss. Generally, weights that
	are higher in magnitude are penalized more.
	"""