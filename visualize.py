import matplotlib.pyplot as plt
import numpy as np


NFL_PASSING_PREFIX = "nfl_passing"
NFL_LABEL_PREFIX = "nfl_label"
def visualize_data_from_file(input_file_name, label_file_name, input_axis, label_axis):
	"""
	Load dataset from file. Input and labels are each expected to be saved in their own files.
	"""
	# load from file
	X = np.load(input_file_name)
	Y = np.load(label_file_name)

	# sanity checks
	print(X.shape, Y.shape)

	# call visualize function
	visualize_data(X, Y, input_axis, label_axis)



###############################################################################
###### HELPER FUNCTIONS
###############################################################################

def visualize_data(X, Y, x_axis, y_axis):
	"""
	Create a 2D plot of some input (X) and output (Y). 1 axis must be selected (by index number)
	from both the X and Y matrices. Lengths of the selected axes must match
	"""
	plt.plot(X[x_axis], Y[y_axis])
	plt.show()



if __name__=='__main__':
	year = 2018
	visualize_data_from_file(
		f'nfl_passing_{year}.npy',
		f'nfl_label_{year}.npy',
		0, 0
	)