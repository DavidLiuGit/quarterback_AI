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

	# extract the axes from the X & Y matrices
	x = X[:,input_axis]
	y = Y[:,label_axis]

	# call visualize function
	visualize_data(x,y)


###############################################################################
###### HELPER FUNCTIONS
###############################################################################

def visualize_data(X, Y):
	"""
	Create a 2D plot of some input (X) and output (Y). 1 axis must be selected (by index number)
	from both the X and Y matrices. Lengths of the selected axes must match
	"""
	plt.scatter(X, Y)
	plt.show()



if __name__=='__main__':
	year = 2018
	for i in range(7):
		visualize_data_from_file(
			f'nfl_passing_{year}.npy',
			f'nfl_label_{year}.npy',
			i, 0
		)