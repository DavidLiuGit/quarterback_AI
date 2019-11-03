import numpy as np
from pprint import pprint
from sportsreference.nfl.teams import Teams
from progress.bar import ChargingBar
# from sportsreference.nfl.schedule import Schedule


YEAR = 2018

def nfl_build_season_passing_data(year):
	"""Build QB Passing dataset from NFL boxscores, for a specified season"""

	# get a list of all the teams that competed in the specified year
	print(f"Gathering passing data for {year} NFL season:")
	teams = Teams(str(year))

	# init empty lists for input and output labels; these lists will be populated as matrices
	input_arr = []
	label_arr = []

	# display progress bar
	bar = ChargingBar('Analyzing teams', max=len(teams))

	# iterate over each team in the list of teams
	for team in teams:
		# get the schedule (list of games) for the team
		games = team.schedule

		# for every game in the schedule, if the game has been played, extract:
		# the team's passing stats, and the result of the game
		for game in games:
			# if there is no (raw) result for the game, skip it (probably has yet to be played)
			if not game._result:
				continue

			# append an input dataset to input_arr list (matrix)
			input_arr.append(nfl_build_game_passing_data(game))

			# append a set of labels to label_arr list (matrix)
			label_arr.append(nfl_build_game_labels(game))

		# increment progress bar
		bar.next()

	# sanity check: lengths of input_arr and label_arr must be the same
	assert len(input_arr)==len(label_arr), "Error: input & label array lengths mismatch!"
	
	# finish and return datasets
	bar.finish()
	return input_arr, label_arr



def nfl_build_game_passing_data(game):
	"""
	Build passing dataset from a single instance of `sportsreference.nfl.schedule.Game`.
	Return list of passing metrics
	"""
	return [
		game.pass_yards,                # YDS
		game.pass_attempts,				# ATT
		game.pass_completions,			# COMP
		game.pass_touchdowns,			# TDs
		game.interceptions,				# INTs
		game.times_sacked,				# SCKs
		game.yards_lost_from_sacks,		# self explanatory
	]



def nfl_build_game_labels(game):
	"""
	Build game labels list from a single instance of `sportsreference.nfl.schedule.Game`.
	"""
	final_result_dict = {
		"L": -1,
		"W": 1,
		"T": 0
	}

	return [
		final_result_dict[game._result],				# game result
		# timestamp_to_seconds(game.time_of_possession),	# time of possession in seconds
	]



###############################################################################
###### HELPER FUNCTIONS
###############################################################################

def timestamp_to_seconds(timestamp):
	"""
	Convert a timestamp in the `MM:SS` format to integer number of seconds
	"""
	time_components = timestamp.split(':')
	total_seconds = int(time_components[0]) * 60 + int(time_components[1])
	return total_seconds



def save_result_npy(data, year, file_prefix):
	"""
	Convert matrix to numpy array and save it locally
	"""
	arr = np.array(data)
	np.save(f"{file_prefix}_{year}.npy", arr)



if __name__=='__main__':
	year_list = [2018, 2017, 2016, 2015, 2014, 2013]

	for year in year_list:
		input_arr, label_arr = nfl_build_season_passing_data(year)
		save_result_npy(input_arr, year, 'nfl_passing')
		save_result_npy(label_arr, year, 'nfl_label')