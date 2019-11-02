import numpy as np
from pprint import pprint
from sportsreference.nfl.teams import Teams
from progress.bar import ChargingBar


YEAR = 2018

def nfl_build_passing_data(year):
	"""Build QB Passing database from NFL boxscores"""

	# get a list of all the teams that competed in the specified year
	print(f"Gathering passing data for {year} NFL season...")
	teams = Teams(str(year))

	# init empty lists for input and output labels; these lists will be populated as matrices
	input_arr = []
	label_arr = []

	# iterate over each team in the list of teams
	for team in teams:
		# get the schedule (list of games) for the team
		games = team.schedule

		# display progress bar
		print(f"{team.name}: {len(games)} games:")
		bar = ChargingBar('Games', max=len(games))

		# for every game in the schedule, if the game has been played, extract:
		# the team's passing stats, and the result of the game
		for game in games:
			# if there is no boxscore for the game, skip it (probably has yet to be played)
			if not game.boxscore:
				continue

			# append an input dataset to input_arr list (matrix)
			input_arr.append([
				game.pass_yards,                # YDS
				game.pass_attempts,				# ATT
				game.pass_completions,			# COMP
				game.pass_touchdowns,			# TDs
				game.interceptions,				# INTs
				game.times_sacked,				# SCKs
				game.yards_lost_from_sacks,		# self explanatory
			])

			# append a set of labels to label_arr list (matrix)
			print(f"{game.result}")
			label_arr.append([
				game.result
			])

			# increment progress bar
			bar.next()

		bar.finish()




if __name__=='__main__':
	nfl_build_passing_data(YEAR)