'''
Francesco Giovanelli - March 2019

Creates a dataset of unfeasible assignments, by using as a reference a dataset containing feasible assignments
'''

import pandas as pd
import math
import numpy as np
import random
import my_utils
import os

'''
Loads data from feasible assignments dataset
'''
def load_feasible_data(feasible_dataset, full_solutions_dataset):
	print("Loading feasible data...")

	#load datasets from files
	full_feasible_df = pd.read_csv(feasible_dataset, delimiter="-", names = ["status", "assignment", "feasible"])
	full_solutions_df = pd.read_fwf(full_solutions_dataset, widths=[1] * 64, header=None)

	#convert single input column in multiple columns (one column for each 'bit' of status)
	status_df = full_feasible_df['status'].apply(lambda s: pd.Series(list(s)).astype(int))
	assignment_df = full_feasible_df['assignment'].apply(lambda a: pd.Series(list(a)).astype(int))
	feasible_df = full_feasible_df['feasible']

	# print dataframes structure
	print("Full DF:")
	print(full_feasible_df.shape)
	print("Full solutions DF:")
	print(full_solutions_df.shape)
	print("status DF:")
	print(status_df.head())
	print(status_df.shape)
	print("assignment DF:")
	print(assignment_df.shape)
	print("feasible DF:")
	print(feasible_df.shape)

	return status_df, assignment_df, feasible_df, full_solutions_df

'''
Creates a compact representation of the feasible status dataset, by merging together status and assignment
Final result is: 64 bit of partial solution + "-" + 1 bit for flag feasible/not feasible (set to 1)
'''
def create_compact_feasible_ds(feasible_solutions_df, feas_assignment_df):
	print("Creating compact representation of the feasible solutions dataset...")

	feasible_compact_df = feasible_solutions_df.add(feas_assignment_df)

	for index, status in feasible_compact_df.iterrows():
		#convert status to string elements
		str_status = status.map(str)
		#convert status to a sequence of 0s and 1s
		collapsed_status = str_status.str.cat(sep="")
		#print to file the new status, along with the "feasible bit" set to 1
		print(collapsed_status + "-1", file=open("DS.FEASIBLE.txt", "a"))


'''
Creates an illegal assignment for each feasible solution, by placing a queen in the same row/column/diagonal of an existing one
'''
def create_illegal_assignments(feasible_solutions_df):
	print("Creating solutions with illegal assignments...")

	for index, status in feasible_solutions_df.iterrows():
		#pick a random illegal assignment type (0: row ; 1: column ; 2: diagonal)
		row_col_diag = random.randint(0,2)
		#get postition of "1s" (queens) in the linearized chessboard
		ones_position = status[(status == 1)].index
		selected_queen_pos = 0
		illegal_queen_position = 0
		
		if len(ones_position) == 0:
			#no queen is present, so go to next status
			continue
		else:
			#if at least a "1" (queen) is present in the chessboard, pick one a random
			selected_queen_pos = random.choice(ones_position)
		
		#insert a queen (1) in an illegal position, based on the position of the selected target queen
		#ROW alteration
		if row_col_diag == 0:
			#get queen's column (a value between [0,7])
			col = int(selected_queen_pos % 8)
			#compute shifting from original position
			shifting = random.randint((col * -1), (7 - col))
			#if shifting is 0 repeat a random pick
			while shifting == 0:
				shifting = random.randint((col * -1), (7 - col))
			#compute new (illegal) queen's position, based on shifting
			illegal_queen_position = shifting + selected_queen_pos

		#COLUMN alteration
		elif row_col_diag == 1:
			#get queen's corresponding row (a value between [0,7])
			row = int(selected_queen_pos / 8)

			if row < 7:
				#for the first 7 rows insert the new queen after 8 position from the selected one
				illegal_queen_position = selected_queen_pos + 8
			else:
				#for last row insert the new queen 8 position before the selected one
				illegal_queen_position = selected_queen_pos - 8

		#DIAGONAL alteration
		elif row_col_diag == 2:
			#get queen's corresponding row (a value between [0,7])
			row = int(selected_queen_pos / 8)
			#get queen's column (a value between [0,7])
			col = int(selected_queen_pos % 8)

			if row != 7 and col != 7:
				illegal_queen_position = selected_queen_pos + 9
			elif row != 7 and col == 7:
				illegal_queen_position = selected_queen_pos + 7
			elif row == 7 and col == 0:
				illegal_queen_position = selected_queen_pos - 7
			elif row == 7 and col != 0:
				illegal_queen_position = selected_queen_pos - 9

		#set the new illegal queen position to "1" in the original status
		status[illegal_queen_position] = 1

		#convert status to string elements
		str_status = status.map(str)
		#convert status to a sequence of 0s and 1s
		collapsed_status = str_status.str.cat(sep="")
		#print to file the new status, along with the "feasible bit" set to 0
		print(collapsed_status + "-0", file=open("DS.UNFEASIBLE.FULL.txt", "a"))


'''
Creates unfeasible partial solutions that cannot generate any full feasible solution
'''
def create_unfeasible_partial_solutions(feasible_solutions_df, full_solutions_df):
	print("Creating unfeasible partial solutions...")

	for index, status in feasible_solutions_df.iterrows():
		#get postition of "1s" (queens) in the linearized chessboard (status)
		ones_position = status[(status == 1)].index
		#get random position for new queen to insert into a feasible status
		rand_queen_position = random.randint(0,63)
		while rand_queen_position in ones_position:
			rand_queen_position = random.randint(0,63)

		#set the new queen position to "1" in the original status
		status[rand_queen_position] = 1

		#check if partial solution is feasible or not
		feasible = my_utils.check_single_solution_feasibility(status, full_solutions_df)

		#if partial solution is unfeasible, it needs to be saved into the "unfeasible" dataset
		if not feasible:
			#convert status to string elements
			str_status = status.map(str)
			#convert status to a sequence of 0s and 1s
			collapsed_status = str_status.str.cat(sep="")
			#print to file the new status, along with the "feasible bit" set to 0
			print(collapsed_status + "-0", file=open("DS.UNFEASIBLE.FULL.txt", "a"))


'''
Creates a dataset with only an instance of a particular solution, useful for the unfeasible dataset
'''
def delete_duplicate_solutions(unfeasible_sol_file):
	print("Eliminating duplicate unfeasible solutions...")

	unique_solutions = set(open(unfeasible_sol_file).readlines())
	out_f = open("DS.UNFEASIBLE.txt", 'w').writelines(set(unique_solutions))

	# remove unfeasible ds with duplicates
	os.remove(unfeasible_sol_file)	

'''
TODO: da completare, ho fatto il merge dei dataset a mano

Creates a full dataset, containing both feasible and unfeasible solutions

Arguments:
	full_ds_name (string): name of the final full dataset file
	negative_data_ratio (float): proportion of negative data in the final dataset

def create_final_dataset(full_ds_name, negative_data_ratio=0.4):
	feasible_count = 0
	unfeasible_count = 0
	positive_data_ratio = 1.0 - negative_data_ratio

	with open(full_ds_name, 'w') as outfile:
		#feasible dataset
		with open("DS.FEASIBLE.txt") as infile:
			for line in infile:
				outfile.write(line)
				feasible_count += 1

		#compute the max number of expected negative data in the final datset file
		unfeasible_max = feasible_count * negative_data_ratio / positive_data_ratio

		#unfeasible dataset
		with open("DS.UNFEASIBLE.txt") as infile:
			for line in infile:
				outfile.write(line)
				unfeasible_count += 1
				#stop when the threshold is reached
				if unfeasible_max == unfeasible_count:
					break			
				
'''

'''
Main
'''
if __name__ == "__main__":
	#load feasible partial and full solutions
	feas_solutions_df, feas_assignment_df, feas_output_df, full_solutions_df= load_feasible_data(feasible_dataset = "DS.MULTIPLE.B.txt", full_solutions_dataset = "DS.FULL.SOLUTIONS.txt")
	#merge partial solutions and assignments for the partial solution dataset
	create_compact_feasible_ds(feas_solutions_df, feas_assignment_df)
	#create unfeasible assignments by violating row, column or diagonal constraints
	create_illegal_assignments(feas_solutions_df)
	#create unfeasible assignments by inserting a queen in a random position, starting from a feasible partial solution
	create_unfeasible_partial_solutions(feas_solutions_df, full_solutions_df)
	#eliminate duplicate partial solutions from the unfeasible solutions dataset
	delete_duplicate_solutions("DS.UNFEASIBLE.FULL.txt")