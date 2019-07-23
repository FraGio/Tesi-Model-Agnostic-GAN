'''
Francesco Giovanelli - March 2019

Utils for generator and discriminator networks
'''

import pandas as pd
import math
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model   
from keras.layers import * 
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
import matplotlib.pylab as plt
import sys
import logging
from datetime import datetime

import generator

'''
Loads GENERATOR data from file and generates proper dataset (training + validation + test)
'''
def load_generator_data(training_data_file, test_data_file):
	#load datasets from files
	training_dataframe = pd.read_csv(training_data_file, delimiter="-", names = ["x", "y"])
	test_dataframe = pd.read_csv(test_data_file, delimiter="-", names = ["x", "y"])

	#TRAINING data
	#convert single input column in multiple columns (one column for each 'bit' of status)
	x_training_dataframe = training_dataframe['x'].apply(lambda x: pd.Series(list(x))).astype(dtype=np.float32)
	#convert labels from string to array one-hot encoded
	y_training_dataframe = training_dataframe['y'].apply(lambda y: pd.Series(list(y))).astype(dtype=np.float32)

	#TEST data
	#convert single input column in multiple columns (one column for each 'bit' of status)
	x_test = test_dataframe['x'].apply(lambda x: pd.Series(list(x))).astype(dtype=np.float32)
	#convert labels from string to array one-hot encoded
	y_test = test_dataframe['y'].apply(lambda y: pd.Series(list(y))).astype(dtype=np.float32)

	#split full training dataset into training (2/3) and validation sets (1/3)
	x_train, x_val, y_train, y_val = train_test_split(x_training_dataframe, y_training_dataframe, test_size=1/3)

	print("train, val, test set shapes:")
	print(x_train.shape)
	print(x_val.shape)
	print(x_test.shape)

	return x_train, y_train, x_val, y_val, x_test, y_test


'''
Loads DISCRIMINATOR data from file and generates proper dataset (training + validation + test)
Different from the original loader, beacause it uses two files where each file is a dataset containing both feasible and unfeasible solutions
'''
def load_discriminator_data_v2(train_dataset, test_dataset, validation=True):
	# load datasets from files
	train_df = pd.read_csv(train_dataset, delimiter="-", names = ["x", "y"])
	test_df = pd.read_csv(test_dataset, delimiter="-", names = ["x", "y"])

	# shuffle datasets
	train_df = shuffle(train_df)
	test_df = shuffle(test_df)
	
	# convert single input column in multiple columns (one column for each 'bit' of status)
	train_x_df = train_df['x'].apply(lambda x: pd.Series(list(x))).astype(dtype=np.float32)
	# extract outputs as floats
	train_y_df = train_df['y'].astype(dtype=np.float32)

	# convert single input column in multiple columns (one column for each 'bit' of status)
	x_test = test_df['x'].apply(lambda x: pd.Series(list(x))).astype(dtype=np.float32)
	# extract outputs as floats
	y_test = test_df['y'].astype(dtype=np.float32)

	# print dataframes structure
	print("Train DS and Test DS shapes:")
	print(train_x_df.shape)
	print(train_y_df.shape)
	print(x_test.shape)
	print(y_test.shape)

	# validation flag off -> return only train & test sets
	if validation == False:
		return train_x_df, train_y_df, x_test, y_test

	# split dataset into training and validation set
	x_train, x_val, y_train, y_val = train_test_split(train_x_df, train_y_df, test_size=1/3)

	print("Train, val, test set shapes:")
	print(x_train.shape)
	print(x_val.shape)
	print(x_test.shape)

	return x_train, y_train, x_val, y_val, x_test, y_test


'''
Loads discriminator data from file and generates a proper training set to use in a GAN network
'''
def load_GAN_data(dataset):
	#load datasets from files
	train_df = pd.read_csv(dataset, delimiter="-", names = ["x", "y"])

	#shuffle datasets
	#train_df = shuffle(train_df)
	
	#convert single input column in multiple columns (one column for each 'bit' of status)
	x_train = train_df['x'].apply(lambda x: pd.Series(list(x))).astype(dtype=np.float32)

	return x_train


'''
Generates a training set for G, composed by noise mixed with sampled data from an existing dataset
A portion of the existing dataset is not used, and it is returned as test set for G
The full test set for G is returned as third element
'''
def create_noise_ds_generator(ds_feasible_gen, num_single_solutions):

	print("Creating noise datasets for Generator network...")

	feasible_ds = load_GAN_data(dataset=ds_feasible_gen)

	feasible_ds_len = feasible_ds.shape[0]
	num_feasible_samples = int(feasible_ds_len / 2)

	# Split in two parts the DS with feasible solutions, then the first part it is used as training set
	feasible_ds = shuffle(feasible_ds)
	trainset_gen = feasible_ds[:num_feasible_samples]
	testset_gen = feasible_ds[-num_feasible_samples:]

	# create collections for feasible solutions based on the number of queens
	feasible_solutions = {}

	for idx, feas_sol in feasible_ds.iterrows():
		num_q = int(np.sum(feas_sol))
		feasible_solutions.setdefault(num_q, []).append(list(feas_sol))

	# Generate noise
	#noise = np.random.binomial(1, 0.065, (1000, 64)).astype(dtype=np.float32)
	noise_list = []

	for num_q in range(1,8):

		# define number of solutions to produce for "num_q" queens
		num_sol = num_single_solutions

		# generate a smaller amount of solutions with 1 or 2 queens, to have lower total feasibility
		if num_q == 1:
			num_sol = int(num_single_solutions / 3)
		elif num_q == 2:
			num_sol = int(num_single_solutions / 2)

		for num in range(num_sol):
			sol = shuffle(np.array([1] * num_q + [0] * (64-num_q)))
			noise_list.append(sol)

	
	# Mix feasible data with noise, so that feasibility is almost equal for the different solution types (num. queens)
	# Increase feasibility of solutions with more than 2 queens
	for num_q in range(3,8):
		count = 0

		# define number of solutions to produce for "num_q" queens
		num_sol = num_single_solutions

		
		# sample less elements from solutions with 6 or 7 queens: there are less of these feasible solutions, so we don't want to use them all or exceed the total number of solutions available
		if num_q == 6:
			num_sol = int(num_single_solutions / 2)
		elif num_q == 7:
			num_sol = int(num_single_solutions / 2)
		

		# solutions with more queens, thus low feasibility -> add feasible solutions
		while count < num_sol:
			noise_list.append(feasible_solutions[num_q][count])
			count += 1
	
	# create noise dataframe
	noise = pd.DataFrame(noise_list, dtype='float32')

	noise = shuffle(noise)
	print("Noise set shape:")
	print(noise.shape)

	return noise, testset_gen, feasible_ds


'''
The goal is to evaluate the ability of G to create solutions starting from empty ones

Arguments:
		generator (model): the generator model to evaluate
		full_sol_df (dataframe): dataframe containing the full 92 solutions, derived from the 12 base sol. expanded
		full_sol_trainset_df (dataframe): dataframe containing the full solutions for the train set, derived from the 8 base sol. of the training set expanded
		num_solutions (int): number of full solutions that G needs to generate
'''
def stochastic_generation(generator, full_sol_file, full_sol_trainset_file, num_solutions=1000, num_queens_to_generate=8):

	#load the 92 full solutions, for the N-Queens problem, into a dataframe
	full_solutions_df = pd.read_fwf(full_sol_file, widths=[1] * 64, header=None)

	#load the train set full solutions, expanded from the 8 base sol., into a dataframe
	full_solutions_trainset_df = pd.read_fwf(full_sol_trainset_file, widths=[1] * 64, header=None)

	# generate empty solutions (zeros in all positions)
	empty_solutions = np.zeros(shape=(num_solutions, full_solutions_df.shape[1]), dtype=np.float32)

	# collection of full solutions
	full_solutions_generated = []

	count = 0

	# iterate on each empty solution
	for empty_sol in empty_solutions:

		#progress bar
		perc = count / float(num_solutions)
		sys.stdout.write("\rSolutions generation - progress: [{0:50s}] {1:.1f}%".format('#' * int(perc * 50), perc * 100))

		# necessary to provide a single solution to G, for its prediction
		sol = np.array([empty_sol,])

		# counter for number of queens in current solution
		num_queens_in_sol = 0

		while num_queens_in_sol < num_queens_to_generate:

			# get assignments probabilities from G
			assignment_prob = generator.predict(sol)

			# choose assignment randomly, based on G's probabilities in output
			selected_assignment = np.random.choice(64, p=assignment_prob[0])

			# check if solutions doesn't already have a queen in the assignment position:
			# this is necessary even if G has a masking layer, because the outcome of G comes from the Softmax,
			# so even if the Softmax receives values set to 0 from the masking layer, it can assign to it a small probability
			if sol[0][selected_assignment] == 0:
				# apply assignment on original solution
				sol[0][selected_assignment] = 1
				# update queens counter
				num_queens_in_sol += 1

		# solution is complete (8 queens): save it
		full_solutions_generated.append(sol[0])

		# for progress bar
		count += 1

	print()
	print("Starting criterion evaluation...")
	print("----------------------------")
	print("1st criterion: n-Queens constraints (row, diagonal, column)")
	print("2nd criterion: n-Queens constraints + bias (row, diagonal, column, no queen in central 2x2 square)")
	print("3rd criterion: bias (no queen in central 2x2 square)")
	print("----------------------------")

	first_criterion_valid_count = 0
	second_criterion_valid_count = 0
	third_criterion_valid_count = 0
	# reset counter prog. bar
	count = 0

	# solutions Series list
	sol_collection = []
	found = False

	# eval 1st criterion (row, column, diagonal)
	for full_sol in full_solutions_generated:

		#progress bar
		perc = count / float(num_solutions)
		sys.stdout.write("\r1st criterion evaluation - progress: [{0:50s}] {1:.1f}%".format('#' * int(perc * 50), perc * 100))

		full_sol_series = pd.Series(full_sol)

		for sol_in_col in sol_collection:
			if full_sol_series.equals(sol_in_col):
				found = True
				break

		if not found:
			sol_collection.append(full_sol_series)

		found = False

		# check if solution is valid for the 1st criterion
		valid = check_single_solution_feasibility(full_sol_series, full_solutions_df)

		if valid:
			first_criterion_valid_count += 1

		count += 1

	# reset counter prog. bar
	count = 0
	print()
	print("Num. unique assignments produced: %d" % len(sol_collection))

	# eval 2nd criterion (row, column, diagonal + no queen in 2x2 central box)
	for full_sol in full_solutions_generated:

		#progress bar
		perc = count / float(num_solutions)
		sys.stdout.write("\r2nd criterion evaluation - progress: [{0:50s}] {1:.1f}%".format('#' * int(perc * 50), perc * 100))

		full_sol_series = pd.Series(full_sol)

		# check if solution is valid for the 2nd criterion
		valid = check_single_solution_feasibility(full_sol_series, full_solutions_trainset_df)

		if valid:
			second_criterion_valid_count += 1

		count += 1

	# reset counter prog. bar
	count = 0
	print()

	# eval 3rd criterion (no queen in 2x2 central box)
	for full_sol in full_solutions_generated:

		#progress bar
		perc = count / float(num_solutions)
		sys.stdout.write("\r3rd criterion evaluation - progress: [{0:50s}] {1:.1f}%".format('#' * int(perc * 50), perc * 100))

		full_sol_arr = np.array(full_sol)

		# check if solution doesn't have any queen in 2x2 central positions
		if full_sol_arr[27] == 0 and full_sol_arr[28] == 0 and full_sol_arr[35] == 0 and full_sol_arr[36] == 0:
			third_criterion_valid_count += 1

		count += 1

	print()
	print("Generated solutions validity - 1st criterion: %.2f%%" % (first_criterion_valid_count / num_solutions * 100))
	print("Generated solutions validity - 2nd criterion: %.2f%%" % (second_criterion_valid_count / num_solutions * 100))
	print("Generated solutions validity - 3rd criterion: %.2f%%" % (third_criterion_valid_count / num_solutions * 100))


'''
Prints values of a tensor
	Arguments:
			tensor (tensor): a tensor

To print a tensor in generator/discriminator use:
	my_utils.print_tensor(K.eval(tensor), tf.Session())
'''
def print_tensor(tensor, sess):
	with sess.as_default():
		print_op = tf.print(tensor, summarize=64)
		with tf.control_dependencies([print_op]):
			out = tf.add(tensor, tensor)
		sess.run(out)


'''
Prints output tensors of a layer
	Usage: x = my_utils.print_layer(x, "x=")
'''
def print_layer(layer, message, first_n=2, summarize=64):
  return keras.layers.Lambda((
    lambda x: tf.Print(x, [x],
                      message=message,
                      first_n=first_n,
                      summarize=summarize)))(layer)


'''
Cheks if a partial solutions for the N-Queens completion problem is feasible or not
by multiplying together all the 92 full solutions and the partial ones.

	Arguments:
		full_solutions_df: dataframe containing all the 92 full solutions
		partial_sol: series containing a single partial solution

	Returns:
		feasible: True/False, based on the feasibility of the partial solution

'''
def check_single_solution_feasibility(partial_sol, full_solutions_df):
	feasible = False

	#count the total number of queens in the partial solution
	queens_num_altered_sol = partial_sol[(partial_sol >= 0.9)].count()

	#for every full solution, compare it with the "altered" partial solution by multiplying them together
	#if the total number of queens (1s) for the partial solution changes, then the "altered" partial solution is unfeasible
	for idx, full_sol in full_solutions_df.iterrows():
		result_solution = partial_sol.multiply(full_sol)
		queens_num_new_sol = result_solution[(result_solution >= 0.9)].count()

		#check if number of queens is the same of the one in the altered partial solution
		#if a match is found, the altered solution is feasible
		if queens_num_new_sol == queens_num_altered_sol:
			feasible = True
			break

	return feasible


'''
Performs predictions on a given dataframe
Then evaluates each prediction to check if it's feasible or unfeasible

	Returns: feasibility rate for the given dataframe
'''
def check_solutions_feasibility(model, target_df, full_solutions_dataset):
	#counter for feasible and unfeasible solutions
	tot_unfeasible_count = 0
	tot_feasible_count = 0
	queens_count = Counter()
	#dict for feasibility ratios of different number of filled cells
	feas_ratios= {}
	#length of dataframe for progress bar
	size_df= len(target_df.index)
	#counter for progress bar and prediction selection
	count = 0
	#counter for illegal queens (useful if G does not have the masking layer)
	illegal_queens_count = Counter()
	#load the 92 full solutions, for the N-Queens problem, into a dataframe
	full_solutions_df = pd.read_fwf(full_solutions_dataset, widths=[1] * 64, header=None)

	df_copy = target_df.copy(deep=True)

	#predict assignments for all partial solutions provided
	predictions = predict_proba_v2(model, df_copy)
	
	#TRAINING data
	for index, solution in df_copy.iterrows():
		#progress bar
		perc = count / float(size_df)
		sys.stdout.write("\rFeasibility check - progress: [{0:50s}] {1:.1f}%".format('#' * int(perc * 50), perc * 100))

		#get prediction corresponding to current solution
		prediction = predictions[count]

		#increase counter
		count +=1
		
		#count the total number of queens in the partial solution
		queens_number = solution[(solution >= 0.9)].count()
		
		#get argument with maximum value in prediction -> represents position of next assignment
		selected_assignment = np.argmax(prediction)

		if solution[selected_assignment] <= 0:
			#apply assignment to a copy of the partial solution
			solution[selected_assignment] = 1
		else:
			illegal_queens_count[str(queens_number)] += 1
			tot_unfeasible_count += 1
			queens_count["UF" + str(queens_number)] += 1  
			continue

		#check if partial solution is feasible or not
		is_feasible = check_single_solution_feasibility(solution, full_solutions_df)
		if is_feasible:
			tot_feasible_count += 1
			queens_count["F" + str(queens_number)] += 1 
		else:
			tot_unfeasible_count += 1
			queens_count["UF" + str(queens_number)] += 1  


	for key, value in illegal_queens_count.items():
		print("*** Queens assigned to a busy position - %s queens:  %d ***" % (key, value))
	

	#compute feasibility ratio
	tot_solutions = tot_feasible_count + tot_unfeasible_count
	feasibility_rate_global = (tot_feasible_count / tot_solutions)
	print()
	print("Feasibility ratio: %f" % feasibility_rate_global)

	#compute feasibility ratio for different number of filled cells
	for num_q in range(0,8):
		feas_n = queens_count["F" + str(num_q)]
		unfeas_n = queens_count["UF" + str(num_q)]
		tot = feas_n + unfeas_n
		print("Total n. of %d queens: %d" % (num_q, tot))
		if tot != 0:
			feas_ratio = (feas_n / tot)
		else:
			#not found any solution with num_q number of queens -> assume 1.0 ratio?
			feas_ratio = 0

		#add ratio to dict
		feas_ratios[num_q] = feas_ratio
		print("Feasibility ratio for %d queens: %f" % (num_q, feas_ratio))

	print("------------------")

	return feasibility_rate_global, feas_ratios


'''
Evaluates Generator and Full Generator (G+Lambda), to show how much the G can improve with the current Lambda layer
'''
def evaluate_generator_training_ability(generator, generator_full):
	'''
	testset_gener = load_GAN_data(dataset="DS_GENERATOR/DS.A.NEW.UNIQUES.B.4.txt", smoothing=False)
	x_trains = load_GAN_data(dataset = "DS_GENERATOR/DS.A.NEW.UNIQUES.B.4.txt", smoothing=False)
	y_trains = load_GAN_data(dataset = "DS_GAN/DS.FEASIBLE.UNIQUES.B.txt", smoothing=False)

	# split dataset into training and validation set
	x_train, x_val, y_train, y_val = train_test_split(x_trains, y_trains, test_size=1/3)
	'''
	x_train = load_GAN_data(dataset = "DS_GENERATOR/DS.A.NEW.UNIQUES.B.4.txt", smoothing=True)
	y_train = np.ones(shape=(x_train.shape[0], 1))

	#evaluate G
	# check feasibility of assignments produced by the generator
	feasibility_rate, feas_ratios = check_solutions_feasibility(generator, pd.DataFrame(x_train[:1000]), "DS.FULL.SOLUTIONS.txt")

	#evaluate FULL G
	### TODO: create proper method
	#feasibility_rate, feas_ratios = check_solutions_feasibility_fullg(generator_full, pd.DataFrame(testset_gener[:1000]), "DS.FULL.SOLUTIONS.txt")

	loss, acc = generator_full.evaluate(x_train, y_train)
	print("G loss: %f, G acc: %f" % (loss, acc))

	monitor_func = TensorBoard(log_dir='./Risultati/tensorboard_data', 
		histogram_freq=1, batch_size=32, write_graph=False, write_grads=True, write_images=False, update_freq='epoch')

	generator_full.fit(x_train, y_train, validation_data=(x_train[:200], y_train[:200]), epochs=5, callbacks=[monitor_func])

	loss, acc = generator_full.evaluate(x_train, y_train)
	print("G loss: %f, G acc: %f" % (loss, acc))

	#evaluate G
	# check feasibility of assignments produced by the generator
	feasibility_rate, feas_ratios = check_solutions_feasibility(generator, pd.DataFrame(x_train[:1000]), "DS.FULL.SOLUTIONS.txt")

	#evaluate FULL G
	### TODO: create proper method
	#feasibility_rate, feas_ratios = check_solutions_feasibility_fullg(generator_full, pd.DataFrame(testset_gener[:1000]), "DS.FULL.SOLUTIONS.txt")


'''
Plots feasibility ratios for training, validation, and test sets in a single plot
'''
def plot_feasibility_ratios(feas_ratios_train, feas_ratios_val, feas_ratios_test):

	plt.xlabel("Num. filled cells")
	plt.ylabel("Feas. ratio")
	
	#Training
	lists_tr = sorted(feas_ratios_train.items()) # sorted by key, return a list of tuples
	x_tr, y_tr = zip(*lists_tr) # unpack a list of pairs into two tuples
	plt.plot(x_tr, y_tr, label="training", c="blue")

	#Validation
	lists_val = sorted(feas_ratios_val.items()) # sorted by key, return a list of tuples
	x_val, y_val = zip(*lists_val) # unpack a list of pairs into two tuples
	plt.plot(x_val, y_val, label="validation", c="gold")

	#Test
	lists_te = sorted(feas_ratios_test.items()) # sorted by key, return a list of tuples
	x_te, y_te = zip(*lists_te) # unpack a list of pairs into two tuples
	plt.plot(x_te, y_te, label="test", c="red")

	plt.legend()
	plt.grid()
	plt.ylim(ymin=0)
	plt.show()


'''
Creates logger for GAN network, that prints both on console and on file
File has timestamp as its name
'''
def create_logger_GAN():

	current_datetime = datetime.now().strftime('%Y%m%d-%H.%M')
	logging.basicConfig(level=logging.DEBUG,
						format='%(message)s',
						datefmt='%m-%d %H:%M',
						filename="Risultati/Logfile GAN/GAN_test-" + current_datetime + ".txt",
						filemode='w')

	console = logging.StreamHandler()
	console.setLevel(logging.INFO)

	# add the handler to the root logger
	logging.getLogger('').addHandler(console)

	return logging
	

"""
	Enhanced prediction method for SVM, takes in consideration also missing class in Y train set
"""
def predict_proba_v2(svm, dataset):

	# explicitly set the possible class labels, otherwise SVM will not output prediction for missing classes
	all_classes = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
		21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
		41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
		61,62,63]) 

	probs = svm.predict_proba(dataset)
	proba_ordered = np.zeros((probs.shape[0], all_classes.size),  dtype=np.float)
	sorter = np.argsort(all_classes) # http://stackoverflow.com/a/32191125/395857
	idx = sorter[np.searchsorted(all_classes, svm.classes_, sorter=sorter)]
	proba_ordered[:, idx] = probs

	return proba_ordered

                      
'''
 Test masking Lambda layer
'''
def test_masking():
	#creation of simulated INPUT values
	val_in = np.array([[0, 1, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0,
			  0, 0, 0, 0, 0, 0, 0, 0,
			   1, 0, 0, 0, 0, 0, 0, 0],
			   [0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0,
			  0, 0, 0, 0, 0, 0, 0, 0,
			   0, 0, 0, 0, 0, 0, 0, 1]])
	#conversion to tensor
	inputs = tf.convert_to_tensor(val_in, dtype=tf.float32)

	#creation of simulated PARTIAL OUTPUT values
	val_out = np.array([list(range(64)), list(range(64))])
	#conversion to tensor
	par_outputs = tf.convert_to_tensor(val_out, dtype=tf.float32)

	#test masking of generator
	masked_out = generator.masking([inputs, par_outputs])
	masked_out = print_layer(masked_out, "masked_out=")

'''
 Test merging Lambda layer
'''
def test_merging():
	#creation of simulated INPUT values
	val_in = np.array([[0, 1, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 1, 0, 0,
			  0, 0, 0, 0, 0, 0, 0, 0,
			   1, 0, 0, 0, 0, 0, 0, 0],
			   [0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0, 0, 0, 0, 1, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0,
			  0, 0, 0, 0, 0, 0, 0, 0,
			   0, 0, 0, 0, 0, 0, 0, 1]])
	#conversion to tensor
	inputs = tf.convert_to_tensor(val_in, dtype=tf.float32)

	#creation of simulated PARTIAL OUTPUT values
	assignments = np.array([[0.45, 0.3, 0.0005, 0, 0, 0.6, 0, 0,
		 0, 0, 0, 0, 0, 0, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0,
		   0, 0, 0, 0.5, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			 0, 0, 0, 0, 0, 0, 0, 0,
			  0, 0, 2.7, 0, 0, 0, 0, 0,
			   0, 1.4, 0, 0, 2.2, 0, 0, 0],
			   [0, 0, 0, 0, 0, 0, 0, 0,
		 0, 0.55, 0, 0, 0, 0, 0.3, 0,
		  0, 0, 3.1, 0, 0, 0, 0, 0,
		   0, 0, 0, 0, 0, 0, 0.1, 0,
			0, 0, 0, 2.5, 0, 0, 0, 0,
			 3.7, 0, 0, 0, 0, 0, 0, 0,
			  0, 0, 0, 0, 0, 0, 0, 0,
			   0, 0, 0, 0.00064, 0, 0, 0, 0]])
	#conversion to tensor
	assignments_tensor = tf.convert_to_tensor(assignments, dtype=tf.float32)

	#test masking of generator
	merged = merging_test([inputs, assignments_tensor])
	print_tensor(K.eval(merged), tf.Session())