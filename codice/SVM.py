# Load libraries
import pandas
import json
import os
import sys
import numpy as np
from pandas.plotting import scatter_matrix
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC

import my_utils

'''
Print feature importance recap for SVC technique
'''
def f_importances(coef, names):
    imp = coef
    print(names)
    imp,names = zip(*sorted(zip(imp[0],names)))
    print(names)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def train(data_file):
	
	# Make preditcions
	predictions = svm.predict(X_validation)
	print(classification_report(Y_validation, predictions))
	


if __name__ == "__main__":

	# ---- GENERATOR ----
	
	# load data for generator network
	x_train_G, y_train_G, x_val_G, y_val_G, x_test_G, y_test_G = my_utils.load_generator_data(training_data_file = "DS_GENERATOR/DS.MULTIPLE.A.txt", 
		test_data_file="DS_GENERATOR/DS.MULTIPLE.B.txt")

	# transform one-hot encoded labels into single integers
	y_train_G_singleVal = y_train_G.idxmax(axis=1)
	y_test_G_singleVal = y_test_G.idxmax(axis=1)

	# create generator SVM
	svm_gen = SVC(gamma='scale', decision_function_shape='ovo', probability=True)

	# train generator SVM
	print("Training the generator SVM...")
	svm_gen.fit(x_train_G, y_train_G_singleVal)
	
	# Evaluate G's accuracy - Train set
	predictions = svm_gen.predict(x_train_G)
	print("SVM generator accuracy: %f" % accuracy_score(y_train_G_singleVal, predictions))

	# Evaluate G's accuracy - Test set
	predictions = svm_gen.predict(x_test_G)
	print("SVM generator accuracy: %f" % accuracy_score(y_test_G_singleVal, predictions))

	# Evaluate G's feasibility ratio on training and test sets
	### Utilizzare "predict_proba_v2" al posto di "predict" nel metodo, per farlo lavorare con SVM ###
	my_utils.check_solutions_feasibility(svm_gen, x_train_G, "DS.FULL.SOLUTIONS.txt")
	my_utils.check_solutions_feasibility(svm_gen, x_test_G, "DS.FULL.SOLUTIONS.txt")
	
	### Utilizzare "predict_proba_v2" al posto di "predict" nel metodo, per farlo lavorare con SVM ###
	my_utils.stochastic_generation(svm_gen, full_sol_file="DS.FULL.SOLUTIONS.txt", full_sol_trainset_file="DS_GENERATOR/DS.SOL.TRAINSET.txt", num_queens_to_generate=8)
	
	
	# ---- DISCRIMINATOR ----

	#load data for discriminator network
	x_train_D, y_train_D, x_test_D, y_test_D = my_utils.load_discriminator_data_v2(train_dataset="DS_DISCRIMINATOR/DS.UNIQUES.B.ALL.txt", 
		test_dataset="DS_DISCRIMINATOR/DS.UNIQUES.A.ALL.txt",
		validation=False)

	# create generator SVM
	svm_disc = SVC(gamma='scale', decision_function_shape='ovo', probability=True)
	
	# train discriminator SVM
	print("Training the discriminator SVM...")
	print(x_train_D.shape)
	print(y_train_D.shape)
	svm_disc.fit(x_train_D, y_train_D)

	# Evaluate D's accuracy - Train set
	predictions = svm_disc.predict(x_train_D)
	print("SVM discriminator accuracy: %f" % accuracy_score(y_train_D, predictions))

	# Evaluate D's accuracy - Test set
	predictions = svm_disc.predict(x_test_D)
	print("SVM discriminator accuracy: %f" % accuracy_score(y_test_D, predictions))