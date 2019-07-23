'''
Francesco Giovanelli - March 2019

Build a Pre-activated ResNet able to generate solutions for the N-queens completion problem
'''

import pandas as pd
import math
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model   
from keras.layers import * 
from keras.utils import to_categorical
from keras import optimizers
from keras import losses
from keras.models import load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

import my_utils


'''
Function that computes a tensor mask for the N-queens problem. This mask helps the generator to generate assignments only on available "rows".

	Arguments:
		tensors (tensors array): collection of tensors

	Returns:
		masked_output (float array): masked partial output (to use in the softmax classifier)
'''
def masking(tensors):
	input_tensor = tensors[0]
	partial_output_tensor = tensors[1]

	#split input tensor into N sub-tensors on y axis (reducing shape of tensor, eg. from (32,64) to (32,8))
	split_tensors = list(tf.split(input_tensor, num_or_size_splits=8, axis=1))
	
	#create tensors with all 1s and all 0s (same dimension as split_tensors)
	tensor_all_ones = K.ones_like(split_tensors[0])
	tensor_all_zeros = K.zeros_like(split_tensors[0])
	
	#list of masked split_tensors
	mask_tensors = []

	for split_tensor in split_tensors:
		#sum each element of each "split" on the y dimension (reducing size of tensor, eg. from (32,8) to (32,1))
		tensor_sum = K.sum(split_tensor, axis=1)
		#create partial mask tensor by checking the sum of each split tensor's row: if sum is zero, the mask's "row" is a tensor of all ones, otherwise it's a tensor of all zeros
		row_tensor_masked = K.switch(K.equal(tensor_sum, 0), tensor_all_ones, tensor_all_zeros)
		#add new partial mask tensor to mask tensor list
		mask_tensors.append(row_tensor_masked)
	
	#build full mask tensor by merging "split tensors" together on y axis
	mask = K.concatenate(mask_tensors, axis=1)
	
	#multiply partial output and mask tensors to compute masked output tensor
	masked_output = keras.layers.multiply([partial_output_tensor, mask])

	return masked_output


'''
Builds a small Neural Network that applies an Attention mechanism to the generators'input

	Arguments:
		inputs (tensor): input tensor from input data
		num_layers (integer): number of classes for the problem (dimension of output)
		num_neurons (integer): number of neurons per layer of the attention network
		activation (string): activation function name

	Returns:
		input_weights (tensor): tensor composed by input weights
'''
def deep_attention(inputs, num_classes=64, num_neurons=300, activation='relu'):

	'''
	Attention network structure: Dense layer - Dense layer - ... - Softmax
	'''
	nn = Dense(units=num_neurons, activation=activation)(inputs)
	nn = Dense(units=num_neurons, activation=activation)(nn)
	nn = Dense(units=num_classes, activation=activation)(nn)
	#compute attention weights using softmax
	####	attention_weights = Dense(units=num_classes, activation='softmax', kernel_initializer='he_normal')(nn)
	#compute input weights using inputs and attention weights
	input_weights = keras.layers.Add()([inputs, nn])

	return input_weights


'''
Function that progressively reduces the learning rate

	Arguments:
		epoch (integer >= 0): current epoch
		lr (float): current learning rate

	Returns:
		lr_new (float): new learning rate
'''
def step_lr_decay(epoch, lr):
	k = 1e-3
	lr_new = lr / (1 + (epoch * k))
	#print("Old LR: %f" % lr)
	#print("New LR: %f" % lr_new)
	return lr_new


'''
Builds a resNet block

	Arguments:
		inputs (tensor): input tensor from input or previous layer
		activation (string): activation name

	Returns:
		nn (tensor): tensor as input to the next layer
'''
def residual_block(inputs):

	#First half
	nn = Dense(units=500)(inputs)
	nn = LeakyReLU(alpha=0.2)(nn)
	nn = Dropout(rate=0.1)(nn)
	#Second half
	nn = Dense(units=200)(nn)
	nn = LeakyReLU(alpha=0.2)(nn)
	nn = Dropout(rate=0.1)(nn)

	return nn


'''
Builds the Generator Network, based on the ResNetv2
	Arguments:
		input_shape (tensor): shape of input image tensor
		res_block_num (int): number of residual blocks
		num_classes (int): number of classes
	Returns: 
		the Keras model of the network
'''
def build_generator(input_shape, res_block_num=100, num_classes=64, attention_on=False, masking_on=True):
	inputs = Input(shape=input_shape)

	#Attention
	if attention_on:
		#compute input weights for the input through attention network
		nn = deep_attention(inputs, num_classes=64, num_neurons=200, activation='relu')
		# First layer
		nn = Dense(units=200)(nn)
		nn = LeakyReLU(alpha=0.1)(nn)
		nn = Dense(units=200)(nn)

	else:		
		# First layer
		nn = Dropout(rate=0.1)(inputs)
		nn = Dense(units=200)(nn)
		nn = LeakyReLU(alpha=0.2)(nn)
		nn = Dropout(rate=0.1)(nn)

	#build sequence of residual blocks
	for index in range(res_block_num):
		#build a single residual block
		res_block_y = residual_block(inputs=nn)
		#add residual block output (f(x, W)) and input's identity mapping (h(x)) to compute the new output y'
		nn = keras.layers.Add()([nn, res_block_y])
	
	#add a classifier on top of the network (BN + activation + softmax)
	#Dense layer with num. neurons equal to num. classes
	nn = Dense(units=num_classes)(nn)

	#Masking before Softmax output
	if masking_on:
		#masking custom layer
		nn = Lambda(masking)([inputs, nn])

	# Softmax output function
	outputs = Activation('softmax')(nn)

	# Instantiate model
	model = Model(inputs=inputs, outputs=outputs)

	#optimization function
	adam = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999) #, decay=1e-3)

	#compile generator model
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
	#print model structure to file
	with open('gen_struct.txt', 'w') as model_file:
		# Pass the file handle in as a lambda function to make it callable
		model.summary(print_fn=lambda x: model_file.write(x + '\n'))

	return model



'''
Performs training for the Generator Network
'''
def train_generator(model, x_train, y_train, x_val, y_val, epochs):

	#early stop of training when validation loss does not improve for X epochs
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

	#learning rate decay through epochs
	lr_decay = LearningRateScheduler(step_lr_decay)

	#save model every epoch
	#model_checkpoint = ModelCheckpoint("model_checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', mode='auto', period=1)

	#train generator newtork
	model.fit(x_train, y_train,
		epochs=epochs,
		validation_data=(x_val, y_val),
		shuffle=True,
		callbacks=[early_stop, lr_decay])

	return model


'''
Evaluates model's performance on Test Set
'''
def test_model(model, x_test_set, y_test_set):

	test_loss, test_acc = model.evaluate(x_test_set, y_test_set)
	print("------------------")
	print("Test loss:", test_loss)
	print("Test accuracy:", test_acc)
	print("------------------")

	

'''
Main
'''
if __name__ == "__main__":
	#load data for generator network
	x_train, y_train, x_val, y_val, x_test, y_test = my_utils.load_generator_data(training_data_file = "DS_GENERATOR/DS.UNIQUES.A.txt", 
		test_data_file="DS_GENERATOR/DS.UNIQUES.B.txt",
		smoothing=False)

	#create generator model
	model = build_generator(input_shape=x_train.shape[1:], res_block_num=2, num_classes=64, attention_on=False, masking_on=True)

	#train generator network
	trained_model = train_generator(model, x_train, y_train, x_val, y_val, epochs=200)

	#evaluate model performance on test set
	test_model(trained_model, x_test, y_test)

	#save generator model to file
	trained_model.save('generator_model_AB_leaky.h5')

	#check solutions feasibility for training, validation and test sets
	feas_ratios_train = my_utils.check_solutions_feasibility(trained_model, x_train[-1000:], "DS.FULL.SOLUTIONS.txt")
	feas_ratios_val = my_utils.check_solutions_feasibility(trained_model, x_val[-1000:], "DS.FULL.SOLUTIONS.txt")
	feas_ratios_test = my_utils.check_solutions_feasibility(trained_model, x_test, "DS.FULL.SOLUTIONS.txt")
	#plot feas. ratios
	#my_utils.plot_feasibility_ratios(feas_ratios_train, feas_ratios_val, feas_ratios_test)
	