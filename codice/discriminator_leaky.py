'''
Francesco Giovanelli - March 2019

Build a Pre-activated ResNet able to discriminate between feasible and unfeasible solutions for the N-Queens Completion problem
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
from keras.models import load_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import my_utils


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
Builds a resNet block

	Arguments:
		inputs (tensor): input tensor from input or previous layer
		activation (string): activation name

	Returns:
		nn (tensor): tensor as input to the next layer
'''
def residual_block(inputs):

	'''
	ResNetv2 sequence: BN - Activation - Dense layer - BN - Activation - Dense layer
	'''
	#First half
	#nn = BatchNormalization()(inputs)
	nn = Dense(units=500)(inputs)
	nn = LeakyReLU(alpha=0.2)(nn)
	nn = Dropout(rate=0.1)(nn)
	#Second half
	#nn = BatchNormalization()(nn)
	nn = Dense(units=200)(nn)
	nn = LeakyReLU(alpha=0.2)(nn)
	nn = Dropout(rate=0.1)(nn)
	
	return nn


'''
Builds the Discriminator Network, based on the ResNetv2
	Arguments:
		input_shape (tensor): shape of input image tensor
		res_block_num (int): number of residual blocks
		num_classes (int): number of classes
	Returns: 
		the Keras model of the network
'''
def build_discriminator(input_shape, res_block_num=100, attention_on=False):
	inputs = Input(shape=input_shape)

	
	#Attention
	if attention_on:
		#First BatchNormalization
		nn = BatchNormalization()(inputs)
		#compute input weights for the input through attention network
		nn = deep_attention(nn, num_classes=64, num_neurons=300, activation='relu')
		
		# First layer
		nn = BatchNormalization()(nn)
		nn = Dropout(rate=0.1)(nn)
		nn = LeakyReLU(alpha=0.2)(nn)
		nn = Dense(units=200)(nn)

	else:		
		# First layer
		#nn = BatchNormalization()(inputs)
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


	#Final layer with sigmoid, for binary classification
	outputs = Dense(units=1, activation='sigmoid')(nn)

	# Instantiate model
	model = Model(inputs=inputs, outputs=outputs)

	#optimization function
	adam = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999) #, decay=1e-3)

	#compile discriminator model
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

	#print model structure to file
	with open('discrim_struct.txt', 'w') as model_file:
		# Pass the file handle in as a lambda function to make it callable
		model.summary(print_fn=lambda x: model_file.write(x + '\n'))

	return model


'''
Performs training for the Discriminator Network
'''
def train_discriminator(model, x_train, y_train, x_val, y_val, epochs):
	
	#early stop of training when validation loss does not improve for X epochs
	early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')

	#learning rate decay through epochs
	lr_decay = LearningRateScheduler(step_lr_decay)

	#train discriminator newtork
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
	print("--------------")
	print("Test loss:", test_loss)
	print("Test accuracy:", test_acc)
	print("--------------")


'''
Main
'''
if __name__ == "__main__":

	# load data for discriminator
	x_train, y_train, x_val, y_val, x_test, y_test = my_utils.load_discriminator_data_v2(train_dataset="DS_DISCRIMINATOR/DS.UNIQUES.B.ALL.txt", 
		test_dataset="DS_DISCRIMINATOR/DS.UNIQUES.A.ALL.txt")

	#create discriminator model
	model = build_discriminator(input_shape=x_train.shape[1:], res_block_num=2, attention_on=False)
	
	# train discriminator network
	trained_model = train_discriminator(model, x_train, y_train, x_val, y_val, epochs=200)
	
	# test discriminator model
	test_model(trained_model, x_test, y_test)
	
	#save discriminator model to file
	trained_model.save('discriminator_model_leaky.h5')