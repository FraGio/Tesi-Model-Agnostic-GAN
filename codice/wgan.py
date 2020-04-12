'''
Francesco Giovanelli - April 2019

Build a Generative Adversarial Network model, using existing critic and Generator networks.
G and D were created to work with the N-Queens Completion problem

source: https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
'''

from __future__ import print_function, division

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
from keras.callbacks import TensorBoard
from keras.utils.generic_utils import Progbar
from sklearn.utils import shuffle
import logging
from collections import Counter
import sys
from functools import partial
from keras.layers.merge import _Merge

import my_utils
import critic
import generator_leaky
import generator


class RandomWeightedAverage(_Merge):
	"""Takes a randomly-weighted average of two tensors. In geometric terms, this
	outputs a random point on the line between each pair of input points.
	Inheriting from _Merge is a little messy but it was the quickest solution I could
	think of. Improvements appreciated."""

	batch_size=64

	def _merge_function(self, inputs):
		weights = K.random_uniform((64, 1, 1, 1))
		return (weights * inputs[0]) + ((1 - weights) * inputs[1])


'''
Builds a WGAN network, by loading existing critic and Generator networks
'''
class WGAN():

	def __init__(self):

		self.latent_dim = 64

		# Create logger to write both on console and on file
		self.logger = my_utils.create_logger_GAN()
		
		# Following parameter and optimizer set as recommended in paper
		self.training_ratio = 5
		self.gradient_penalty_weight = 10 # As per the paper
		optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)#optimizers.RMSprop(lr=0.00005)

		# Build and compile the critic
		self.critic = self.build_critic(pre_trained=False, res_blocks_num=4)

		# Build the generator
		self.generator = self.build_generator(pre_trained=False, res_blocks_num=2, masking=True)
		
		# The generator takes partial solutions as input and generates proper assignments
		partial_solutions = Input(shape=(self.latent_dim,))
		assignments = self.generator(partial_solutions)

		# Merge partial solutions and generated assignments, to compute new generated solutions
		merging_out = keras.layers.Add()([partial_solutions, assignments])

		# Create the full Generator model (original G + Add operation)
		self.generator_full = Model(partial_solutions, merging_out)
		gen_solutions = self.generator_full(partial_solutions)

		# For the combined model we will only train the generator
		self.critic.trainable = False

		# The critic takes generated images as input and determines validity
		valid = self.critic(gen_solutions)

		# The combined model  (stacked generator and critic)
		self.combined = Model(partial_solutions, valid)
		self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

		# For the combined model we will only train the generator
		self.critic.trainable = True
		self.generator_full.trainable = False

		# Critic
		real_samples = Input(shape=(self.latent_dim,))
		generator_input_for_critic = Input(shape=(self.latent_dim,))
		generated_samples_for_critic = self.generator_full(generator_input_for_critic)
		critic_output_from_generator = self.critic(generated_samples_for_critic)
		critic_output_from_real_samples = self.critic(real_samples)

		# We also need to generate weighted-averages of real and generated samples,
		# to use for the gradient norm penalty.
		averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_critic])

		# We then run these samples through the discriminator as well. Note that we never
		# really use the discriminator output for these samples - we're only running them to
		# get the gradient norm for the gradient penalty loss.
		averaged_samples_out = self.critic(averaged_samples)

		# The gradient penalty loss function requires the input averaged samples to get
		# gradients. However, Keras loss functions can only have two arguments, y_true and
		# y_pred. We get around this by making a partial() of the function with the averaged
		# samples here.
		partial_gp_loss = partial(self.gradient_penalty_loss,
								  averaged_samples=averaged_samples,
								  gradient_penalty_weight=self.gradient_penalty_weight)
		# Functions need names or Keras will throw an error
		partial_gp_loss.__name__ = 'gradient_penalty'

		# If we don't concatenate the real and generated samples, however, we get three
		# outputs: One of the generated samples, one of the real samples, and one of the
		# averaged samples, all of size BATCH_SIZE. This works neatly!
		self.critic_model = Model(inputs=[real_samples,
											generator_input_for_critic],
									outputs=[critic_output_from_real_samples,
											 critic_output_from_generator, 
											 averaged_samples_out])

		# We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
		# the real and generated samples, and the gradient penalty loss for the averaged samples
		self.critic_model.compile(optimizer=optimizer,
									loss=[self.wasserstein_loss,
										  self.wasserstein_loss,
										  partial_gp_loss])

		# Save combined model structure on file
		with open('combined_struct.txt', 'w') as model_file:
			# Pass the file handle in as a lambda function to make it callable
			self.combined.summary(print_fn=lambda x: model_file.write(x + '\n'))


	# WGAN custom loss
	def wasserstein_loss(self, y_true, y_pred):
		return K.mean(y_true * y_pred)

	# Calculates the gradient penalty loss for a batch of "averaged" samples.
	def gradient_penalty_loss(self, y_true, y_pred, averaged_samples, gradient_penalty_weight):
   
		gradients = K.gradients(y_pred, averaged_samples)[0]
		# compute the euclidean norm by squaring ...
		gradients_sqr = K.square(gradients)
		#   ... summing over the rows ...
		gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
		#   ... and sqrt
		gradient_l2_norm = K.sqrt(gradients_sqr_sum)
		# compute lambda * (1 - ||grad||)^2 still for each single sample
		gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
		# return the mean as loss over all the batch samples
		return K.mean(gradient_penalty)


	def build_generator(self, model_name="Generator", pre_trained=True, res_blocks_num=2, masking=True):

		print("Loading Generator network...")

		if pre_trained:
			model = load_model('generator_model_LB_leaky.h5', custom_objects={"tf": tf, "keras": keras})
		else:
			model = generator_leaky.build_generator(input_shape=(64,), res_block_num=res_blocks_num, masking_on=masking)
			
		model.name = model_name

		print("Generator network - Loaded")

		return model

	def build_critic(self, model_name="Critic", pre_trained=True, res_blocks_num=2):

		print("Loading Critic network...")

		model = critic.build_critic(input_shape=(64,), res_block_num=res_blocks_num)

		model.name = model_name

		print("Critic network - Loaded")

		return model


	'''
	Performs training of the WGAN
	'''
	def train(self, epochs, batch_size=64, eval_interval=10, full_eval_interval=50, train_set="A", test_set="B"):

		self.logger.info("epochs: %d | batch_size: %d" % (epochs, batch_size))

		# Load the datasets for the Discriminator network
		train_x_discrim_feas = my_utils.load_GAN_data(dataset="DS_GAN/DS.UNIQUES.FEASIBLE." + train_set + ".txt")
		train_x_discrim_unfeas = my_utils.load_GAN_data(dataset="DS_GAN/DS.UNIQUES." + train_set + ".UNFEASIBLE.txt")

		train_x_discrim, train_y_discrim, test_x_discrim, test_y_discrim = my_utils.load_discriminator_data_v2(train_dataset="DS_DISCRIMINATOR/DS.UNIQUES." + train_set + ".ALL.txt", 
			test_dataset="DS_DISCRIMINATOR/DS.UNIQUES." + test_set + ".ALL.txt",
			validation=False)

		#load the 92 full solutions, for the N-Queens problem, into a dataframe
		full_solutions_df = pd.read_fwf("DS.FULL.SOLUTIONS.txt", widths=[1] * 64, header=None)

		# Load training and test set for the Generator network
		original_uniqe_trainset_gener = my_utils.load_GAN_data(dataset="DS_GENERATOR/DS.UNIQUES." + test_set + ".txt")
		original_uniqe_testset_gener = my_utils.load_GAN_data(dataset="DS_GENERATOR/DS.UNIQUES." + train_set + ".txt")

		original_trainset_gener = my_utils.load_GAN_data(dataset="DS_GENERATOR/DS.UNIQUES." + test_set + ".txt")
		original_testset_gener = my_utils.load_GAN_data(dataset="DS_GENERATOR/DS.UNIQUES." + train_set + ".txt")
		original_testset_gener_len = original_testset_gener.shape[0]
		num_samples = int(original_testset_gener_len / 2)

		
		# Split in two parts the DS used as the original test set for G; the first part it is used as training set, the second as test set
		
		original_testset_gener = shuffle(original_testset_gener)
		
		train_x_gener = original_testset_gener#[:num_samples]
		#test_x_gener = original_testset_gener[-num_samples:]
		'''
		# Generate noise DS for G training and G predictions
		train_x_gener, test_x_gener, testset_gener = my_utils.create_noise_ds_generator("DS_GENERATOR/DS.A.NEW.UNIQUES." + train_set + ".4.txt", num_single_solutions=550)
		'''

		# Adversarial ground truths
		valid = np.ones((batch_size, 1), dtype=np.float32)
		fake = -valid
		dummy = np.zeros((batch_size, 1), dtype=np.float32)
		

		for epoch in range(1, epochs + 1):

			print("\n### Epoch {}/{} ###".format(epoch, epochs))

			# Initialize counters for total G&D loss and G&D accuracy
			d_loss_tot = 0
			d_loss_real_tot = 0
			d_loss_fake_tot = 0
			d_acc_tot = 0
			g_loss_tot = 0
			g_acc_tot = 0

			# shuffle datasets
			train_x_discrim_feas = shuffle(train_x_discrim_feas)
			train_x_discrim_unfeas = shuffle(train_x_discrim_unfeas)
			train_x_gener = shuffle(train_x_gener)

			minibatches_size = batch_size * self.training_ratio
			num_batches = int(train_x_gener.shape[0] // (batch_size * self.training_ratio))

			# set up progress bar
			progress_bar = Progbar(target=num_batches)

			for index in range(num_batches):

				discriminator_minibatches = train_x_discrim_feas[index * minibatches_size:(index + 1) * minibatches_size]

				for j in range(self.training_ratio):

					### Critic ###

					# sample randomly from D dataset
					feas_solutions = discriminator_minibatches[j * batch_size:(j + 1) * batch_size]

					# Unfeasible data
					#unfeas_solutions = train_x_discrim_unfeas[j * batch_size:(j + 1) * batch_size]

					# Sample randomly from G dataset
					gen_samples = train_x_gener[j * batch_size:(j + 1) * batch_size]

					# Use the Generator Network to produce a batch of new assignments from the sampled partial solutions
					gen_solutions = self.generator_full.predict(gen_samples)

					# Train the Critic on valid and fake data
					d_loss_glob = self.critic_model.train_on_batch([feas_solutions, gen_solutions], [valid, fake, dummy])

					d_loss_real_tot += d_loss_glob[0]
					d_loss_fake_tot += d_loss_glob[1]
					
					# Compute total loss
					d_loss = 0.5 * np.add(d_loss_glob[0], d_loss_glob[1])
					d_loss_tot += d_loss


				### Generator ###

				gen_samples = train_x_gener[index * batch_size:(index + 1) * batch_size]

				# Train the generator (to have the critic label samples as valid)
				g_loss, g_acc = self.combined.train_on_batch(gen_samples, valid)

				g_loss_tot += g_loss
				g_acc_tot += g_acc


				progress_bar.update(index + 1)


			# Print loss and accuracy
			d_loss_avg = d_loss_tot/(num_batches+self.training_ratio)
			d_loss_real_avg = d_loss_real_tot/(num_batches+self.training_ratio)
			d_loss_fake_avg = d_loss_fake_tot/(num_batches+self.training_ratio)
			g_loss_avg = g_loss_tot/num_batches
			g_acc_avg = g_acc_tot/num_batches
			self.logger.info("---------------------------------------------------")
			self.logger.info("%d [D loss: %f (real: %f, fake: %f)] [G loss: %f, acc.: %.2f]" % 
				(epoch, d_loss_avg, d_loss_real_avg, d_loss_fake_avg, g_loss_avg, g_acc_avg))
			self.logger.info("---------------------------------------------------")
			

			# If at interval, compute the feasibility of generator's outputs on the full training & test sets, and eval critic on train & test sets
			if epoch % full_eval_interval == 0:

				# check feasibility of assignments produced by the generator
				global_feas_rate, feas_ratios = my_utils.check_solutions_feasibility(self.generator, pd.DataFrame(original_uniqe_testset_gener), "DS.FULL.SOLUTIONS.txt")
				
				# Print only on file
				self.logger.debug("Generator - Train set WGAN (original test set) - Feas.rate: %f" % global_feas_rate)
				self.logger.debug(feas_ratios)

				# check feasibility of assignments produced by the generator
				global_feas_rate, feas_ratios = my_utils.check_solutions_feasibility(self.generator, pd.DataFrame(original_uniqe_trainset_gener[:3000]), "DS.FULL.SOLUTIONS.txt")
				
				# Print only on file
				self.logger.debug("Generator - Test set WGAN (original training set) - Feas.rate: %f" % global_feas_rate)
				self.logger.debug(feas_ratios)

				# Evaluate critic on training set
				train_loss, train_acc = self.critic.evaluate(train_x_discrim, train_y_discrim)
				self.logger.info("critic - Train set WGAN loss: %f" % train_loss)
				self.logger.info("critic - Train set WGAN accuracy: %f" % train_acc)

				# Evaluate critic on test set
				test_loss, test_acc = self.critic.evaluate(test_x_discrim, test_y_discrim)
				self.logger.info("critic - Test set WGAN loss: %f" % test_loss)
				self.logger.info("critic - Test set WGAN accuracy: %f" % test_acc)

			# If at interval, compute the feasibility of G on a small test set and evaluate D on training set
			elif epoch % eval_interval == 0:

				num_unique_assign, first_criterion_perc, second_criterion_perc, third_criterion_perc = my_utils.stochastic_generation(self.generator, 
					full_sol_file="DS.FULL.SOLUTIONS.txt", 
					full_sol_trainset_file="DS_GENERATOR/DS.SOL.TRAINSET.txt", 
					num_queens_to_generate=8)

				self.logger.info("Num. unique assignments produced: %d" % num_unique_assign)
				self.logger.info("Generated solutions validity - 1st criterion: %.2f%%" % first_criterion_perc)
				self.logger.info("Generated solutions validity - 2nd criterion: %.2f%%" % second_criterion_perc)
				self.logger.info("Generated solutions validity - 3rd criterion: %.2f%%" % third_criterion_perc)



if __name__ == '__main__':
	wgan = WGAN()
	wgan.train(epochs=600, batch_size=64, eval_interval=10, full_eval_interval=300, train_set="B", test_set="A")
	#save generator model to file
	wgan.generator.save('generator_model_WGAN.h5')