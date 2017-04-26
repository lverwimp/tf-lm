#! /usr/bin/env python

import tensorflow as tf
from abc import ABCMeta
import run_epoch
import os

class trainer(object):
	'''
	Class for training a neural net with a fixed training scheme.
	'''
	
	def __init__(self, sv, session, config, train_lm, valid_lm, data, train_data, valid_data):
		
		self.sv = sv
		self.session = session
		self.config = config
		self.train_lm = train_lm
		self.valid_lm = valid_lm
		self.data = data
		self.train_data = train_data
		self.valid_data = valid_data
		self.model_name = self.config['save_path'] + os.path.basename(os.path.normpath(self.config['save_path']))

		self.train_runner = run_epoch.run_epoch(self.session, self.train_lm, self.data, self.train_data, eval_op=self.train_lm.train_op)
		
		self.valid_runner = run_epoch.run_epoch(self.session, self.valid_lm, self.data, self.valid_data, eval_op=None, test=False) 
		

	def train(self):

		stopOrNot = False

		for i in range(self.config['max_max_epoch']):
			
			self.decide_lr(i)

			if stopOrNot:
				break # stop training

			train_perplexity = self.train_runner()

			print('Epoch: {0} Train Perplexity: {1}'.format(i + 1, train_perplexity))

			valid_perplexity = self.valid_runner() # NEW CACHE
			
			print('Epoch: {0} Valid Perplexity: {1}'.format(i + 1, valid_perplexity))

			stopOrNot = self.decide_next_step(i, valid_perplexity)

		# final version of model = version of last epoch
		last_version = '{0}.{1}'.format(self.model_name,i+1)
		final_model = '{0}.final'.format(self.model_name)
		os.system('ln -f -s {0} {1}'.format(last_version, final_model))

	def validate(self):
		valid_perplexity = self.valid_runner()
		print('Valid Perplexity: {0}'.format(valid_perplexity))

	def decide_lr(self, i):
		# learning rate decay
		if 'lr_decay' in self.config:
			decay = self.config['lr_decay'] ** max(i + 1 - self.config['max_epoch'], 0.0)
			learning_rate = self.config['learning_rate'] * decay
		# fixed learning rate
		else:
			learning_rate = self.config['learning_rate']

		print('learning rate: {0}'.format(learning_rate))
		self.train_lm.assign_lr(self.session, learning_rate)

	def decide_next_step(self, i, valid_perplexity):

		if 'save_path' in self.config:
			print('Saving model to {0}.{1}'.format(self.model_name,i+1))
			self.sv.saver.save(self.session, '{0}.{1}'.format(self.model_name,i+1))

		# do not stop training
		return False


		
class earlyStopping(trainer):
	'''
	Train with early stopping (stop training if the validation ppl did not improve for the last x epochs)
	combined with exponential decay of the learning rate.
	'''
	def __init__(self, sv, session, config, train_lm, valid_lm, data, train_data, valid_data):

		super(earlyStopping, self).__init__(sv, session, config, train_lm, valid_lm, data, train_data, valid_data)

		if not isinstance(config['early_stop'], int):
			raise AssertionError('early_stop in config file should be an integer \
						(the number of validation ppls you compare with).')
		else:
			self.val_ppls = []
			self.num_times_no_improv = 0

	def decide_next_step(self, i, valid_perplexity):
		stopOrNot = False
		print('current list of validation ppls of previous epochs: {0}'.format(self.val_ppls))
				
		if i > self.config['early_stop']-1:
			print('epoch {0}: check whether validation ppl has improved'.format(i+1))

			for previous_ppl in self.val_ppls:
				if valid_perplexity >= previous_ppl:
					print('current validation ppl ({0}) is higher than previous validation ppl ({1})'.format(
						valid_perplexity, previous_ppl))
					self.num_times_no_improv += 1
				else:
					print('current validation ppl ({0}) is lower than previous validation ppl ({1})'.format(
						valid_perplexity, previous_ppl))

			self.val_ppls.pop(0)
		else:
			print('epoch {0}: do NOT check whether validation ppl has improved'.format(i+1))

		self.val_ppls.append(valid_perplexity)

		print('new list of validation ppls of previous epochs: {0}'.format(self.val_ppls))

		if self.num_times_no_improv >= self.config['early_stop']:
			# stop training
			print('Stop training.')
			stopOrNot = True
		else:
			if 'save_path' in self.config:
				print('Saving model to {0}.{1}'.format(self.model_name,i+1))
				self.sv.saver.save(self.session, '{0}.{1}'.format(self.model_name,i+1))

		return stopOrNot




class validHalve(trainer):
	'''
	Other type of early stopping: if validation ppl is not better than previous ppl, 
	halve the learning rate and continue training until the learning rate has been halved x times.
	'''
	def __init__(self, sv, session, config, train_lm, valid_lm, data, train_data, valid_data):

		super(validHalve, self).__init__(sv, session, config, train_lm, valid_lm, data, train_data, valid_data)

		if not isinstance(config['valid_halve'], int):
			raise ValueError(
				"Specify the number of times the learning rate may be halves before stopping training ('valid_halve').")

		self.previous_valid_ppl = 10000000
		self.num_times_halved = 0
		self.learning_rate = config['learning_rate']
		self.num_halves = config['valid_halve']

	def train(self):

		stopOrNot = False

		self.decide_lr(self.config['learning_rate'])

		for i in range(self.config['max_max_epoch']):

			if stopOrNot:
				break # stop training

			train_perplexity = self.train_runner()

			print('Epoch: {0} Train Perplexity: {1}'.format(i + 1, train_perplexity))

			valid_perplexity = self.valid_runner()
			print('Epoch: {0} Valid Perplexity: {1}'.format(i + 1, valid_perplexity))

			stopOrNot = self.decide_next_step(i, valid_perplexity)

	

	def decide_next_step(self, i, valid_perplexity):
		stopOrNot = False
		if valid_perplexity >= self.previous_valid_ppl:
			print('New valid ppl ({0}) bigger than previous ({1})'.format(valid_perplexity, self.previous_valid_ppl))
			self.num_times_halved += 1
			# if learning rate was already halved x times before, stop training loop
			if self.num_times_halved > self.num_halves:
				print('Learning rate was halved already {0} times. Stop training.'.format(self.num_halves))
				# stop training
				stopOrNot = True

			else:
				# halve the learning rate
				self.learning_rate = self.learning_rate / 2.0
				self.decide_lr(self.learning_rate)
					
		else:
			print('New valid ppl ({0}) smaller than previous ({1}). Continue training.'.format(
				valid_perplexity, self.previous_valid_ppl))
			if 'save_path' in self.config:
				print('Saving model to {0}'.format(self.model_name))
				self.sv.saver.save(self.session, self.model_name)

			self.previous_valid_ppl = valid_perplexity

		return stopOrNot

	def decide_lr(self, learning_rate):
		print('learning rate: {0}'.format(learning_rate))
		self.train_lm.assign_lr(self.session, learning_rate)
