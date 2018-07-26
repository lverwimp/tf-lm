#! /usr/bin/env python

import tensorflow as tf
from abc import ABCMeta
import run_epoch
import os

class trainer(object):
	'''
	Class for training a neural net.
	This class continues training for max_max_epoch epochs (specified in config file).
	'''

	def __init__(self, session, saver, config, train_lm, valid_lm, data, train_data, valid_data):

		self.session = session
		self.saver = saver
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

			print('Start training epoch...')
			train_perplexity = self.train_runner()

			print('Epoch: {0} Train Perplexity: {1}'.format(i + 1, train_perplexity))
			
			self.saver.save(self.session, os.path.join(self.config['save_path'],'epoch{0}'.format(i)))

			print('Start validating...')
			valid_perplexity = self.valid_runner()

			print('Epoch: {0} Valid Perplexity: {1}'.format(i + 1, valid_perplexity))

			stopOrNot = self.decide_next_step(i, valid_perplexity)

	def validate(self):
		valid_perplexity = self.valid_runner()
		print('Valid Perplexity: {0}'.format(valid_perplexity))

	def decide_lr(self, i):

		fetches = {
			"lr": self.train_lm.lr,
			"epoch": self.train_lm.epoch,
		}

		vals = self.session.run(fetches, feed_dict={})
		learning_rate = vals["lr"]
		i = vals["epoch"]

		# learning rate decay
		if 'lr_decay' in self.config:
			decay = self.config['lr_decay'] ** max(i + 1 - self.config['max_epoch'], 0.0)
			learning_rate = self.config['learning_rate'] * decay

		print('Learning rate: {0}'.format(learning_rate))

		self.train_lm.assign_lr(self.session, learning_rate)
		self.train_lm.assign_epoch(self.session, (i+1))

	def decide_next_step(self, i, valid_perplexity):

		# no early stopping
		return False



class earlyStopping(trainer):
	'''
	Train with early stopping: stop training if the validation ppl did not improve for the last x epochs.
	Config file should contain 'early_stop' + integer.
	'''
	def __init__(self, session, saver, config, train_lm, valid_lm, data, train_data, valid_data):

		super(earlyStopping, self).__init__(session, saver, config, train_lm, valid_lm, data, train_data, valid_data)

		if not isinstance(config['early_stop'], int):
			raise IOError('early_stop in config file should be an integer \
						(the number of validation ppls you compare with).')
		else:
			self.val_ppls = []
			self.num_times_no_improv = 0

	def decide_next_step(self, i, valid_perplexity):
		stopOrNot = False

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

		self.val_ppls.append(valid_perplexity)

		if self.num_times_no_improv >= self.config['early_stop']:
			# stop training
			stopOrNot = True
		else:
			if 'save_path' in self.config:
				print('Saving model to {0}.{1}'.format(self.model_name,i+1))
				self.saver.save(self.session, '{0}.{1}'.format(self.model_name,i+1))

		return stopOrNot
