#! /usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math, os, sys

# set to True if you want to see what the batches look like
PRINT_SAMPLES = False

class run_epoch(object):
	'''
	Calls batch generator, feeds the batches to the model and calculates the perplexity.
	'''
	def __init__(self, session, model, data_object, data_set, eval_op=None, test=False):

		self.session = session
		self.model = model
		self.data_object = data_object
		self.data_set = data_set
		self.eval_op = eval_op
		self.test = test
		
			
	def __call__(self):	
		costs = 0.0 # cross entropy based on normal (LM only) probabilities
		iters = 0
		# state = initial state of the model
		state = self.session.run(self.model.initial_state)

		fetches = {
			"cost": self.model.cost,
			"final_state": self.model.final_state, # c and h of previous time step (for each hidden layer)
			"input_sample": self.model.input_sample,
			"target_sample": self.model.target_sample,
			"softmax": self.model.softmax
			}
		
		# self.eval_op = _train_op in training phase	
		if self.eval_op is not None:
			fetches["eval_op"] = self.eval_op
				
		end_reached = False

		while True:		
			# get batch of data
			if self.test:
				if 'per_sentence' in self.model.config:
					x, y, end_reached, seq_lengths = self.data_object.get_batch(self.data_set, test=True)
				else:
					x, y, end_reached = self.data_object.get_batch(self.data_set, test=True)
			else:
				if 'per_sentence' in self.model.config:
					x, y, end_reached, seq_lengths = self.data_object.get_batch(self.data_set)
				else:
					x, y, end_reached = self.data_object.get_batch(self.data_set)
					
			# if end of data file is reached, end of epoch is reached
			if end_reached:
				break

			# create feed_dict
			if 'per_sentence' in self.model.config:
				feed_dict = {self.model.inputs: x, self.model.targets: y, self.model.seq_length: seq_lengths}
			else:
				feed_dict = {self.model.inputs: x, self.model.targets: y}
			
			# use state of previous time step/initial state in the beginning
			for i, (c, h) in enumerate(self.model.initial_state):
				feed_dict[c] = state[i].c
				feed_dict[h] = state[i].h

			# run the model
			vals = self.session.run(fetches, feed_dict)

			# debugging: print every sample (input + target) that is fed to the model
			if PRINT_SAMPLES:
				self.print_samples(vals['input_sample'], vals['target_sample'])			

			# sentence-level processing: reset state after each sentence
			if 'per_sentence' in self.model.config:
				# testing: batch size = 1, so first check if the end of the sentence is reached
				if self.test:
					end_of_sentence = ['<eos>', self.data_object.PADDING_SYMBOL]
					if self.data_object.id_to_item[vals['target_sample'][0][0]] in end_of_sentence:
						state = self.session.run(self.model.initial_state)
					else:
						state = vals["final_state"]
				# training/validation: 1 batch = 1 sentence, so always reset the state
				else:
					state = self.session.run(self.model.initial_state)
			# otherwise, use final state of previous state as initial state for next batch
			else:
				state = vals["final_state"]

			cost = vals["cost"]
			softmax = np.transpose(vals["softmax"])

			costs += cost
			iters += self.model.num_steps	
			
			ppl = np.exp(costs / iters)


		return ppl
		
	def print_samples(self, input_sample, target_sample):
		''' For debugging purposes: if PRINT_SAMPLES = True, print each sample that is given to the model.'''
		print('input_sample:', end="")
		
		if 'add_word' in self.model.config:
			word_input_sample = input_sample[1]
			input_sample = input_sample[0]

		for row in input_sample:
			for col in row:

				if 'char_ngram' in self.model.config:

					# loop over positions in numpy array
					for pos in range(len(col)):

						if col[pos] > 0.0:
							if col[pos] > 1.0:
								print(u'{0} ({1}) '.format(self.data_object.id_to_ngram[pos], 
									col[pos]).encode('utf-8'), end="")
							else:
								# id_to_item[0] = id_to_ngram
								print(u'{0} '.format(self.data_object.id_to_ngram[pos]).encode(
									'utf-8'), end="")
					print('; ', end='')
					
				elif 'word_char_concat' in self.model.config:
					print(u'{0} '.format(self.data_object[0].id_to_item[col]).encode('utf-8'), end="")

				else:
					print(u'{0} '.format(self.data_object.id_to_item[col]).encode('utf-8'), end="")

			print('')

		if 'add_word' in self.model.config:
			print('input_sample in words:', end='')
			for row in word_input_sample:
				for col in row:
					print(u'{0} '.format(self.data_object.id_to_item[1][col]).encode('utf-8'), end="")
				print('')

		print('target_sample:', end="")
		for row in target_sample:
			for col in row:

				#if 'char_ngram' in self.model.config:
					# id_to_item[0] = id_to_word
				#	print(u'{0} '.format(self.data_object.id_to_item[1][col]).encode('utf-8'), end="")
				#else:
				
				if 'word_char_concat' in self.model.config:
					print(u'{0} '.format(self.data_object[0].id_to_item[col]).encode('utf-8'), end="")
				else:
					print(u'{0} '.format(self.data_object.id_to_item[col]).encode('utf-8'), end="")
	
			print('')

class rescore(run_epoch):
	'''
	Get probabilities per sentence, for a sentence-level LM.
	Difference with run_epoch: __call__ needs the hypothesis to rescore as argument.
	'''

	def __init__(self, session, model, data_object, data_set, eval_op=None, test=False):

		super(rescore, self).__init__(session, model, data_object, data_set, eval_op, test)
		
		try:
			self.results_f = open(self.model.config['result'], 'w', 0)
		except IOError:
			print('Failed opening results file {0}'.format(self.model.config['result'])) 
			
		self.counter_hypotheses = 0

	def __call__(self, hypothesis):
		iters = 0
		total_log_prob = 0.0
		# state = initial state of the model
		state = self.session.run(self.model.initial_state)
		
		# counter for the number of words
		counter = 0
		
		fetches = {
			"cost": self.model.cost,
			"final_state": self.model.final_state, # c and h of previous time step (for each hidden layer)
			"input_sample": self.model.input_sample,
			"target_sample": self.model.target_sample,
			"softmax": self.model.softmax
			}
		
		# _train_op in training phase	
		if self.eval_op is not None:
			fetches["eval_op"] = self.eval_op
				
		end_reached = False

		while True:

			# format data
			x = np.array(hypothesis[counter]).reshape((1,1))
			y = np.array(hypothesis[counter+1]).reshape((1,1))

			# create feed_dict
			feed_dict = {self.model.inputs: x, self.model.targets: y}
			
			for i, (c, h) in enumerate(self.model.initial_state):
				feed_dict[c] = state[i].c
				feed_dict[h] = state[i].h

			# run the model
			vals = self.session.run(fetches, feed_dict)	
			
			# debugging: print every sample (input + target) that is fed to the model
			if PRINT_SAMPLES:
				self.print_samples(vals['input_sample'], vals['target_sample'])

			softmax = vals['softmax']
			state = vals["final_state"]
			
			# target/next word
			next_word = self.data_object.id_to_item[vals['target_sample'][0][0]]
			
			# probability of 'target' word = next word in the n-best hypothesis
			prob_next_word = softmax[0][y[0][0]]
			
			# end of sentence reached: write prob of current sentence to file and stop loop 
			if next_word == '<eos>':
				total_log_prob += np.log10(prob_next_word)
		
				self.results_f.write('{0}\n'.format(total_log_prob))
				break

			else:
				total_log_prob += np.log10(prob_next_word)

			counter += 1
			








