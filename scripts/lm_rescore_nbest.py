# based on ptb tutorial (27/10/16)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os, sys

import numpy as np
import tensorflow as tf

import reader
import configuration 
from writer import writer

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config", None,"Configuration file")
FLAGS = flags.FLAGS

# turn this switch on if you want to see the mini-batches that are being processed
PRINT_SAMPLES = False 

# turn this switch on for debugging
DEBUG = True

def debug(string):
	if DEBUG:
		sys.stderr.write('DEBUG: {0}'.format(string))


class inputLM(object):
	"""The input data: words or characters."""

	def __init__(self, config, data, name=None):
		flattened_data = [word for sentence in data for word in sentence] # flatten list of lists
		self.batch_size = batch_size = config['batch_size']
		self.num_steps = num_steps = config['num_steps']
		self.epoch_size = ((len(flattened_data) // batch_size) - 1) // num_steps

		# input_data = Tensor of size batch_size x num_steps, same for targets (but shifted 1 step to the right)
		self.input_data, self.targets = reader.ptb_producer(data, config, name=name)



class LM(object):
	"""Word- or character-level LM."""

	def __init__(self, is_training, config, input_):
		self._input = input_
		self._input_sample = input_.input_data
		self._target_sample = input_.targets

		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config['word_size']
		vocab_size = config['vocab_size']

		if config['layer'] == 'LSTM':
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=config['forget_bias'], state_is_tuple=True)
		else:
			raise ValueError("Only LSTM layers implemented so far. Set layer = LSTM in config file.")

		# apply dropout
		if is_training and config['dropout'] < 1:
			single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=config['dropout'])

		# multiple hidden layers
		cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config['num_layers'], state_is_tuple=True)

		# for a network with multiple LSTM layers, 
		# initial state = tuple (size = number of layers) of LSTMStateTuples, each containing a zero Tensor for c and h (each batch_size x size) 
		self._initial_state = cell.zero_state(batch_size, tf.float32)

		# embedding lookup table
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
			#inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
			inputs = tf.nn.embedding_lookup(embedding, self._input_sample)

		if is_training and config['dropout'] < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		inputs = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, num_steps, inputs)]

		# feed inputs to network: outputs = predictions, state = new hidden state
		outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

		output = tf.reshape(tf.concat(1, outputs), [-1, size])

		# output layer weights
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

		# get scores
		logits = tf.matmul(output, softmax_w) + softmax_b

		# normalize scores -> probabilities
		softmax_output = tf.nn.softmax(logits)

		# reshape tensor of dimension [None,vocab_size] to [vocab_size]
		reshaped = tf.reshape(softmax_output, [vocab_size]) 

		# get probability of target word
		prob_tensor = tf.gather(reshaped, self._target_sample[0,:]) 
    		self._target_prob = tf.reduce_sum(prob_tensor) 

		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(self._target_sample, [-1])],
				[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state


		# do not update weights if you are not training
		if not is_training:
			return

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def input_data(self):
		return self._input_data

	@property
	def targets(self):
		return self._targets

	@property
	def input_sample(self):
		return self._input_sample

	@property
	def target_sample(self):
		return self._target_sample

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

  	@property
  	def target_prob(self):
    		return self._target_prob

def print_samples(input_sample, target_sample, id_to_word):
	''' For debugging purposes: if PRINT_SAMPLES = True, print each sample that is given to the model.'''
	print('input_sample:')
	for row in input_sample:
		for col in row:
			print('{0} '.format(id_to_word[col]), end="")
		print('')
	print('target_sample:')
	for row in target_sample:
		for col in row:
			print('{0} '.format(id_to_word[col]), end="")
		print('')



def run_epoch(session, model, id_to_word, out, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	total_log_prob = 0.0
	num_words = 0
	# state = initial state of the model
	state = session.run(model.initial_state)

	# fetches = what the graph will return
	fetches = {
		"cost": model.cost,
		"final_state": model.final_state, # c and h of previous time step (for each hidden layer)
		"input_sample": model.input_sample,
		"target_sample": model.target_sample,
		"target_prob": model.target_prob,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		# feed_dict = input for the graph
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		# feed the data ('feed_dict') to the graph and return everything that is in 'fetches'
		vals = session.run(fetches, feed_dict)

		# debugging: print every sample (input + target) that is fed to the model
		if PRINT_SAMPLES:
			print_samples(vals['input_sample'], vals['target_sample'], id_to_word)
			
		cost = vals["cost"]
		state = vals["final_state"]
		target_prob = vals["target_prob"]
		
		# end of sentence reached: write prob of current sentence to file and reset total_log_prob 
		if id_to_word[vals['target_sample'][0][0]] == '<eos>':
			total_log_prob += np.log10(target_prob)
			num_words += 1
			# final probability of sentence: sum of probabilities for all words normalised by the number of words
			out.write('{0}\n'.format(total_log_prob / num_words))
			total_log_prob = 0.0
			num_words = 0
		
		# only count probability for non-padding symbols
		elif id_to_word[vals['target_sample'][0][0]] != '@' or id_to_word[vals['input_sample'][0][0]] != '@':
			total_log_prob += np.log10(target_prob)
			num_words += 1


	#return total_log_probs


def main(_):
	if FLAGS.config == None:
		raise ValueError("Please specify a configuration file.")
	else:
		config = configuration.get_config(FLAGS.config)

	fout = file(config['log'],'w')
	sys.stdout = writer(sys.stdout, fout)

	print('configuration:')
	for par,value in config.iteritems():
		print('{0}\t{1}'.format(par, value))

	eval_config = config.copy() # same parameters for evaluation, except for:
	eval_config['batch_size'] = 1 # batch_size
	eval_config['num_steps'] = 1 # and number of steps

	# hypotheses = list of all hypotheses in n-best list
	all_data, id_to_word, total_length, hypotheses = reader.ptb_raw_data(config)

	# if processing per sentence
	if 'per_sentence' in config:
		# set num_steps = total length of each (padded) sentence
		config['num_steps'] = total_length
		# vocab is expanded with <bos> and padding symbol @
		config['vocab_size'] = len(id_to_word)
		eval_config['vocab_size'] = len(id_to_word)
		debug('vocabulary size: {0}\n'.format(config['vocab_size']))

	with tf.Graph().as_default():

		with tf.name_scope("Test"):
			test_hypotheses = inputLM(config=eval_config, data=hypotheses, name="Hypotheses")
			with tf.variable_scope("Model", reuse=None):
				mtest = wordLM(is_training=False, config=eval_config, input_=test_hypotheses)

		# sv = training helper that checkpoints models and computes summaries
		sv = tf.train.Supervisor(logdir=config['save_path'])

		# managed_session launches the checkpoint and summary services
		with sv.managed_session() as session:

			# restore variables from disk
      			sv.saver.restore(session, config['lm'])
      			print("Model restored.")

			out = open(config['result'], 'w')
			
			print('Start rescoring...')
			run_epoch(session, mtest, id_to_word, out)

			out.close()				


if __name__ == "__main__":
	tf.app.run()
