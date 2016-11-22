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

### Condor ###
if 'LD_LIBRARY_PATH' not in os.environ:
	os.environ['LD_LIBRARY_PATH'] = '/users/spraak/lverwimp/.local/lib/python2.7/site-packages/tensorflow:/users/spraak/lverwimp/.local/lib/python2.7/site-packages/cuda:/usr/local/cuda-7.5/lib64:/usr/local/cuda-8.0/lib64'
	try:
		os.system('/usr/bin/python ' + ' '.join(sys.argv))
		sys.exit(0)
	except Exception, exc:
		sys.exit(1)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("config", None,"Configuration file")
FLAGS = flags.FLAGS

# turn this switch on if you want to see the mini-batches that are being processed
PRINT_SAMPLES = False 

# turn this switch on for debugging
DEBUG = False

def debug(string):
	if DEBUG:
		sys.stderr.write('DEBUG: {0}'.format(string))

class wordInput(object):
	"""The input data: words."""

	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config['batch_size']
		self.num_steps = num_steps = config['num_steps']
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps

		# input_data = Tensor of size batch_size x num_steps, same for targets (but shifted 1 step to the right)
		self.input_data, self.targets = reader.ptb_producer(data, config, name=name)


class wordLM(object):
	"""Word-based LM."""

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
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

		if is_training and config['dropout'] < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		inputs = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, num_steps, inputs)]

		# feed inputs to network: outputs = predictions, state = new hidden state
		outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

		output = tf.reshape(tf.concat(1, outputs), [-1, size])
		softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(input_.targets, [-1])],
				[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state

		# do not update weights if you are not training
		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)

		# tvars = list of trainable variables
		tvars = tf.trainable_variables()

		# calculate gradients of cost with respect to variables in tvars 
		# + clip the gradients by max_grad_norm: 
		#	for each gradient: gradient * max_grad_norm / max (global_norm, max_grad_norm)
		# 	where global_norm = sqrt(sum([l2norm(x)**2 for x in gradient]))
		# 	l2norm = sqrt(sum(squared values))
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config['max_grad_norm'])

		# optimizer: stochastic gradient descent or adam
		if config['optimizer'] == 'sgd':
			optimizer = tf.train.GradientDescentOptimizer(self._lr)
		elif config['optimizer'] == 'adam':
			optimizer = tf.train.AdamOptimizer()
		else:
			raise ValueError("Specify an optimizer: stochastic gradient descent (sgd) or adam")

		# apply gradients + increment global step
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())

		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)


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




def run_epoch(session, model, id_to_word, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	# state = initial state of the model
	state = session.run(model.initial_state)

	fetches = {
		"cost": model.cost,
		"final_state": model.final_state, # c and h of previous time step (for each hidden layer)
		"input_sample": model.input_sample,
		"target_sample": model.target_sample,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)

		if PRINT_SAMPLES:
			print('input_sample:')
			for row in vals["input_sample"]:
				for col in row:
					print('{0} '.format(id_to_word[col]), end="")
				print('')
			print('target_sample:')
			for row in vals["target_sample"]:
				for col in row:
					print('{0} '.format(id_to_word[col]), end="")
				print('')

		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), iters * model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)


def main(_):
	if FLAGS.config == None:
		raise ValueError("Please specify a configuration file.")
	else:
		config = configuration.get_config(FLAGS.config)

	fout = file(os.path.join('logs', os.path.basename(config['name'])) + '.log','w')
	sys.stdout = writer(sys.stdout, fout)

	print('configuration:')
	for par,value in config.iteritems():
		print('{0}\t{1}'.format(par, value))

	eval_config = config.copy() # same parameters for evaluation, except for:
	eval_config['batch_size'] = 1 # batch_size
	eval_config['num_steps'] = 1 # and number of steps

	# all_data = tuple (train_data, valid_data, test_data), id_to_word = mapping from ids to words, 
	# total_length = total length of all padded sentences in case the data is processed per sentence
	all_data, id_to_word, total_length = reader.ptb_raw_data(config)
	train_data = all_data[0]
	valid_data = all_data[1]
	test_data = all_data[2]

	# if processing per sentence
	if 'per_sentence' in config:
		# set num_steps = total length of each (padded) sentence
		config['num_steps'] = total_length
		# vocab is expanded with <bos> and padding symbol @
		config['vocab_size'] = len(id_to_word)
		eval_config['vocab_size'] = len(id_to_word)

	with tf.Graph().as_default():

		# always use the same seed for random initialization (to better compare models)
		tf.set_random_seed(1) 
		initializer = tf.random_uniform_initializer(-config['init_scale'], config['init_scale'])

		with tf.name_scope("Train"):
			train_input = wordInput(config=config, data=train_data, name="TrainInput")
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				m = wordLM(is_training=True, config=config, input_=train_input)
			tf.scalar_summary("Training Loss", m.cost)
			tf.scalar_summary("Learning Rate", m.lr)

		with tf.name_scope("Valid"):
			valid_input = wordInput(config=config, data=valid_data, name="ValidInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mvalid = wordLM(is_training=False, config=config, input_=valid_input)
			tf.scalar_summary("Validation Loss", mvalid.cost)

		with tf.name_scope("Test"):
			test_input = wordInput(config=eval_config, data=test_data, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = wordLM(is_training=False, config=eval_config,input_=test_input)

		# sv = training helper that checkpoints models and computes summaries
		sv = tf.train.Supervisor(logdir=config['save_path'])

		# managed_session launches the checkpoint and summary servieces
		with sv.managed_session() as session:

			if 'early_stop' in config:
				debug('early stopping\n')
				if DEBUG and not isinstance(config['early_stop'], int):
					raise AssertionError('early_stop in config file should be an integer \
						(the number of validation ppls you compare with).')
				else:
					val_ppls = []

			# training loop
			for i in range(config['max_max_epoch']):

				# calculate exponential decay
				lr_decay = config['lr_decay'] ** max(i + 1 - config['max_epoch'], 0.0)

				# assign new learning rate to session + run the session
				m.assign_lr(session, config['learning_rate'] * lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))

				train_perplexity = run_epoch(session, m, id_to_word, eval_op=m.train_op, verbose=True)
				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

				valid_perplexity = run_epoch(session, mvalid, id_to_word)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

				if 'early_stop' in config:
					num_times_no_improv = 0
					debug('current list of validation ppls of previous epochs: {0}\n'.format(val_ppls))
					
					if i > config['early_stop']-1:
						debug('epoch {0}: check whether validation ppl has improved\n'.format(i+1))

						if DEBUG and (len(val_ppls) != config['early_stop']):
							raise AssertionError('Length of list of validation ppls should be equal to the early stopping value.')

						for previous_ppl in val_ppls:
							if valid_perplexity >= previous_ppl:
								debug('current validation ppl ({0}) is higher than previous validation ppl ({1})\n'.format(valid_perplexity, previous_ppl))
								num_times_no_improv += 1
							else:
								debug('current validation ppl ({0}) is lower than previous validation ppl ({1})\n'.format(valid_perplexity, previous_ppl))

						val_ppls.pop(0)
					else:
						debug('epoch {0}: do NOT check whether validation ppl has improved\n'.format(i+1))

					val_ppls.append(valid_perplexity)

					debug('new list of validation ppls of previous epochs: {0}\n'.format(val_ppls))

					if num_times_no_improv == config['early_stop']:
						best_model = 0
						best_ppl = val_ppls[0]
						# find previous model with best validation ppl
						for idx, previous_ppl in enumerate(val_ppls[1:]):
							if previous_ppl < best_ppl:
								best_ppl = previous_ppl
								best_model = idx
				
						# filename of the best model
						file_best_model = '{0}.{1}'.format(config['name'], i - (config['early_stop'] - best_model))
						name_best_model = '{0}.final'.format(config['name'])
						debug('model with best validation ppl: epoch {0} (ppl {1})'.format(best_model, best_ppl))

						# set best model to 'final model'
						os.system('ln -s {0} {1}'.format(file_best_model, name_best_model))
						break
					else:
						if 'save_path' in config:
							print('Saving model to {0}.{1}'.format(config['name'],i+1))
							#sv.saver.save(session, '{0}best_valid_ppl_{1}'.format(config['save_path'], i) , global_step=sv.global_step)
							sv.saver.save(session, '{0}.{1}'.format(config['name'],i+1))
			

			test_perplexity = run_epoch(session, mtest, id_to_word)
			print("Test Perplexity: %.3f" % test_perplexity)

			# no early stopping: just take model of last epoch as final model 
			if not 'early_stop' in config:
				print('No early stopping, saving final model to {0}.final'.format(config['name']))
				#sv.saver.save(session, '{0}.final'.format(config['name']), global_step=sv.global_step)
				sv.saver.save(session, '{0}.final'.format(config['name']))


if __name__ == "__main__":
	tf.app.run()
