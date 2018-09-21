#! /usr/bin/env python
# language model class

import tensorflow as tf
import numpy as np
import functools, collections

class lm(object):
	'''
	Standard LSTM Language Model:
	Predicting the next word given the previous words, or
	predicting the next character given the previous characters.
	'''

	def __init__(self, config, is_training, reuse):
		'''
		Arguments:
			config: configuration dictionary
			is_training: boolean indicating whether we are training or not
			reuse: boolean indicating whether variables should be shared or not
		'''

		self.init_variables(config, reuse)

		# create model
		self.create_graph(is_training)
		self.create_output_weights()

		output = self.get_output(is_training)

		cost = self.calc_cost(output, is_training)

		# do not update weights if you are not training
		if not is_training:
			return
		else:
			self.update_model(cost)

	def init_variables(self, config, reuse):
		'''
		Initialize class variables.
		'''
		self.config = config

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']
		self.size = config['size']
		self.vocab_size = config['vocab_size']
		self.reuse = reuse

		if not 'output_vocab_size' in config:
			self.output_vocab_size = self.vocab_size
		else:
			self.output_vocab_size = config['output_vocab_size']

		self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='inputs')
		self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='targets')

		if 'per_sentence' in config:
			self.seq_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="sequence_lengths")

			if 'bidirectional' in config:
				length_mask = self.num_steps - 1
			else:
				length_mask = self.num_steps

			bool_mask = tf.sequence_mask(self.seq_length, length_mask)
			float_mask = tf.cast(bool_mask, tf.float32)
			self.final_mask = tf.reshape(float_mask, [self.batch_size*length_mask])


	def create_graph(self, is_training):
		'''
		Creates LSTM graph.
		'''
		with tf.variable_scope("lstm"):

			self.cell = tf.contrib.rnn.MultiRNNCell(
					[tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=self.config['forget_bias'],
					state_is_tuple=True, reuse=self.reuse) for _ in range(self.config['num_layers'])],
					state_is_tuple=True)

			if is_training and self.config['dropout'] < 1:
				self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.config['dropout'])

			if 'bidirectional' in self.config:
				self.cell_bw = tf.contrib.rnn.MultiRNNCell(
					[tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=self.config['forget_bias'],
					state_is_tuple=True, reuse=self.reuse) for _ in range(self.config['num_layers'])],
					state_is_tuple=True)
				if is_training and self.config['dropout'] < 1:
					self.cell_bw = tf.contrib.rnn.DropoutWrapper(self.cell_bw, output_keep_prob=self.config['dropout'])

			# for a network with multiple LSTM layers,
			# initial state = tuple (size = number of layers) of LSTMStateTuples,
			# each containing a zero Tensor for c and h (each batch_size x size)
			self._initial_state = self.cell.zero_state(self.batch_size, tf.float32)

			if 'bidirectional' in self.config:
				self._initial_state_bw = self.cell_bw.zero_state(self.batch_size, tf.float32)

	def create_output_weights(self):
		'''
		Creates output weight matrix and bias.
		'''

		if 'bidirectional' in self.config:
			output_size = 2 * self.size
		else:
			output_size = self.size

		# output weight matrix and bias
		with tf.variable_scope("output_layer_weights"):
			self.softmax_w = tf.get_variable("softmax_w",
				[output_size, self.output_vocab_size], dtype=tf.float32)
			self.softmax_b = tf.get_variable("softmax_b",
				[self.output_vocab_size], dtype=tf.float32)

	def get_output(self, is_training):
		'''
		Feeds self.inputs to the graph and returns the output.
		'''
		with tf.variable_scope("get_output"):
			input_embeddings = self.get_input_embeddings(is_training)

			self.outputs, state = self.feed_to_lstm(input_embeddings)

			if 'bidirectional' in self.config:
				output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.size*2], name="reshape_output")
			else:
				output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.size], name="reshape_output")

		return output

	def get_input_embeddings(self, is_training):
		'''
		Creates embedding lookup table and returns the embeddings for self.inputs.
		'''

		with tf.name_scope("embedding_lookup"):
			self.embedding = tf.get_variable("embedding", [self.vocab_size, self.size], dtype=tf.float32)

			# returns Tensor of size [batch_size x num_steps x size]
			inputs = tf.nn.embedding_lookup(self.embedding, self.inputs, name="input_embeddings")

			# use droput on the input embeddings
			if is_training and self.config['dropout'] < 1:
				inputs = tf.nn.dropout(inputs, self.config['dropout'], name="dropout_inputs")

		return inputs

	def feed_to_lstm(self, inputs):
		'''
		Feeds input embeddings and returns the outputs and the hidden state.
		Input arguments:
			inputs: input embeddings; Tensor of size [batch_size x num_steps x size]
		Returns:
			outputs: outputs of the LSTM; Tensor of size [batch_size x num_steps x size]
			state: the hidden states of the LSTM after processing all inputs; Tensor of size [batch-size x size]
		'''

		state = self._initial_state
		if 'bidirectional' in self.config:
			state_bw = self._initial_state_bw

		if 'bidirectional' in self.config:
			if 'per_sentence' in self.config:
				(outputs_fw, outputs_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
					self.cell, self.cell_bw, inputs, sequence_length=self.seq_length,
					initial_state_fw=state, initial_state_bw=state_bw)
			else:
				(outputs_fw, outputs_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
					self.cell, self.cell_bw, inputs, sequence_length=None,
					initial_state_fw=state, initial_state_bw=state_bw)

			# concatenate outputs and states of forward and backward LSTM
			# time step of current input = t
			# since we are training a LM, we should use the forward output from time step t,
			# but the backward output from time step t+1 because that output is generated (in TF) by reversing
			# the input sequence before feeding it to the RNN and then reversing the output again
			# hence, for an input sequente "the cat lies on the mat", the forward output contains p(cat|the), p(lies|the cat)...
			# and the backward output contains p(the|mat...cat), p(cat|mat...lies)
			# since the first element of the backward output already contains the target of the first element of the forward output,
			# we delete that first element
			if self.num_steps > 1:
				outputs_fw = tf.slice(outputs_fw, [0,0,0], [self.batch_size, self.num_steps-1, self.size]) # NEW
				outputs_bw = tf.slice(outputs_bw, [0,1,0], [self.batch_size, self.num_steps-1, self.size]) # NEW

			outputs = tf.concat([outputs_fw, outputs_bw], 2)
			state = state_fw

		elif 'per_sentence' in self.config:
			outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=self.seq_length, initial_state=state)

		else:
			# feed inputs to network: outputs = predictions, state = new hidden state
			outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=None, initial_state=state)

		self._final_state = state
		if 'bidirectional' in self.config:
			self._final_state_bw = state_bw

		return outputs, state


	def calc_cost(self, output, is_training):
		'''
		Calculates final predictions and the loss.
		'''

		with tf.name_scope("get_outputs"):
			# logits/scores = Tensor of size [batch_size*num_steps x vocab_size]
			self.logits = tf.matmul(output, self.softmax_w) + self.softmax_b

			self.softmax = tf.nn.softmax(self.logits, name="softmax")

			self.loss = self.get_loss(output, is_training)

			# sentence-level batching: mask the loss for the padding
			# no mask during testing since num_steps = 1 and batch_size = 1
			if 'per_sentence' in self.config and (self.num_steps > 1 or self.batch_size > 1):
				self.masked_loss = tf.multiply(self.loss, self.final_mask)
				self.unnormalized_loss = tf.reduce_sum(self.masked_loss, name="reduce_loss")
				self.cost = self.unnormalized_loss / self.batch_size

			else:

				# sum all loss values
				self.reduce_loss = tf.reduce_sum(self.loss, name="reduce_loss")

				# cost = average loss per batch
				self.cost = self.reduce_loss / self.batch_size

		return self.cost

	def update_model(self, cost):
		with tf.name_scope("train_model"):

			self.lr = tf.Variable(float(self.config['learning_rate']), trainable=False, name="learning_rate")
			self.epoch = tf.Variable(0, trainable=False, name="epoch")

			# tvars = list of trainable variables
			tvars = tf.trainable_variables()

			# calculate gradients of cost with respect to variables in tvars
			# + clip the gradients by max_grad_norm:
			#	for each gradient: gradient * max_grad_norm / max (global_norm, max_grad_norm)
			# 	where global_norm = sqrt(sum([l2norm(x)**2 for x in gradient]))
			# 	l2norm = sqrt(sum(squared values))
			grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.config['max_grad_norm'], name="clipped_gradients")

			# get correct optimizer
			optimizer = self.get_optimizer()

			# apply gradients + increment global step
			self.train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				tf.train.get_global_step(),
				name="train_op")

			self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
			self.lr_update = tf.assign(self.lr, self.new_lr)

			self.new_epoch = tf.placeholder(tf.int32, shape=[], name="new_epoch")
			self.epoch_update = tf.assign(self.epoch, self.new_epoch)


	def assign_lr(self, session, lr_value):
		'''
		Assign lr_value as learning rate.
		'''
		session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

	def assign_epoch(self, session, epoch_value):
		'''
		Assign epoch_value as epoch.
		'''
		session.run(self.epoch_update, feed_dict={self.new_epoch: epoch_value})

	def get_loss(self, output, is_training):
		'''
		Calculates loss based on 'output'.
		Input:
			output: output of LSTM
			is_training: boolean that indicates whether we're training or not
		Returns:
			loss based on full or sampled softmax
		'''
		if self.config['softmax'] == 'full':

			if 'bidirectional' in self.config and self.num_steps > 1:
				targets = tf.reshape(tf.slice(self.targets, [0,0], [self.batch_size, self.num_steps-1]), [-1])
			else:
				targets = tf.reshape(self.targets, [-1])

			return tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=targets,
					logits=self.logits)

		elif self.config['softmax'] == 'sampled':
			# number of classes to randomly sample per batch
			NUM_SAMPLED = 32

			# sampled softmax is only for training
			if is_training:

				if 'bidirectional' in self.config:
					targets = tf.reshape(tf.slice(self.targets, [0,0], [self.batch_size, self.num_steps-1]), [-1,1])
				else:
					targets = tf.reshape(self.targets, [-1,1])

				return tf.nn.sampled_softmax_loss(
					weights=tf.transpose(self.softmax_w),
					biases=self.softmax_b,
					inputs=output,
					labels=targets,
					num_sampled=NUM_SAMPLED,
					num_classes=self.vocab_size)
			else:

				if 'bidirectional' in self.config:
					targets = tf.reshape(tf.slice(self.targets, [0,0], [self.batch_size, self.num_steps-1]), [-1])
				else:
					targets = tf.reshape(self.targets, [-1])

				return tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=targets,
					logits=self.logits)
		else:
			raise ValueError("Specify which softmax should be used: full or sampled.")


	def get_optimizer(self):
		'''
		Returns the optimizer asked for.
		'''
		if self.config['optimizer'] == 'sgd':
			return tf.train.GradientDescentOptimizer(self.lr)
		elif self.config['optimizer'] == 'adam':
			# default learning rate = 1e-3
			return tf.train.AdamOptimizer(self.config['learning_rate'])
		elif self.config['optimizer'] == 'adagrad':
			return tf.train.AdagradOptimizer(self.config['learning_rate'])
		else:
			raise ValueError("Specify an optimizer: stochastic gradient descent (sgd), adagrad or adam.")


	@property
	def initial_state(self):
		return self._initial_state

	@property
	def initial_state_bw(self):
		return self._initial_state_bw

	@property
	def input_sample(self):
		return self.inputs

	@property
	def target_sample(self):
		return self.targets

	@property
	def final_state(self):
		return self._final_state

	@property
	def final_state_bw(self):
		return self._final_state_bw

class lm_ngram(lm):
	'''
	LSTM that takes character n-grams + optionally word embeddings as input
	and predicts a distribution over words in the output.
	'''

	def __init__(self, config, is_training, reuse):

		super(lm_ngram, self).__init__(config, is_training, reuse)

	def init_variables(self, config, reuse):
		self.config = config

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']
		self.size = config['size']
		self.vocab_size = config['vocab_size']
		self.output_vocab_size = self.vocab_size
		self.use_cache = False
		self.reuse = reuse

		if 'per_sentence' in config:
			self.seq_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="sequence_lengths")

		if 'add_word' in self.config:
			self.word_size = config['word_size']
			self.input_vocab_size = config['input_vocab_size']
			self.char_size = self.size - self.word_size
			print('size for words: {0}'.format(self.word_size))
			print('vocab size for input: {0}'.format(self.input_vocab_size))
		else:
			self.char_size = self.size


		# n-gram input
		self.inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_steps, self.config['input_size']], name='inputs')
		# word input
		if 'add_word' in self.config:
			self.input_words = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='input_words')
		self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='targets')


	def get_input_embeddings(self, is_training):
		'''
		Instead of a simple word embedding lookup, this returns an n-gram embedding lookup
		or a concatenation of n-gram and word embeddings.
		'''
		with tf.name_scope("embedding_lookup"):
			self.embedding = tf.get_variable("embedding", [self.config['input_size'], self.char_size], dtype=tf.float32)

			# in order to do matrix-matrix multiplication, we need to reduce the dimension of self.inputs from 3 to 2:
			# reshape inputs of size [batch_size x num_steps x input_size] to [batch_size*num_steps x input_size]
			reshaped = tf.reshape(self.inputs, [self.batch_size * self.num_steps, self.config['input_size']])

			# self._input_embeddings = Tensor of size [batch_size*num_steps x char_size]
			input_embeddings = tf.matmul(reshaped, self.embedding)

			# reshape back to format [batch_size x num_steps x char_size]
			inputs_reshaped = tf.reshape(input_embeddings, [self.batch_size, self.num_steps, self.char_size])

			if 'add_word' in self.config:
				self.word_embedding = tf.get_variable("word_embedding",
					[self.input_vocab_size, self.word_size], dtype=tf.float32)

				# word_inputs = Tensor of size [batch_size x num_steps x word_size]
				word_inputs = tf.nn.embedding_lookup(self.word_embedding, self.input_words)

				# append character n-gram and word inputs
				inputs_reshaped = tf.concat([inputs_reshaped, word_inputs], 2)

		return inputs_reshaped


class lm_charwordconcat(lm):
	'''
	LSTM that takes the concatenation of word and character embeddings as input
	and predicts a distribution over words in the output.
	'''

	def __init__(self, config, is_training, reuse):

		super(lm_charwordconcat, self).__init__(config, is_training, reuse)

	def init_variables(self, config, reuse):
		self.config = config

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']
		self.size = config['size']
		self.vocab_size = config['vocab_size']
		self.output_vocab_size = self.vocab_size
		self.reuse = reuse

		if 'per_sentence' in config:
			self.seq_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="sequence_lengths")

		if self.config['order'] == 'both':
			self.num_char = 2 * self.config['num_char']
		else:
			self.num_char = self.config['num_char']
		self.char_size = self.config['char_size']
		self.word_size = self.size - (self.num_char * self.char_size)
		self.char_vocab_size = self.config['char_vocab_size']

		self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='inputs')
		self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='targets')

		self.input_chars = [self.input_char(pos) for pos in range(self.num_char)]

	def input_char(self, pos):
		with tf.variable_scope(str(pos)):
			return tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='char_input')

	def get_input_embeddings(self, is_training):
		'''
		Instead of a simple word embedding lookup, this returns an n-gram embedding lookup
		or a concatenation of n-gram and word embeddings.
		'''
		with tf.name_scope("embedding_lookup"):

			all_char_embeddings = []
			for pos in range(self.num_char):
				char_embedding = self.get_char_embedding(pos)
				all_char_embeddings.append(char_embedding)

			# append character embeddings
			char_inputs = tf.concat(all_char_embeddings, 2)

			self.word_embedding = tf.get_variable("word_embedding", [self.vocab_size, self.word_size], dtype=tf.float32)

			# word_inputs = Tensor of size [batch_size x num_steps x word_size]
			word_inputs = tf.nn.embedding_lookup(self.word_embedding, self.inputs)

			# append character and word inputs
			inputs_concat = tf.concat([char_inputs, word_inputs], 2)

			if is_training and self.config['dropout'] < 1:
				inputs_concat = tf.nn.dropout(inputs_concat, self.config['dropout'], name="dropout_inputs")

		return inputs_concat

	def get_char_embedding(self, pos):
		with tf.variable_scope(str(pos)):
			self.char_embedding = tf.get_variable("char_embedding", [self.char_vocab_size, self.char_size], dtype=tf.float32)

			curr_char = self.input_chars[pos]

			char_input = tf.nn.embedding_lookup(self.char_embedding, curr_char)

			return char_input
