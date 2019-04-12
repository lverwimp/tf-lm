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

class lm_cache(lm):

	def __init__(self, config, word_weights, is_training, reuse, use_cache=False):

		super(lm_cache, self).__init__(config, is_training, reuse)

		self.use_cache = use_cache

		# only use cache if we are not training
		if not is_training and use_cache:

			with tf.name_scope("cache_model"):

				# create (+ initialize) all extra variables needed for the cache model
				self.init_cache_variables(word_weights)

				# split targets in a list of tensors per time_step
				self._targets_split = targets_split = tf.split(self.targets, self.num_steps, 1)

				# dummy operation (for first step in loop)
				update_op = tf.no_op(name="eval_op")

				# calculate cache prob and update cache for every time step
				list_cache_probs = []
				for time_step in range(self.num_steps):

					# update_op should be applied before executing the following steps (i.e. cache should be updated)
					with tf.control_dependencies([update_op]):

						# calculate cache prob
						if 'reg_cache' in self.config:
							# regular cache prob (based on frequency of words only)
							cache_probs = self.calc_reg_cache_prob()
						else:
							# neural cache prob (based on frequency of words + similarity between hidden states)
							cache_probs = self.calc_neural_cache_prob(self.outputs[time_step])
						list_cache_probs.append(cache_probs)

						print('targets_split[time_step]: {0}'.format(targets_split[time_step]))

						# update cache
						update_cache_words, update_cache_states = self.update_cache(
								targets_split[time_step], self.outputs[time_step])
						update_op = tf.group(update_cache_words, update_cache_states)

				# assign new cache probs to variable
				with tf.control_dependencies([update_op]):
					# concatenate the list of cache prob tensors for all time steps
					all_cache_probs = tf.concat(list_cache_probs, 0, name="cache_probs_all_steps")

					self.cache_probs_op = self.all_cache_probs.assign(all_cache_probs).op

				# calculate interpolation of normal and cache probabilities
				with tf.control_dependencies([self.cache_probs_op]):
					self.interpolate_probs(self.softmax, self.all_cache_probs)


				##### the code below is only used for rescoring (!!! batch size = 1 and num steps = 1) #####

				if 'num_hypotheses' in self.config:
					# get log probability of target word
					prob_target_interp = tf.gather(tf.reshape(self.result_interpolation,
						[self.vocab_size]), tf.gather(self.targets, 0))
					self.prob_target_interp_op = self.prob_target_interp.assign(prob_target_interp).op

					with tf.control_dependencies([self.prob_target_interp_op]):

						# new logprob for whole sentence: add logprob of current word to logprob of sentence
						new_prob = tf.add(prob_target_interp, self.prob_sentence_interp)
						self.prob_sentence_interp_op = self.prob_sentence_interp.assign(new_prob).op

					# if end of sentence reached, add cache words, cache states and
					# prob of sentence to memory of previous hypotheses
					self.update_prev_hyp_ops = self.update_cache_end_sentence()

					# keep track of cache of best hypothesis from previous segment
					self.keep_best_prev = self.get_best_prev_segment()


	def init_cache_variables(self, word_weights):
		'''
		This function initialized all extra variables needed for the cache model.
		Input:
			word_weights: None or dictionary of word-specific weights
		'''
		self.cache_size = self.config['cache_size']
		self.interp = tf.constant(self.config["interp"], name="interp_param")

		self.num_hyp = tf.Variable(0, name='num_hyp', trainable=False, dtype=tf.int32)
		self.increment_num_hyp_op = tf.assign(self.num_hyp, self.num_hyp+1)
		self.initialize_num_hyp_op = tf.assign(self.num_hyp, 0)

		if not 'reg_cache' in self.config:
			self.flatness = tf.constant(self.config["flatness"], name="flatness")

		if word_weights != None:
			weights = collections.OrderedDict(sorted(word_weights.items())).values()
			self.word_weights = tf.constant([weights])
			if 'scale_ww' in self.config:
				self.word_weights = self.config['scale_ww'] * self.word_weights

			if 'select_cache' in self.config:
				# if the weights are scaled, also scale the threshold!
				if 'scale_ww' in self.config:
					select_cache = self.config['scale_ww'] * self.config['select_cache']
				else:
					select_cache = self.config['select_cache']

				# check which word weights are greater than or equal to the threshold
				self.greater = greater = tf.greater_equal(self.word_weights, select_cache)

				# transpose and convert True to 1 and False to 0
				self.thresholded_weights = tf.transpose(tf.cast(greater, tf.float32),
					name="thresholded_weights")

		with tf.name_scope("cache"):
			# cache states
			self.cache_states = tf.Variable(tf.zeros([self.size, self.cache_size, self.batch_size]),
				name="cache_states", dtype=tf.float32)
			# cache words (initialize to vocab_size+1: this index means that the cache slot is empty)
			self.cache_words = tf.Variable(tf.fill([self.cache_size, self.batch_size], self.vocab_size+1),
				name="cache_words")

		with tf.name_scope("probs"):
			self.all_cache_probs = tf.Variable(tf.zeros([self.batch_size*self.num_steps,
					self.vocab_size]), name="cache_probabilities", dtype=tf.float32)

			self.result_interpolation = tf.Variable(tf.zeros([self.batch_size*self.num_steps,
					self.vocab_size]), name="interpolated_probs", dtype=tf.float32)

			self.prob_target_interp = tf.Variable(tf.zeros([1]),
					name="curr_prob_target", dtype=tf.float32)

			self.prob_sentence_interp = tf.Variable(tf.zeros([1]),
					name="curr_prob_sentence", dtype=tf.float32)


		if 'num_hypotheses' in self.config:
			# variables that keep track of probs, cache words and cache states from previous hypotheses
			with tf.name_scope("memory_previous_hypotheses"):
				self.cache_words_prev_hyp = tf.Variable(tf.fill([self.config['num_hypotheses'], self.cache_size,
					self.batch_size], self.vocab_size+1), name="memory_words")
				self.cache_states_prev_hyp = tf.Variable(tf.zeros([self.config['num_hypotheses'], self.size,
					self.cache_size, self.batch_size]), name="memory_states", dtype=tf.float32)

				self.sentence_probs_prev_hyp = tf.Variable(tf.zeros([self.config['num_hypotheses']]),
							name="memory_probs")

			with tf.name_scope("cache_best_prev"):
				# variables that will keep track with what the cache needs to be initialized
				self.cache_words_best_prev = tf.Variable(tf.fill([self.cache_size, self.batch_size],
					self.vocab_size+1), name="cache_words_best_prev")
				self.cache_states_best_prev = tf.Variable(tf.zeros([self.size, self.cache_size,
					self.batch_size]), name="cache_states_best_prev", dtype=tf.float32)

				# initialize cache with cache of best previous hypothesis
				init_words_best_prev = self.cache_words.assign(self.cache_words_best_prev).op
				init_states_best_prev = self.cache_states.assign(self.cache_states_best_prev).op

				self.init_cache_best_prev = tf.group(init_words_best_prev, init_states_best_prev, name="init_cache_best_prev")

				self.cache_words.read_value()
				self.cache_states.read_value()

		if 'num_hypotheses' in self.config:
			to_initialize = [self.cache_states, self.cache_words, self.cache_words_prev_hyp, self.cache_states_prev_hyp,
				self.sentence_probs_prev_hyp, self.cache_words_best_prev, self.cache_states_best_prev]
		else:
			to_initialize = [self.cache_states, self.cache_words]

		# initializes cache to empty cache - used at the beginning to avoid
		# using data stored in cache from previous session
		with tf.name_scope("init_cache_empty"):
			self.init_cache_op = tf.variables_initializer(to_initialize, name="init_cache")


	def calc_reg_cache_prob(self):
		'''
		Calculates the cache probability for a regular cache model.
		'''
		with tf.name_scope("calc_neural_cache_prob"):

			# remove last word from cache
			real_cache_words = tf.slice(self.cache_words, [0,0], [self.cache_size-1, self.batch_size],
					name="current_cache_words")

			# make one-hot vectors of words in the cache
			self.cache_words_one_hot  = cache_words_one_hot = tf.one_hot(real_cache_words, self.vocab_size, axis=0,
				name="cache_words_one_hot")

			# exponential decay
			if 'exp_decay' in self.config:
				print('apply exponential decay')

				if not 'decay_rate' in self.config:
					raise IOError("Specify a decay_rate in the config file.")

				weights_np = np.exp(-self.config['decay_rate']*np.arange(self.cache_size-1, 0, -1)) #NEW

				weights_tf = tf.constant(weights_np, dtype=tf.float32, name="exp_decay_weights")

				weighted_words = tf.multiply(tf.reshape(cache_words_one_hot, [self.vocab_size, self.cache_size-1]),
						weights_tf, name="weighted_words") #NEW

				cache_words_one_hot = tf.reshape(weighted_words, [self.vocab_size, self.cache_size-1, 1])

			# sum over one hot vectors to get frequencies
			self.sum_one_hot = sum_one_hot = tf.reduce_sum(cache_words_one_hot, axis=1)

			# normalize
			self.normalized_probs = normalized_probs = sum_one_hot / tf.reduce_sum(sum_one_hot)

			return tf.transpose(normalized_probs)

	def calc_neural_cache_prob(self, states_one_step):
		'''
		Calculates the neural cache probability.
		Input:
			states_one_step = Tensor of size [batch_size x size], states for one time step for all batches
		Returns:
			probs_all_batches: cache probabilities for all words in the vocabulary for all batches
						size = [batch_size x vocab_size]
		'''

 		with tf.name_scope("calc_neural_cache_prob"):

			# remove last word + state from cache
			real_cache_words = tf.slice(self.cache_words, [0,0], [self.cache_size-1, self.batch_size],
					name="current_cache_words")

			real_cache_states = tf.slice(self.cache_states, [0,0,0], [self.size, self.cache_size-1, self.batch_size],
					name="current_cache_states")

			# split cache_states in a list of batch_size Tensors of size [size x cache_size x 1]
			# each element in the list is thus the cache for a certain batch
			split_cache_states = tf.split(real_cache_states, self.batch_size, 2, name="split_cache_states_per_batch")

			# make one-hot vectors of words in the cache
			cache_words_one_hot = cache_words_one_hot = tf.one_hot(real_cache_words, self.vocab_size, axis=0,
					name="cache_words_one_hot") # NEW

			# add a row of 1's, to prevent 'not a number (nan)' at the end:
			# if all probs are 0, the normalization will try to divide by 0
			cache_words_one_hot = tf.concat([cache_words_one_hot, tf.ones([1, self.cache_size-1, self.batch_size])], 0)

			# split the dot products and one hot vectors per batch
			split_one_hots = tf.split(cache_words_one_hot, self.batch_size, 2, name="split_cache_words_one_hot")
			split_curr_states = tf.split(states_one_step, self.batch_size, 0, name="split_curr_states")

			list_cache_probs = []
			print('self.batch_size: {0}'.format(self.batch_size))
			# calculate cache probabilities per batch
			for batch_num in range(self.batch_size):
				# compute cache probability:
				# 	exp(flatness parameter * dot product of current hidden states in batch
				# 	with hidden states in cache for that batch)
				# result = Tensor of size [num_steps x cache_size]
				self.dot_product = dot_product = tf.exp(self.flatness * tf.matmul(split_curr_states[batch_num],
						tf.reshape(split_cache_states[batch_num], [self.size, self.cache_size-1])),
						name="unnorm_cache_prob_dot_product") # NEW

				self.one_hot_batch = one_hot_batch = tf.reshape(split_one_hots[batch_num], [self.vocab_size+1, self.cache_size-1],
						name="one_hot_cache_words_reshaped") # NEW

				print('self.one_hot_batch: {0}'.format(one_hot_batch))

				# multiply one hot vectors of words with cache probs to get cache probability for all words in the cache
				# if a word does not occur in the cache, the cache probability is 0
				# result = Tensor of size [vocab_size x num_steps]
				self.cache_probs_single_batch = cache_probs = tf.matmul(one_hot_batch, tf.transpose(dot_product),
						name="unnorm_cache_probs_vocab_dummy")

				# remove cache probability for dummy word (index vocab_size+1) (size = [vocab_size])
				self.cache_probs_sliced = cache_probs = tf.slice(tf.reshape(cache_probs, [self.vocab_size+1]), [0],
						[self.vocab_size], name="cache_probs_vocab_no_dummy")

				if self.config['weights_comb'] == 'info_log_linear':
					cache_probs = tf.add(cache_probs, 0.000001)

				# normalize cache probs
				self.cache_probs_batch_norm = cache_probs_batch = cache_probs / tf.reduce_sum(cache_probs)

				cache_probs_no_nan = tf.where(tf.is_nan(cache_probs_batch), tf.zeros_like(cache_probs_batch), cache_probs_batch)
				cache_probs_batch = cache_probs_no_nan

				list_cache_probs.append(cache_probs_batch)

			probs_all_batches = tf.stack(list_cache_probs, name="stack_cache_probs_all_batches")

			return probs_all_batches

	def update_cache(self, targets_one_step, states_one_step):
		'''
		Adds targets to self.cache_words and states to self.cache_states + reduces the size until cache_size.
		Inputs:
			targets_one_step: targets for all batches for one time step ([batch_size x 1])
			states_one_step: hidden states for all batches for one time step ([batch_size x size])
		Returns:
			update_cache_words: operation that effectively assigns the new words to the cache
			update_cache_states: operation that effectively assigns the new states to the cache
		'''

		with tf.name_scope("update_cache"):

			with tf.name_scope("update_words"):

				# 1) add the target words to the cache

				# reshape such that vector can be concatenated with matrix
				self._reshaped_words = reshaped_words = tf.reshape(targets_one_step, [1, self.batch_size],
						name="reshaped_words_one_step")

				# concatenate cache matrix with targets of current time step
				self._concat_words = tf.concat([self.cache_words, reshaped_words], 0, name="concat_cache_targets")

				# take slice starting at second element in cache (index 1) (first element drops out of cache)
				# until cache_size is reached (contains the new targets)
				sliced_words = tf.slice(self._concat_words, [1,0], [self.cache_size, self.batch_size],
						name="sliced_cache_targets")

				to_assign = sliced_words

				# only add words to the cache with an information weight > threshold
				if 'select_cache' in self.config:

					# make one-hot vectors of target words
					# size = [vocab_size x 1 x batch_size]
					self.target_words_one_hot = target_words_one_hot = tf.one_hot(targets_one_step,
						self.vocab_size, axis=0, name="target_words_one_hot")
					print('self.target_words_one_hot: {0}'.format(self.target_words_one_hot))
					print('tf.reshape(target_words_one_hot, [self.vocab_size, self.batch_size]: {0}'.format(tf.reshape(target_words_one_hot, [self.vocab_size, self.batch_size])))

					# multiply one-hot vectors with thresholded weights: only words with a weight
					# higher than the threshold/that have a 1 in mult should be added to the cache
					self.mult = mult = tf.multiply(tf.reshape(target_words_one_hot, [self.vocab_size, self.batch_size]),
						self.thresholded_weights)
					print('mult: {0}'.format(mult))

					# sum over weighted one-hots (if target word has zero weight, the 1 will become a 0)
					# + cast to integer
					self.sum_mult = sum_mult = tf.cast(tf.reduce_sum(mult, axis=0), tf.int32)

					# if the sum is larger than 0, condition is True --> word should be added
					condition = tf.greater(sum_mult, 0)
					self.condition = condition = tf.reshape(condition, [])

					# if the sum is larger than 0, assign new cache, otherwise use old cache
					self.to_assign = to_assign = tf.cond(condition, lambda: sliced_words, lambda: self.cache_words)

				# assign new tensor to self.cache_words
				update_cache_words = self.cache_words.assign(to_assign).op

			with tf.name_scope("update_states"):

				# 2) add the hidden states to the cache

				# reshape states to [size x 1 x batch_size] such that we can concatenate with cache_states
				reshaped_states = tf.reshape(tf.transpose(states_one_step), [self.size, 1, self.batch_size],
						name="reshaped_states_one_step")

				# concatenate cache_states with the new states
				concat_states = tf.concat([self.cache_states, reshaped_states], 1, name="concat_cache_states")

				# slice to reduce size of cache again
				sliced_states = tf.slice(concat_states, [0,1,0], [self.size, self.cache_size, self.batch_size],
						name="sliced_cache_states")

				to_assign = sliced_states

				# only add states to the cache with an information weight > threshold
				if 'select_cache' in self.config:
					# if the sum is larger than 0, assign new cache, otherwise use old cache
					to_assign = tf.cond(condition, lambda: sliced_states, lambda: self.cache_states)

				# assign new tensor to self.cache_states
				update_cache_states = self.cache_states.assign(to_assign).op

		return update_cache_words, update_cache_states

	def interpolate_probs(self, normal_probs, cache_probs):
		'''
		Calculates the interpolation of the softmax probabilities with the cache probabilities.
		Inputs:
			normal_probs: softmax probabilities of the LM, Tensor of size [batch_size*num_steps x vocab_size]
			cache_probs: cache probabilities, Tensor of size [batch_size*num_steps x vocab_size]
		Creates self.result_interpolation_op, which assigns the interpolated probabilities to self.result_interpolation.
		'''
		with tf.name_scope("interpolate_probabilities"):
			# information-weighted linear interpolation
			if self.config['weights_comb'] == 'info_linear':
				info_weighted_normal_probs = tf.multiply(normal_probs, (1-self.word_weights),
					name="weighted_normal_probs")
				info_weighted_cache_probs = tf.multiply(cache_probs, self.word_weights,
					name="weighted_cache_probs")

				result_interpolation = tf.log(tf.add(info_weighted_normal_probs, info_weighted_cache_probs,
					name="interpolated_probs"))
				self.result_interpolation_op = self.result_interpolation.assign(result_interpolation)

			# information-weighted log-linear interpolation
			elif self.config['weights_comb'] == 'info_log_linear':
				info_weighted_normal_log_probs = tf.pow(normal_probs, (1-self.word_weights),
					name="weighted_normal_log_probs")
				info_weighted_cache_log_probs = tf.pow(cache_probs, self.word_weights,
					name="weighted_cache_log_probs")

				result_interpolation = tf.log(tf.multiply(info_weighted_normal_log_probs, info_weighted_cache_log_probs,
					name="interpolated_probs"))
				self.result_interpolation_op = self.result_interpolation.assign(result_interpolation)

			# linear interpolation
			elif self.config['weights_comb'] == 'linear':
				self.weighted_normal_probs = weighted_normal_probs = tf.multiply(normal_probs, (1-self.interp), name="weighted_normal_probs")
				self.weighted_cache_probs = weighted_cache_probs =  tf.multiply(cache_probs, self.interp, name="weighted_cache_prob")

				result_interpolation = tf.log(tf.add(weighted_normal_probs, weighted_cache_probs,
					name="interpolated_probs"))
				self.result_interpolation_op = self.result_interpolation.assign(result_interpolation)

			else:
				raise ValueError("Specify an interpolation method (weights_comb): linear, info_linear or log_linear")

	def update_cache_end_sentence(self):
		'''
		Used for updating the memory of previous hypotheses while doing N-best rescoring.
			- prob of current sentence (self.prob_sentence_interp) is added to self.sentence_probs_prev_hyp
			- cache words of current sentence (self.cache_words) are added to self.cache_words_prev_hyp
			- cache states of currente sentence (self.cache_states) are added to self.cache_states_prev_hyp
		Returns:
			self.update_prev_hyp_ops: group of operations for updating the memory of previous hypotheses
		'''
		with tf.name_scope("update_prev_hyp"):
			with tf.control_dependencies([self.prob_sentence_interp_op]):

				# add prob of sentence to memory
				concat_probs = tf.concat([tf.reshape(self.sentence_probs_prev_hyp,
					[self.config['num_hypotheses'],1]),
					tf.reshape(self.prob_sentence_interp,[1,1])],
					0, name="concat_prev_sen_probs")

				sliced_probs = tf.slice(tf.reshape(concat_probs, [self.config['num_hypotheses']+1]), [1],
						[self.config['num_hypotheses']], name="sliced_prev_sen_probs")

				assign_prev_prob_op = self.sentence_probs_prev_hyp.assign(sliced_probs)

				# add cache words to memory
				concat_words = tf.concat([self.cache_words_prev_hyp,
					tf.reshape(self.cache_words, [1, self.cache_size, self.batch_size])],
					0, name="concat_prev_sen_words")

				sliced_words = tf.slice(concat_words, [1,0,0],
					[self.config['num_hypotheses'], self.cache_size, self.batch_size],
					name="sliced_prev_sen_words")

				assign_prev_words_op = self.cache_words_prev_hyp.assign(sliced_words)

				# add cache states to memory
				concat_states = tf.concat([self.cache_states_prev_hyp,
					tf.reshape(self.cache_states, [1, self.size, self.cache_size, self.batch_size])],
					0, name="concat_prev_sen_states")

				sliced_states = tf.slice(concat_states, [1,0,0,0],
					[self.config['num_hypotheses'], self.size, self.cache_size, self.batch_size],
					name="sliced_prev_sen_words")

				assign_prev_states_op = self.cache_states_prev_hyp.assign(sliced_states)

				return tf.group(assign_prev_prob_op , assign_prev_words_op, assign_prev_states_op)


	def get_best_prev_segment(self):
		'''
		Used for getting the best hypothesis (+ cache) from the previous segment while doing N-best rescoring.
		This function looks at what the most likely hypothesis was, retrieves its cache words and cache states
		and assigns them to self.cache_words_best_prev and self.cache_states_best_prev.
		Returns:
			self.keep_best_prev: group of operations to assign cache words and states of best hypothesis of previous segment

		'''
		with tf.name_scope("get_best_prev_segment"):
			with tf.control_dependencies([self.update_prev_hyp_ops]):

				self.index_best = tf.Variable(0, name='index_best', trainable=False, dtype=tf.int64)
				self.assign_index_best_op = tf.assign(self.index_best,
						tf.argmax(tf.slice(self.sentence_probs_prev_hyp,
						[self.config['num_hypotheses'] - self.num_hyp], [self.num_hyp])))

				with tf.control_dependencies([self.assign_index_best_op]):

					# this makes sure that self.index_best is actually updated
					self.index_best = self.index_best.read_value()

					# cache words belonging to hypothesis with highest prob
					cache_words_best_prev = tf.slice(self.cache_words_prev_hyp,
						[self.config['num_hypotheses'] - self.num_hyp + tf.cast(self.index_best, dtype=tf.int32), 0, 0],
						[1, self.cache_size, self.batch_size], name="words_best_prev_hyp")

					keep_words_best_prev = self.cache_words_best_prev.assign(
						tf.reshape(cache_words_best_prev, [self.cache_size, self.batch_size])).op

					self.cache_words_best_prev.read_value()

					# cache states belonging to hypothesis with highest prob
					cache_states_best_prev = tf.slice(self.cache_states_prev_hyp,
						[self.config['num_hypotheses'] - self.num_hyp + tf.cast(self.index_best, dtype=tf.int32), 0, 0, 0],
						[1, self.size, self.cache_size, self.batch_size], name="states_best_prev_hyp")

					keep_states_best_prev = self.cache_states_best_prev.assign(
						tf.reshape(cache_states_best_prev, [self.size, self.cache_size, self.batch_size])).op

					self.cache_states_best_prev.read_value()

					return tf.group(keep_words_best_prev, keep_states_best_prev, name="keep_best_words_states")
