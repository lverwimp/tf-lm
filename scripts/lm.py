#! /usr/bin/env python
# language model class

import tensorflow as tf
import functools, collections

class lm(object):

	def __init__(self, config, is_training, reuse):

		self.init_variables(config, reuse)
		
		output = self.lstm(is_training)

		self.create_output_weights()

		cost = self.calc_cost(output, is_training)
		
		# do not update weights if you are not training
		if not is_training:
			return
		else:
			self.update_model(cost)
			
	def init_variables(self, config, reuse):
		'''
		Initialize model constants/placeholders.
		'''
		self.config = config

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']
		self.size = config['size']
		self.vocab_size = config['vocab_size']
		self.use_cache = False
		self.reuse = reuse

		self.inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='inputs')
		self.targets = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_steps], name='targets')
		
		if 'per_sentence' in config:
			self.seq_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="sequence_lengths")
			
	def lstm(self, is_training):
		'''
		Create LSTM and get outputs.
		'''
		with tf.variable_scope("lstm"):
			
			self.cell = tf.contrib.rnn.MultiRNNCell(
					[tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=self.config['forget_bias'], 
					state_is_tuple=True, reuse=self.reuse) for _ in range(self.config['num_layers'])], 
					state_is_tuple=True)
			if is_training and self.config['dropout'] < 1:
				self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, output_keep_prob=self.config['dropout'])

			# for a network with multiple LSTM layers, 
			# initial state = tuple (size = number of layers) of LSTMStateTuples, 
			# each containing a zero Tensor for c and h (each batch_size x size) 
			self._initial_state = self.cell.zero_state(self.batch_size, tf.float32)

			inputs = self.get_input_embeddings(is_training)

			self.outputs, state = self.get_outputs(inputs)
			output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.size], name="reshape_output")
			
		return output

	def get_input_embeddings(self, is_training):
		'''
		Convert self.inputs (indices) to word embeddings.
		'''
		# embedding lookup table
		with tf.name_scope("embedding_lookup"):
			self.embedding = tf.get_variable("embedding", [self.vocab_size, self.size], dtype=tf.float32)
			# returns Tensor of size [batch_size x num_steps x size]
			inputs = tf.nn.embedding_lookup(self.embedding, self.inputs, name="input_embeddings")
			
			if is_training and self.config['dropout'] < 1:
				inputs = tf.nn.dropout(inputs, self.config['dropout'], name="dropout_inputs")
				
		return inputs
			
	def get_outputs(self, inputs):
		'''
		Feed inputs to network and get the output and new state.
		'''
		state = self._initial_state
		
		if 'per_sentence' in self.config:
			outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=self.seq_length, initial_state=state) 
		else:
			# feed inputs to network: outputs = predictions, state = new hidden state
			outputs, state = tf.nn.dynamic_rnn(self.cell, inputs, sequence_length=None, initial_state=state) 
			
		self._final_state = state
				
		return outputs, state
		
	def create_output_weights(self):
		''' 
		Weights of the output layer.
		'''
		# output weight matrix and bias
		with tf.variable_scope("output_layer_weights"):
			self.softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size], dtype=tf.float32)
			self.softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
		
	def calc_cost(self, output, is_training):
		'''
		Calculates the cost/average loss per batch.
		'''
		with tf.name_scope("get_outputs"):
			# logits/scores = Tensor of size [batch_size*num_steps x vocab_size]
			self.logits = tf.matmul(output, self.softmax_w) + self.softmax_b

			self._softmax = tf.nn.softmax(self.logits, name="softmax")

			self._loss = loss = self.get_loss(output, is_training)

			# sum all loss values
			self._reduce_loss = reduce_loss = tf.reduce_sum(loss, name="reduce_loss")

			# cost = average loss per batch
			self._cost = cost = reduce_loss / self.batch_size 
	
		return cost
			
	def update_model(self, cost):
		'''
		During training: update model parameters based on 'cost'.
		'''
	
		with tf.name_scope("train_model"):

			self._lr = tf.Variable(0.0, trainable=False)

			# tvars = list of trainable variables
			tvars = tf.trainable_variables()

			# calculate gradients of cost with respect to variables in tvars 
			# + clip the gradients by max_grad_norm: 
			#	for each gradient: gradient * max_grad_norm / max (global_norm, max_grad_norm)
			# 	where global_norm = sqrt(sum([l2norm(x)**2 for x in gradient]))
			# 	l2norm = sqrt(sum(squared values))
			grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), self.config['max_grad_norm'], name="clipped_gradients")

			self.global_step = tf.Variable(0, name='global_step', trainable=False)

			# get correct optimizer
			optimizer = self.get_optimizer()

			# apply gradients + increment global step
			self._train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				global_step=self.global_step,
				name="train_op")

			self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
			self._lr_update = tf.assign(self._lr, self._new_lr)
					

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	def get_loss(self, output, is_training):
		'''
		Calculates loss based on 'output'.
		Input:
			output: output of LSTM
			is_training: boolean that indicates whether we're training or not
		Returns:
			cross entropy loss based on full or sampled softmax
		'''
		if self.config['softmax'] == 'full':
			return tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=tf.reshape(self.targets, [-1]), 
					logits=self.logits) 

		elif self.config['softmax'] == 'sampled':
			# number of classes to randomly sample per batch
			NUM_SAMPLED = 32 

			# sampled softmax is only for training
			if is_training:
				return tf.nn.sampled_softmax_loss(
					weights=tf.transpose(self.softmax_w), 
					biases=self.softmax_b, 
					inputs=output, 
					labels=tf.reshape(self.targets, [-1,1]), 
					num_sampled=NUM_SAMPLED,
					num_classes=self.vocab_size) 
			else:
				return tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=tf.reshape(self.targets, [-1]), 
					logits=self.logits) 
		else:
			raise ValueError("Specify which softmax should be used: full or sampled.")


	def get_optimizer(self):
		if self.config['optimizer'] == 'sgd':
			return tf.train.GradientDescentOptimizer(self._lr)
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
	def cost(self):
		return self._cost

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
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def loss(self):
		return self._loss

	@property
	def softmax(self):
		return self._softmax

