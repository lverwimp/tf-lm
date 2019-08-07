#! /usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np
import math, os, sys, time, io

PRINT_SAMPLES = False # if PRINT_SAMPLES is True, all input and target batches are printed
PRINT_INTERMEDIATE = True # if PRINT_INTERMEDIATE is True, ppl and time after every 100 batches are printed

class run_epoch(object):
	'''
	Runs one epoch (one pass over all data) of training, or calculates the validation or test perplexity.
	'''
	def __init__(self, session, model, data_object, data_set, eval_op=None, test=False, cache=False):

		self.session = session
		self.model = model
		self.data_object = data_object
		self.data_set = data_set
		self.eval_op = eval_op
		self.test = test
		self.use_cache = cache

		if 'cache_size' in self.model.config and self.model.use_cache:
			# initialize cache to empty cache
			self.session.run(self.model.init_cache_op)

	def __call__(self):
		costs = 0.0 # cross entropy based on normal (LM only) probabilities
		iters = 0
		cache_iters = 0
		final_interp_cross_entropy = 0.0 # cross entropy based on interpolated probabilities

		# state = initial state of the model
		state = self.get_init_state()

		# create fetches dictionary = what we want the graph to return
		fetches = self.create_fetches()

		# fetches = tuple for cache model
		if self.model.use_cache:
			fetches, fetches_two = fetches

		end_reached = False

		# initialize batching
		if 'per_sentence' and 'stream_data' in self.model.config:
			data_file = self.init_batching()
		else:
			self.init_batching()
			data_file = None

		# start iterating over data
		while True:
			start_time = time.time()

			# get batch of data
			x, y, end_reached, seq_lengths = self.get_batch(data_file)

			# if end of data file is reached, end of epoch is reached
			if end_reached:
				break

			# testing: ignore padding symbols for sentence-level models
			# (training: padding symbols are fed to the graph but ignored with seq_length)
			if self.test and 'per_sentence' in self.model.config:
				if 'word_char_concat' in self.model.config:
					# self.data_object = tuple of LMData and multipleLMDataChar, x = tuple of word and characters
					if self.data_object[0].id_to_item[int(x[0][0][0])] == self.data_object[0].PADDING_SYMBOL or \
							self.data_object[0].id_to_item[int(y[0][0][0])] == self.data_object[0].PADDING_SYMBOL:
						continue
				else:
					if self.data_object.id_to_item[int(x[0][0])] == self.data_object.PADDING_SYMBOL or \
							self.data_object.id_to_item[int(y[0][0])] == self.data_object.PADDING_SYMBOL:
						continue

			# create feed_dict = what we feed into the graph
			feed_dict = self.create_feed_dict(x, y, state, seq_lengths)

			# run the model
			# vals contains the values for the variables specified in fetches after feeding feed_dict to the model
			vals = self.session.run(fetches, feed_dict)

			if self.model.use_cache:
				# 2nd run to check whether the variables changed the way we wanted it
				vals_two = self.session.run(fetches_two, feed_dict)

			# debugging: print every sample (input + target) that is fed to the model
			if PRINT_SAMPLES:
				self.print_samples(vals['input_sample'], vals['target_sample'])

			# determine new state: emtpy state or state of previous time step
			state = self.get_new_state(vals)

			if 'per_sentence' in self.model.config and (self.model.num_steps > 1 or self.model.batch_size > 1):
				costs += vals["unnormalized_loss"]
				iters += np.sum(seq_lengths)
			else:
				# cost = -ln(prob_target_word) / batch_size
				costs += vals["cost"]
				iters += self.model.num_steps

			# if it's a cache model, calculate the ppl including the cache probs
			if self.model.use_cache:
				interp_ppl, final_interp_cross_entropy, cache_iters = self.calc_cache_ppl(
						vals, vals_two,final_interp_cross_entropy, cache_iters, seq_lengths)
			else:
				interp_ppl = None

			ppl = np.exp(costs / iters)

			# if PRINT_INTERMEDIATE is True, ppl and time after each batch is printed
			# can be changed to only printing after processing a certain amount of data
			if PRINT_INTERMEDIATE and (iters % (self.model.num_steps*100) == 0):
				print('ppl {0} ({1} seconds)'.format(ppl, time.time() - start_time))

		return ppl, interp_ppl

	def get_init_state(self):
		'''
		Initialize to empty state.
		'''

		state = self.session.run(self.model.initial_state)
		if 'bidirectional' in self.model.config:
			state_bw = self.session.run(self.model.initial_state_bw)
			state = (state, state_bw)

		return state

	def get_final_state(self, vals):
		'''
		Initialize to state of previous time step.
		'''

		if self.use_cache and 'rescore' in self.model.config:
			state = vals[0]["final_state"]
		else:
			state = vals["final_state"]

		if 'bidirectional' in self.model.config:
			state_bw = vals["final_state_bw"]
			state = (state, state_bw)

		return state

	def get_new_state(self, vals):
		'''
		Determine to what the state of the next time step should be initialized.
		'''

		# sentence-level processing
		if 'per_sentence' in self.model.config:
			# sentence-level testing mode
			if self.test:
				# end of sentence: initialize state to zeros
				if self.data_object.id_to_item[vals['target_sample'][0][0]] in ['<eos>',self.data_object.PADDING_SYMBOL]:
					state = self.get_init_state()
				# otherwise: use final state of previous batch
				else:
					state = self.get_final_state(vals)
			# sentence-level training mode: 1 batch = 1 sentence, so reset state after every batch,
			# unless explicitly mentioned in the config file with 'across_sentence'
			else:
				if 'across_sentence' in self.model.config:
					state = self.get_final_state(vals)
				else:
					state = self.get_init_state()
		# otherwise, use final state of previous state as initial state for next batch
		else:
			state = self.get_final_state(vals)

		return state

	def create_fetches(self):
		'''
		Creates a dictionary containing model variables for which we want the new values.
		'''

		fetches = {
			"cost": self.model.cost,
			"final_state": self.model.final_state, # c and h of previous time step (for each hidden layer)
			"input_sample": self.model.input_sample,
			"target_sample": self.model.target_sample,
			}

		# _train_op in training phase
		if self.eval_op is not None:
			fetches["eval_op"] = self.eval_op

		if self.model.use_cache:
			fetches["result_interpolation_op"] = self.model.result_interpolation_op
			fetches["cache_probs_op"] = self.model.cache_probs_op

			fetches_two = {
				"result_interpolation": self.model.result_interpolation,
				"all_cache_probs": self.model.all_cache_probs,
				"cache_words": self.model.cache_words
				}

			fetches = (fetches, fetches_two)

		if 'bidirectional' in self.model.config:
			fetches["final_state_bw"] = self.model.final_state_bw

		if 'per_sentence' in self.model.config and (self.model.num_steps > 1 or self.model.batch_size > 1):
			fetches["unnormalized_loss"] = self.model.unnormalized_loss

		return fetches

	def create_feed_dict(self, x, y, state, seq_lengths):
		'''
		Creates a dictionary containing the data that will be fed to the placeholders of the model.
		'''

		if 'add_word' in self.model.config:
			x_char_ngrams, x_words = x
			feed_dict = {self.model.inputs: x_char_ngrams, self.model.input_words: x_words, self.model.targets: y}

		elif 'word_char_concat' in self.model.config:
			x_words, x_chars = x
			feed_dict = {self.model.inputs: x_words, self.model.targets: y}
			for pos in range(self.model.num_char):
				feed_dict[self.model.input_chars[pos]] = x_chars[pos]

		elif 'per_sentence' in self.model.config:
			feed_dict = {self.model.inputs: x, self.model.targets: y, self.model.seq_length: seq_lengths}

		else:
			feed_dict = {self.model.inputs: x, self.model.targets: y}

		if 'bidirectional' in self.model.config:
			state, state_bw = state

		for i, (c, h) in enumerate(self.model.initial_state):
			feed_dict[self.model.initial_state[i]] = state[i]

			if 'bidirectional' in self.model.config:
				feed_dict[self.model.initial_state_bw[i]] = state_bw[i]

		return feed_dict


	def init_batching(self):
		'''
		Initialize batching.
		'''

		if self.test:
			if 'per_sentence' in self.model.config:
				if 'stream_data' in self.model.config:
					data_file = self.data_object.init_batching(self.data_set)
					return data_file
				else:
					self.data_object.init_batching(self.data_set, test=True)
			elif 'word_char_concat' in self.model.config:
				self.data_object[0].init_batching(self.data_set[0], test=True)
				self.data_object[1].init_batching(self.data_set[1], test=True)
			else:
				self.data_object.init_batching(self.data_set, test=True)
		else:
			if 'per_sentence' in self.model.config:
				if 'stream_data' in self.model.config:
					data_file = self.data_object.init_batching(self.data_set)
					return data_file
				else:
					self.data_object.init_batching(self.data_set)
			elif 'word_char_concat' in self.model.config:
				self.data_object[0].init_batching(self.data_set[0])
				self.data_object[1].init_batching(self.data_set[1])
			else:
				self.data_object.init_batching(self.data_set)

	def get_batch(self, data_file):
		'''
		Get a single batch.
		Arguments:
			data_file: in case of streaming data, data_file is the file from
				which the data is read; otherwise it is None
		Returns:
			x: input
			y: target
			end_reached: boolean indicating whether the end of the file is reached or not
			seq_lengths: length of every sentence for sentence-level batching; otherwise, empty list
		'''

		seq_lengths = []
		if 'per_sentence' in self.model.config or 'char_rnn' in self.model.config:
			if 'stream_data' in self.model.config:
				x, y, end_reached, seq_lengths = self.data_object.get_batch(data_file, self.test)
			else:
				x, y, end_reached, seq_lengths = self.data_object.get_batch()

		elif 'word_char_concat' in self.model.config:
			x_words, y, end_reached = self.data_object[0].get_batch()
			x_chars = self.data_object[1].get_batch()
			x = (x_words, x_chars)

		else:
			x, y, end_reached = self.data_object.get_batch()

		return x, y, end_reached, seq_lengths

	def calc_cache_ppl(self, vals, vals_two, final_interp_cross_entropy, cache_iters, seq_lengths):

		# placeholder for cross entropy of current sample
		tmp_interp_ce = 0.0

		cache_probs = vals_two["all_cache_probs"]
		sum_cache_probs = np.sum(cache_probs)

		# only calculate prob if cache is non-empty (= if cache probs are not nan)
		if not np.isnan(sum_cache_probs):
			# to calculate perplexity for cache model
			cache_iters += self.model.num_steps

			targets_reshaped = vals["target_sample"].reshape([self.model.batch_size*self.model.num_steps])
			result_interpolation = vals_two["result_interpolation"]

			counter_num_steps = 0
			counter_batches = 0
			for i in range(self.model.batch_size*self.model.num_steps):

				if 'per_sentence' in self.model.config and i > seq_lengths[counter_batches] - 1:
					print('out of sentence')
					pass
				else:
					# negative log prob of target word according to interpolated probs
					tmp_interp_ce += -(result_interpolation[0][targets_reshaped[i]])

				counter_num_steps += 1

				if counter_num_steps % self.model.num_steps == 0:
					counter_batches += 1

			# normalize over batches
			tmp_interp_ce = tmp_interp_ce / self.model.batch_size

			# add to total cross entropy
			final_interp_cross_entropy += tmp_interp_ce

			# divide by cache_iters
			interp_ppl = np.exp(final_interp_cross_entropy / (cache_iters))

		else:
			interp_ppl = None

		return interp_ppl, final_interp_cross_entropy, cache_iters

	def print_samples(self, input_sample, target_sample):
		'''
		For debugging purposes: if PRINT_SAMPLES = True, print each sample that is given to the model.
		'''

		print('input_sample:', end="")
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

		print('target_sample:', end="")
		for row in target_sample:
			for col in row:

				if 'word_char_concat' in self.model.config:
					print(u'{0} '.format(self.data_object[0].id_to_item[col]).encode('utf-8'), end="")
				else:
					print(u'{0} '.format(self.data_object.id_to_item[col]).encode('utf-8'), end="")

			print('')

class rescore(run_epoch):
	'''
	Used for re-scoring hypotheses, generating next word(s) or generating debug file.
	'''

	def __init__(self, session, model, data_object, data_set, eval_op=None, test=False, cache=False):

		super(rescore, self).__init__(session, model, data_object, data_set, eval_op, test, cache)

		try:
			os.makedirs(os.path.dirname(self.model.config['result']))
		except OSError:
			pass

		try:
			self.results_f = io.open(self.model.config['result'], 'w',
				buffering=1, encoding=self.data_object.encoding)
		except IOError:
			print('Failed opening results file {0}'.format(self.model.config['result']))
			sys.exit(1)

		# variables to calculate precision, recall and f1
		if 'predict_next' in self.model.config:
			self.print_predictions = False
			self.num_predictions = 0

		# counter for hypotheses for cache model
		if 'cache_size' in self.model.config and self.model.use_cache:
			feed_dict = {self.model.inputs: [[0]], self.model.targets: [[0]]}
			self.session.run(self.model.initialize_num_hyp_op, feed_dict)
			self.counter_hypotheses = 0
			self.prev_id = ''
			self.first_word = True

	def __call__(self, hypothesis):

		total_log_prob = 0.0
		self.print_predictions = False
		self.num_predictions = 0

		# state = initial state of the model
		state = self.get_init_state()

		# counter for the number of words
		if 'hyp_with_ids':
			# skip hypothesis id
			counter = 1
		else:
			counter = 0

		fetches, fetches_two = self.create_fetches()

		end_reached = False
		predicted_word = None

		if self.model.use_cache:

			# initialize cache
			self.init_cache(hypothesis, counter)

		while True:

			# format data
			if 'predict_next' in self.model.config and self.print_predictions:
				x, y, hypothesis = self.format_data(hypothesis, counter, predicted_word)
			else:
				x, y, hypothesis = self.format_data(hypothesis, counter)

			if 'predict_next' in self.model.config:
				input_word = self.data_object.id_to_item[x[0][0]]
				# if the end of the sentence is not yet reached
				# and if we are not printing the predictions
				# (prediction will be added to the hypothesis, so we don't have to print it twice)
				if input_word != self.data_object.PADDING_SYMBOL and \
						not self.print_predictions and \
						input_word != '<bos>':

					if 'interactive' in self.model.config:
						if input_word == '<eos>':
							print('')
						else:
							if 'char' in self.model.config:
								print(u'{0}'.format(input_word).encode('utf-8'), end='')
							else:
								print(u'{0} '.format(input_word).encode('utf-8'), end='')

					if 'char' in self.model.config:
						self.results_f.write(u'{0}'.format(input_word))
					else:
						self.results_f.write(u'{0} '.format(input_word))

			# testing: ignore padding symbols for sentence-level models
			if 'per_sentence' in self.model.config:
				if 'word_char_concat' in self.model.config:
					# self.data_object = tuple of LMData and multipleLMDataChar, x = tuple of word and characters
					if self.data_object[0].id_to_item[x[0][0][0]] == self.data_object[0].PADDING_SYMBOL or \
							self.data_object[0].id_to_item[y[0][0][0]] == self.data_object[0].PADDING_SYMBOL:
						continue
				else:
					if self.data_object.id_to_item[x[0][0]] == self.data_object.PADDING_SYMBOL or \
							self.data_object.id_to_item[y[0][0]] == self.data_object.PADDING_SYMBOL:
						continue

			# create feed_dict
			feed_dict = self.create_feed_dict(x, y, state)

			# run the model
			vals = self.session.run(fetches, feed_dict)

			if self.use_cache:
				# 2nd run to check whether the variables changed the way we wanted it
				vals_two = self.session.run(fetches_two, feed_dict)
				vals = (vals, vals_two)
				softmax = vals[0]['softmax']
			else:
				softmax = vals['softmax']

			state = self.get_final_state(vals)

			current_word, next_word = self.get_words(vals)

			prob_next_word = self.get_prob_next_word(vals, softmax, y, next_word)

			# debugging: print every sample (input + target) that is fed to the model
			if PRINT_SAMPLES:
				if 'add_word' in self.model.config:
					input_samples = (vals['input_sample'], vals["input_sample_words"])
				else:
					input_samples = vals['input_sample']
				self.print_samples(input_samples, vals['target_sample'])

			# print file similar to -debug 2 files in SRILM, to compute optimal interpolation weights
			if 'debug2' in self.model.config:

				prob_next_word = softmax[0][y[0][0]]
				log_prob_next_word = np.log10(prob_next_word)

				if next_word == '<eos>':
					next_word = '</s>'

				if current_word == '<bos>':
					current_word = '<s>'

				# only the p( next_word | current_word ) = ... lines are needed
				self.results_f.write(u'\tp( {0} | {1} )\t= [3gram] {2} [ {3} ]\n'.format(
					next_word, current_word, prob_next_word, log_prob_next_word))

				if next_word == '</s>' or counter >= len(hypothesis)-2:
					break

			# print next prediction
			elif 'predict_next' in self.model.config:

				if next_word == '<eos>':
					self.print_predictions = True

				if self.print_predictions:

					if 'bidirectional' in self.model.config:

						for i in xrange(softmax.shape[0]):

							index_predicted_word = np.argmax(softmax[i])

						# processes hypothesis all at once: break out of while-loop
						self.results_f.write(u'\n')
						break

					else:
						index_predicted_word = np.argmax(softmax[0])

						# word dictionary
						predicted_word = self.data_object.id_to_item[index_predicted_word]

						if 'sample_multinomial' in self.model.config:
							# softmax is float32 --> cast to float64 and renormalize to make sure
							# the multinomial function doesn't give a valueerror because the probs do not sum to 1
							normalized_softmax = np.divide(softmax[0].ravel().astype(np.float64), np.sum(softmax[0].ravel().astype(np.float64)))
							# draw a sample according to the multinomial distribution
							sampled = np.argmax(np.random.multinomial(1, normalized_softmax))
							predicted_word = self.data_object.id_to_item[sampled]
							while predicted_word == self.data_object.unk:
								sampled = np.argmax(np.random.multinomial(1, normalized_softmax))
								predicted_word = self.data_object.id_to_item[sampled]


						if 'max_num_predictions' in self.model.config and self.num_predictions >= self.model.config['max_num_predictions']:
							self.results_f.write(u'\n')
							break
						elif not 'max_num_predictions' in self.model.config and self.num_predictions >= 100:
							self.results_f.write(u'\n')
							break
						elif not 'predict_mult_sentences' in self.model.config and predicted_word == '<eos>':
							self.results_f.write(u'\n')
							break
						else:
							if 'char' in self.model.config:
								if 'interactive' in self.model.config:
									if predicted_word == '<eos>':
										print('')
									else:
										print(u'{0}'.format(predicted_word).encode('utf-8'), end='')
								self.results_f.write(u'{0}'.format(predicted_word))
							else:
								if 'interactive' in self.model.config:
									if predicted_word == '<eos>':
										print('')
									else:
										print(u'{0} '.format(predicted_word).encode('utf-8'), end='')
								self.results_f.write(u'{0} '.format(predicted_word))

					self.num_predictions += 1

			# normal rescoring: print probabilities for each sentence
			else:
				# end of sentence reached: write prob of current sentence to file and stop loop
				if next_word == '<eos>' or current_word == '<eos>':
					total_log_prob += prob_next_word

					self.results_f.write(u'{0}\n'.format(total_log_prob))
					break

				else:
					#total_log_prob += np.log10(prob_next_word)
					total_log_prob += prob_next_word

			counter += 1

	def format_data(self, hypothesis, counter, predicted_word=None):
		'''
		Retrieve current input and target from the input hypothesis.
		Arguments:
			hypothesis: current input hypothesis
			counter: current position in the hypothesis
			predicted_word: if we are predicting the next word(s),
				the predicted word is added to the input
		Returns:
			x: current input
			y: current target
			hypothesis: current input hypothesis, possibly updated with predicted_word
		'''

		# add predicted word to the input
		if 'predict_next' in self.model.config and predicted_word != None:
			hypothesis = np.insert(hypothesis, counter, self.data_object.item_to_id[predicted_word])

		if 'bidirectional' in self.model.config:
			x = np.array(hypothesis[:-1]).reshape((1, self.model.num_steps))
			y = np.array(hypothesis[1:]).reshape((1, self.model.num_steps))

		else:
			x = np.array(hypothesis[counter]).reshape((1,1))
			y = np.array(hypothesis[counter+1]).reshape((1,1))


		return x, y, hypothesis

	def create_fetches(self):
		'''
		Creates a dictionary containing model variables for which we want the new values.
		'''

		fetches = {
			"final_state": self.model.final_state, # c and h of previous time step (for each hidden layer)
			"input_sample": self.model.input_sample,
			"target_sample": self.model.target_sample,
			"softmax": self.model.softmax
			}
		if 'add_word' in self.model.config:
			fetches["input_sample_words"] = self.model.input_sample_words

		# _train_op in training phase
		if self.eval_op is not None:
			fetches["eval_op"] = self.eval_op

		if self.model.use_cache:

			fetches["prob_sentence_interp_op"] = self.model.prob_sentence_interp_op
			fetches["result_interpolation_op"] = self.model.result_interpolation_op
			fetches["cache_probs_op"] = self.model.cache_probs_op
			fetches["prob_target_interp_op"] = self.model.prob_target_interp_op

			fetches_two = {
				"result_interpolation": self.model.result_interpolation,
				"all_cache_probs": self.model.all_cache_probs,
				"cache_words": self.model.cache_words,
				"softmax": self.model.softmax,
				"prob_sentence_interp": self.model.prob_sentence_interp,
				"prob_target_interp": self.model.prob_target_interp
			}
		else:
			fetches_two = None

		if 'bidirectional' in self.model.config:
			fetches["final_state_bw"] = self.model.final_state_bw

		return fetches, fetches_two

	def create_feed_dict(self, x, y, state):
		'''
		Creates a dictionary containing the data that will be fed to the placeholders of the model.
		'''

		if 'add_word' in self.model.config:
			feed_dict = {self.model.inputs: x[0], self.model.input_words: x[1], self.model.targets: y}
		elif 'per_sentence' in self.model.config:
			feed_dict = {self.model.inputs: x, self.model.targets: y, self.model.seq_length: np.array([1])}
		else:
			feed_dict = {self.model.inputs: x, self.model.targets: y}

		if 'bidirectional' in self.model.config:
			state, state_bw = state

		for i, (c, h) in enumerate(self.model.initial_state):
			feed_dict[self.model.initial_state[i]] = state[i]

			if 'bidirectional' in self.model.config:
				feed_dict[self.model.initial_state_bw[i]] = state_bw[i]

		return feed_dict

	def get_words(self, vals):
		'''
		Retrieve current input and target word.
		Argument:
			vals: result of running the graph
		Returns:
			current_word: string or list of strings of the current input word
			next_word: string or list of strings of the current target word
		'''

		# get current word: list for bidirectional model, otherwise string
		if 'bidirectional' in self.model.config:
			current_word = []
			for i in xrange(self.model.num_steps):
				current_word.append(self.data_object.id_to_item[vals['input_sample'][0][i]])

		elif self.use_cache:
			current_word = self.data_object.id_to_item[vals[0]['input_sample'][0][0]]

		else:
			current_word = self.data_object.id_to_item[vals['input_sample'][0][0]]

		# get next word: list for bidirectional model, otherwise string
		if 'bidirectional' in self.model.config:
			next_word = []
			for i in xrange(self.model.num_steps):
				next_word.append(self.data_object.id_to_item[vals['target_sample'][0][i]])

		elif self.use_cache:
			next_word = self.data_object.id_to_item[vals[0]['target_sample'][0][0]]

		else:
			next_word = self.data_object.id_to_item[vals['target_sample'][0][0]]

		return current_word, next_word

	def get_prob_next_word(self, vals, softmax, y, next_word):
		'''
		Retrieve probability that the model assigned to the target word.
		Arguments:
			softmax: softmax after feeding the current data to the network
			y: current target word
		Returns:
			prob_next_word: log probability of the current target word
		'''

		if self.use_cache:
			vals_two = vals[1]

			cache_prob_next_word = np.log(vals_two["all_cache_probs"][0][y[0][0]])

			softmax_prob_next_word = np.log(vals[0]["softmax"][0][y[0][0]])

			prob_next_word = vals[1]["result_interpolation"][0][y[0][0]]

			# if the cache is still empty, use softmax prob
			if math.isnan(prob_next_word):
				prob_next_word = softmax_prob_next_word

		else:

			prob_next_word = np.log10(softmax[0][y[0][0]])

		return prob_next_word

	def init_cache(self, hypothesis, counter):

		# counts the number of hypotheses already seen for a segment
		self.counter_hypotheses += 1

		# HACK: feed dummy variables because they will be processed once
		# by the model, so if they are the same for all hypotheses,
		# the comparison between the hypotheses is not influenced
		feed_dict = {self.model.inputs: [[0]], self.model.targets: [[0]]}

		# if we have reached a new segment
		# IMPORTANT: this assumes that there are EXACTLY num_hypotheses hypotheses per segment
		if self.counter_hypotheses % self.model.config['num_hypotheses'] == 0:

			self.new_segment_cache(feed_dict)

		# if the number of hypotheses is not the same for each segment,sentence.insert(0, self.item_to_id['<bos>']) # CHANGED
		# you should work with hypothesis ids, add 'hyp_with_ids' to the config file
		# and we will determine that we have reached a new segment once the id changes
		elif 'hyp_with_ids' in self.model.config and hypothesis[0].split('-')[0] != self.prev_id and self.prev_id != '':
			self.new_segment_cache(feed_dict)

		# end of sentence/hypothesis (but NOT end of segment) reached:
		# update memory with prob, cache words and cache states of previous hypothesis
		elif not self.first_word:
			self.session.run(self.model.update_prev_hyp_ops, feed_dict)

		self.first_word = False

		if 'hyp_with_ids' in self.model.config:
			self.prev_id = hypothesis[0].split('-')[0]

		# increment counter
		self.session.run(self.model.increment_num_hyp_op, feed_dict)

		# run to assign cache of best hypothesis from previous segment to current cache
		self.session.run(self.model.init_cache_best_prev)

		# initialize logprob for sentence to 0
		self.session.run(self.model.prob_sentence_interp.initializer)

	def new_segment_cache(self, feed_dict):

		# determine cache of best hypothesis of previous segment
		self.session.run(self.model.keep_best_prev, feed_dict)

		# re-initialize the counter for the number of hypotheses
		self.session.run(self.model.initialize_num_hyp_op, feed_dict)

		# reset counter of hypotheses
		self.counter_hypotheses = 0
