#! /usr/bin/env python

import tensorflow as tf
import os
import lm_data

class multipleLMDataChar(object):
	'''
	Class used for handling the character data in character-word language models
	(concatenation of character and word embeddings).
	This automatically calls the lm_data.LMData class for each character.
	'''

	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		self.config = config
		self.eval_config = eval_config
		self.TRAIN = TRAIN
		self.VALID = VALID
		self.TEST = TEST

		self.all_data_objects = []
		self.final_vocab_size = 0

		if config['order'] == 'begin_first':
			all_train_data, all_valid_data, all_test_data = self.get_chars(
					"char_", config['num_char'])
		elif config['order'] == 'end_first':
			all_train_data, all_valid_data, all_test_data = self.get_chars(
					"char-invert_", config['num_char'])
		elif config['order'] == 'both':
			all_train_data, all_valid_data, all_test_data = self.get_chars(
					"char_", config['num_char'])
			all_train_data_end, all_valid_data_end, all_test_data_end = self.get_chars(
					"char-invert_", config['num_char'], second_time=True)

			all_train_data.extend(all_train_data_end)
			all_valid_data.extend(all_valid_data_end)
			all_test_data.extend(all_test_data_end)

			print('len all_train_data: {0}'.format(len(all_train_data)))

		else:
			raise ValueError("Specify in which order the characters should be added: starting from the beginning of the "
				"word (begin_first), from the end of the word (end_first) or an equal number from the beginning "
				"and the end of the word.")

		self.all_data = (all_train_data, all_valid_data, all_test_data)

	def get_chars(self, prefix, num_char, second_time=False):
		'''
		This function reads the characters from file.
		Inputs:
			prefix: "char_" if the characters are used starting from the beginning of the word,
					"char-invert_" if the characters are used starting from the end of the word
			num_char: the number of characters that should be read
			second_time: for adding characters in both order, second_time=True if this function is called for the second time
				(to ensure that the same character vocabulary is being used)
		Returns:
			all_train_data, all_valid_data, all_test_data
		'''
		all_train_data = []
		all_valid_data = []
		all_test_data = []

		for pos in range(num_char):

			# characters on position x should be in <data_path>/features/char/<prefix>_<type_of_data>_<x>
			# use aux_scripts/make_char_feat_files.py to generate the character files
			char_train_file = os.path.join(self.config['data_path'], 'features/char/{0}train_{1}'.format(prefix,pos))
			char_valid_file = os.path.join(self.config['data_path'], 'features/char/{0}valid_{1}'.format(prefix,pos))
			char_test_file = os.path.join(self.config['data_path'], 'features/char/{0}test_{1}'.format(prefix,pos))

			# use LMData and not charData because characters are already separated by space
			curr_char_obj = lm_data.LMData(self.config, self.eval_config, self.TRAIN, self.VALID, self.TEST)
			self.all_data_objects.append(curr_char_obj)

			curr_char_obj.train_path = char_train_file
			curr_char_obj.valid_path = char_valid_file
			curr_char_obj.test_path = char_test_file
			curr_char_obj.encoding = "latin-1"

			if pos > 0 or second_time:
				# not the first character: re-use vocabulary mapping from previous characters
				curr_char_obj.item_to_id = self.char_to_id
				curr_char_obj.id_to_item = self.id_to_char

			(curr_char_train_data, curr_char_valid_data, curr_char_test_data), vocab_size, _ = curr_char_obj.get_data()

			all_train_data.append(curr_char_train_data)
			all_valid_data.append(curr_char_valid_data)
			all_test_data.append(curr_char_test_data)
			if vocab_size > self.final_vocab_size:
				self.final_vocab_size = vocab_size

			# keep vocabulary mapping for next characters
			self.char_to_id = curr_char_obj.item_to_id
			self.id_to_char = curr_char_obj.id_to_item

		return all_train_data, all_valid_data, all_test_data

	def get_data(self):
		return self.all_data, self.final_vocab_size

	def init_batching(self, all_char_data, test=False):
		if self.config['order'] == 'both':
			self.total_num_char = 2 * self.config['num_char']
		else:
			self.total_num_char = self.config['num_char']

		for pos in range(self.total_num_char):
			self.all_data_objects[pos].init_batching(all_char_data[pos], test=test)

	def get_batch(self):

		all_x = []
		for pos in range(self.total_num_char):

			x, _, _ = self.all_data_objects[pos].get_batch()
			all_x.append(x)

		return all_x

