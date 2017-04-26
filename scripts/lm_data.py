#! /usr/bin/env python

import tensorflow as tf
import os, collections
from abc import abstractmethod
import numpy as np

class LMData(object):
	'''
	Class for word-level data going across sentence boundaries.
	'''
	def __init__(self, config, eval_config):

		self.config = config
		self.eval_config = eval_config
	
		
		if config['vocab']:
			# data file in which words not in vocabulary are converted to UNK
			# e.g. vocabulary of 50k words: new data file 'train_50k-unk.txt'
			train_file = "train_" + str(config['vocab']) + "k-unk.txt"
			valid_file = "valid_" + str(config['vocab']) + "k-unk.txt"
			test_file = "test_" + str(config['vocab']) + "k-unk.txt"
			self.train_path = os.path.join(config['data_path'], train_file)
			self.valid_path = os.path.join(config['data_path'], valid_file)
			self.test_path = os.path.join(config['data_path'], test_file)
		else: 
			self.train_path = os.path.join(config['data_path'], "train.txt")
			self.valid_path = os.path.join(config['data_path'], "valid.txt")
			self.test_path = os.path.join(config['data_path'], "test.txt")

		self.batch_size = config['batch_size']
		self.num_steps = config['num_steps']

		self.eval_batch_size = eval_config['batch_size']
		self.eval_num_steps = eval_config['num_steps']

		self.iterator = 0
		self.end_reached = False

		# if your data has another encoding than utf-8, specify it in the config file
		if 'encoding' in self.config:
			self.encoding = self.config['encoding']
		# default encoding = utf-8
		else:
			self.encoding = "utf-8"

		self.id_to_item = {}
		self.item_to_id = {}

		# this is specific to your data: here we assume that all unknown words should be mapped to
		# lowercase <unk>, and if uppercase <UNK> occurs in the data then it will be mappend to lowercase <unk>
		self.unk = '<unk>'
		self.replace_unk = '<UNK>'
			
		# rescoring: test set = rescore set
		if 'rescore' in self.config:
			self.test_path = self.config['rescore']
			
		# if you want to get the ppl of the validation set only (with --train False and --valid False)
		if 'valid_as_test' in self.config:
			self.test_path = self.valid_path
		
		self.PADDING_SYMBOL = '@'

	def read_items(self, filename):
		'''Returns a list of all WORDS in filename.'''
		with tf.gfile.GFile(filename, "r") as f:
			# Wikitext: more than 1 sentence per line, also introduce <eos> at ' . '
			if self.config['data_path'].startswith('/users/spraak/lverwimp/data/WikiText/'):
				data = f.read().decode(self.encoding).replace("\n", " <eos> ").replace(" . "," <eos> ").split()
			else:
				data = f.read().decode(self.encoding).replace("\n", " <eos> ").split()

			# make sure there is only 1 symbol for unknown words
			data = [self.unk if word==self.replace_unk else word for word in data]

			return data

	@abstractmethod
	def calc_longest_sent(self, all_data):
		raise NotImplementedError("Abstract class.")
	
	@abstractmethod
	def padding(self, dataset, total_length):
		raise NotImplementedError("Abstract class.")

	@abstractmethod
	def pad_data(self, all_data, max_length):
		raise NotImplementedError("Abstract class.")

	def build_vocab(self, filename):
		'''Returns a word-to-id and id-to-word (or character-to-id and id-to-character) mapping 
		for all words (or characters) in filename.'''

		data = self.read_items(filename)

		counter = collections.Counter(data)

		# counter.items() = list of the words in data + their frequencies, then sorted according to decreasing frequency
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	
		# words = list of all the words (in decreasing frequency)
		items, _ = list(zip(*count_pairs))

		# make a dictionary with a mapping from each word to an id; word with highest frequency gets lowest id etc.
		self.item_to_id = dict(zip(items, range(len(items))))

		# reverse dictionary
		self.id_to_item = dict(zip(range(len(items)), items))

		# make sure there is a special token for unknown words
		if not self.unk in self.item_to_id:
			self.item_to_id[self.unk] = len(self.item_to_id)
			self.id_to_item[len(self.id_to_item)] = self.unk
	
		if 'per_sentence' in self.config:
			if self.PADDING_SYMBOL not in self.item_to_id:
				self.item_to_id[self.PADDING_SYMBOL] = len(self.item_to_id)
				self.id_to_item[len(self.id_to_item)] = self.PADDING_SYMBOL
			else:
				raise ValueError("{0} used as padding symbol but occurs in text.".format(self.PADDING_SYMBOL))
						

	def file_to_item_ids(self, filename):
		'''Returns list of all words/characters (mapped to their ids) in the file, either one long list or a list of lists per sentence.'''
		data = self.read_items(filename)
		return [self.item_to_id[item] for item in data if item in self.item_to_id]

	def read_data(self):
		# if the vocabulary mapping is not pre-initialized, make one based on the training data
		if not self.item_to_id:
			self.build_vocab(self.train_path)
		else:
			# check whether the data file contains words that are not yet in the vocabulary mapping
			self.extend_vocab(self.train_path)

		# list of all words in training data converted to their ids
		train_data = self.file_to_item_ids(self.train_path)

		# list of all words in validation data converted to their ids
		valid_data = self.file_to_item_ids(self.valid_path)

		# list of all words in test data converted to their ids
		test_data = self.file_to_item_ids(self.test_path)

		all_data = (train_data, valid_data, test_data)

		return all_data

	def get_data(self):
		all_data = self.read_data()
		return all_data, len(self.id_to_item), 0

	def get_batch(self, data, test=False):
	
		if test:
			batch_size = self.eval_batch_size
			num_steps = self.eval_num_steps
		else:
			batch_size = self.batch_size
			num_steps = self.num_steps

		# beginning of data set: set self.end_reached to False (was set to True if another data set is already processed)
		if self.iterator == 0:
			self.end_reached = False

		data_len = len(data)

		# to divide data in batch_size batches, each of length batch_len
		batch_len = data_len // batch_size 

		# number of samples that can be taken from the batch_len slices
		if num_steps != 1:
			num_samples = batch_len // num_steps
		else:
			num_samples = (batch_len // num_steps) - 1

		# remove last part of the data that doesn't fit in the batch_size x num_steps samples
		data = data[:batch_size * batch_len]

		# convert to numpy array: batch_size x batch_len
		data_array = np.array(data).reshape(batch_size, batch_len)

		# take slice of batch_size x num_steps
		x = data_array[:, self.iterator * num_steps : (self.iterator * num_steps) + num_steps]
		# targets = same slice but shifted one step to the right
		y = data_array[:, (self.iterator * num_steps) +1 : (self.iterator * num_steps) + num_steps + 1]

		# if iterated over the whole dataset, set iterator to 0 to start again
		if self.iterator >= num_samples:
			self.iterator = 0
			self.end_reached = True
		# otherwise, increase count
		else:
			self.iterator += 1
	
		return x, y, self.end_reached
		

class charData(LMData):
	'''
	Train on character level rather than word level.
	'''
	def read_items(self, filename):
		'''Returns a list of all CHARACTERS in filename.'''
		with tf.gfile.GFile(filename, "r") as f:
			# Wikitext: more than 1 sentence per line, also introduce <eos> at ' . '
			if self.config['data_path'].startswith('/users/spraak/lverwimp/data/WikiText/'):
				# TO DO: change this such that ' . ' is also replaced by <eos>
				data = ['<eos>' if x == '\n' else x for x in f.read().decode(self.encoding)]
			else:
				data = ['<eos>' if x == '\n' else x for x in f.read().decode(self.encoding)]
			return data # single list with all characters in the file


class wordSentenceData(LMData):
	'''
	Feed sentence per sentence to the network, 
	each sentence padded until the length of the longest sentence.
	'''
	def __init__(self, config, eval_config):

		super(wordSentenceData, self).__init__(config, eval_config)

	def read_sentences(self, filename):
		'''
		Returns a list with all sentences in filename, each sentence is split in words.
		'''
		with tf.gfile.GFile(filename, "r") as f:
			if self.config['data_path'].startswith('/users/spraak/lverwimp/data/WikiText/'):
				all_sentences = f.read().decode(self.encoding).replace("\n", "<eos>").replace(" . "," <eos> ").split("<eos>")
			else:
				all_sentences = f.read().decode(self.encoding).replace("\n", "<eos>").split("<eos>")
			# remove empty element at the end
			if all_sentences[-1] == '':
				all_sentences = all_sentences[:-1]
			# split sentence in words
			for i in xrange(len(all_sentences)):
				all_sentences[i] = all_sentences[i].split()

			return all_sentences

	def build_vocab(self, filename):
		'''Returns a word-to-id and id-to-word (or character-to-id and id-to-character) mapping 
		for all words (or characters) in filename.'''

		data = self.read_items(filename)

		counter = collections.Counter(data)

		# counter.items() = list of the words in data + their frequencies, then sorted according to decreasing frequency
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	
		# words = list of all the words (in decreasing frequency)
		words, _ = list(zip(*count_pairs))

		# make a dictionary with a mapping from each word to an id; word with highest frequency gets lowest id etc.
		self.item_to_id = dict(zip(words, range(len(words))))

		# reverse dictionary
		self.id_to_item = dict(zip(range(len(words)), words))

		# make sure there is a special token for unknown words
		if not self.unk in self.item_to_id:
			self.item_to_id[self.unk] = len(self.item_to_id)
			self.id_to_item[len(self.id_to_item)] = self.unk

		# if processing per sentence: add beginning of sentence symbol + padding symbol
		self.item_to_id['<bos>'] = len(self.item_to_id)
		self.id_to_item[len(self.id_to_item)] = '<bos>'

		if self.PADDING_SYMBOL not in self.item_to_id:
			self.item_to_id[self.PADDING_SYMBOL] = len(self.item_to_id)
			self.id_to_item[len(self.id_to_item)] = self.PADDING_SYMBOL
		else:
			raise ValueError("{0} used as padding symbol but occurs in text.".format(self.PADDING_SYMBOL))


	def calc_longest_sent(self, all_data):
		'''Returns length of longest sentence occurring in all_data.'''
		max_length = 0
		for dataset in all_data:
			for sentence in dataset:
				if len(sentence) > max_length:
					max_length = len(sentence)
		return max_length

	def padding(self, dataset, total_length):
		'''Add <bos> and <eos> to each sentence in dataset + pad until max_length.'''
		seq_lengths = []
		for sentence in dataset:
			seq_lengths.append(len(sentence)+2) # +2 for <bos> and <eos>
			# beginning of sentence symbol
			sentence.insert(0, self.item_to_id['<bos>'])
			# end of sentence symbol
			sentence.append(self.item_to_id['<eos>'])
			# pad rest of sentence until maximum length
			num_pads = total_length - len(sentence)
			for pos in xrange(num_pads):
				sentence.append(self.item_to_id[self.PADDING_SYMBOL])
		return dataset, seq_lengths

	def pad_data(self, all_data, max_length):
		'''Pad each dataset in all_data.'''
				
		# <bos> and <eos> should be added 
		# + 1 extra padding symbol to avoid having target sequences which end on the beginning of the next sentence
		total_length = max_length + 3
	 
		if isinstance(all_data, tuple):
			padded_all = ()
			seq_lengths_all = ()
			for dataset in all_data:
				padded_dataset, seq_length = self.padding(dataset, total_length)
				padded_all += (padded_dataset,)
				seq_lengths_all += (seq_length,)
		else:
			padded_all, seq_lengths_all = self.padding(all_data, total_length)

		return padded_all, seq_lengths_all

	def file_to_item_ids(self, filename):
		data = self.read_sentences(filename)
		data_ids = []
		for sentence in data: 
			data_ids.append([self.item_to_id[item] for item in sentence if item in self.item_to_id])
		return data_ids

	def get_data(self):
		#all_data, id_to_item, item_to_id = self.read_data()
		all_data = self.read_data()

		max_length = self.calc_longest_sent(all_data)
		self.num_steps = max_length + 3
		padded_data, seq_lengths = self.pad_data(all_data, max_length)

		# return max_length+2 and not +3 because the last padding symbol is only there 
		# to make sure that the target sequence does not end with the beginning of the next sequence
		return padded_data, len(self.id_to_item), max_length+2, seq_lengths

	def get_batch(self, data, test=False):
		if test:
			batch_size = self.eval_batch_size
			num_steps = self.eval_num_steps
		else:
			batch_size = self.batch_size
			num_steps = self.num_steps
		length_sentence = self.num_steps
		
		if self.iterator == 0:
			self.end_reached = False
		
		words = data[0]
		seq_lengths = data[1]
		
		if not test:

			data_len = len(words)*len(words[0])

			# to divide data in batch_size batches, each of length batch_len
			batch_len = data_len // batch_size 

			# number of sentences that fit in 1 batch_len
			num_sentences_batch = batch_len // (length_sentence+1)
		
			# we want batch_len to be a multiple of num_steps (=size of padded sentence)
			batch_len = num_sentences_batch * (length_sentence+1)

			# remove last part of the data that doesn't fit in the batch_size x num_steps samples
			words = words[:batch_size * num_sentences_batch]

			# convert to numpy array: batch_size x batch_len*num_steps
			data_array = np.array(words).reshape(batch_size, num_sentences_batch*length_sentence)

			# take slice of batch_size x num_steps
			x = data_array[:, self.iterator * num_steps : (self.iterator * num_steps) + num_steps - 1]
			# targets = same slice but shifted one step to the right
			y = data_array[:, (self.iterator * num_steps) +1 : (self.iterator * num_steps) + num_steps ]
			
			# convert seq_lengths to numpy array
			seql_array = np.array(seq_lengths)
			
			# take slice of sequence lengths for all elements in the batch
			seql = seql_array[self.iterator * batch_size : (self.iterator+1) * batch_size]

			# if iterated over the whole dataset, set iterator to 0 to start again
			if self.iterator >= num_sentences_batch:
				self.iterator = 0
				self.end_reached = True
			# otherwise, increase count
			else:
				self.iterator += 1

		else:
			# only for testing, this assumes that batch_size and num_steps are 1!

			len_data = len(words)*len(words[0])
			data_array = np.array(words).reshape(len_data)

			x = data_array[self.iterator: self.iterator + 1]
			y = data_array[self.iterator + 1 : self.iterator + 2]
			
			# num_steps = 1 so no sequence length needed
			seql = [1]

			if self.iterator >= len_data-1:
				self.iterator = 0
				self.end_reached = True
			# otherwise, increase count
			else:
				self.iterator += 1

			x = [x]
			y = [y]

		return x, y, self.end_reached, seql


class wordSentenceDataRescore(wordSentenceData):
	'''
	Rescore N-best lists with model trained cross-sentence boundaries.
	'''

	def __init__(self, config, eval_config):

		super(wordSentenceDataRescore, self).__init__(config, eval_config)

	def file_to_item_ids(self, filename):
		data = self.read_sentences(filename)
		data_ids = []
		for sentence in data: 
			# difference: words not in vocabulary are mapped to the unk symbol
			data_ids.append([self.item_to_id[item] if item in self.item_to_id else self.item_to_id[self.unk] for item in sentence])
		return data_ids

	def read_data(self):
		self.build_vocab(self.train_path)

		# only read test data
		# list of all words in test data converted to their ids
		test_data = self.file_to_item_ids(self.test_path)

		all_data = test_data

		return all_data

	def get_data(self):
		all_data = self.read_data()

		max_length = self.config['num_steps'] - 3
		padded_data, _ = self.pad_data(all_data, max_length)

		# return max_length+2 and not +3 because the last padding symbol is only there 
		# to make sure that the target sequence does not end with the beginning of the next sequence
		return padded_data, len(self.id_to_item), max_length+2


