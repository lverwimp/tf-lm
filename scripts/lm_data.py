#! /usr/bin/env python

from __future__ import print_function

import tensorflow as tf
import os, collections, sys, subprocess, codecs
from abc import abstractmethod
import numpy as np

def flattern(A):
	'''
	Flatten a list containing a combination of strings and lists.
	Copied from https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python.
	'''
	rt = []
	for i in A:
		if isinstance(i,list): rt.extend(flattern(i))
		else: rt.append(i)
	return rt

def save_item_to_id(item_to_id, file, encoding):
	'''
	Saves a item_to_id mapping to file.
	'''

	out = codecs.open(file, 'w', encoding)
	for item, id_ in item_to_id.iteritems():
		if item == '':
			print('EMPTY ELEMENT')
		if item == ' ':
			print('SPACE')
		out.write(u'{0}\t{1}\n'.format(item, id_).encode('utf-8'))
	out.close()

def load_item_to_id(file, encoding):
	'''
	Loads an item_to_id mapping and corresponding id_to_item mapping from file.
	'''

	item_to_id = {}
	id_to_item = {}

	for line in codecs.open(file, 'r', encoding):
		l = line.strip().split()
		item_to_id[l[0]] = l[1]
		id_to_item[l[1]] = l[0]

	return item_to_id, id_to_item

class LMData(object):
	'''
	The input data: words, batches across sentence boundaries.
	'''

	def __init__(self, config, eval_config, TRAIN, VALID, TEST):
		'''
		Arguments:
			config: configuration dictionary, specifying all parameters used for training
			eval_config: configuration dictionary, specifying all parameters used for testing
			TRAIN: boolean indicating whether we want to train or not
			VALID: boolean indicating whether we want to validate or not
			TEST: boolean indicating whether we want to test or not
		'''

		self.config = config
		self.eval_config = eval_config
		self.TRAIN = TRAIN
		self.VALID = VALID
		self.TEST = TEST

		# if we want to train with a limited vocabulary, words not in the vocabulary
		# should already be mapped to UNK
		# data files should be of format train_50k-unk.txt etc. for a 50k vocabulary
		if config['vocab']:
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

		# default encoding = utf-8, specify in config file if otherwise
		if 'encoding' in self.config:
			self.encoding = self.config['encoding']
		else:
			self.encoding = "utf-8"

		self.id_to_item = {}
		self.item_to_id = {}

		# by default, unknown words are represented with <unk>
		# if this is not the case for a certain dataset, add it here
		if self.config['data_path'].startswith('/users/spraak/lverwimp/data/CGN') or \
				self.config['data_path'].startswith('/users/spraak/lverwimp/data/WSJ/88') or\
				self.config['data_path'].startswith('/users/spraak/lverwimp/data/MGB_Arabic'):
			self.unk = '<UNK>'
			self.replace_unk = '<unk>'
		else:
			self.unk = '<unk>'
			self.replace_unk = '<UNK>'

		if 'rescore' in self.config and isinstance(self.config['rescore'], str):
			self.test_path = self.config['rescore']
		elif 'predict_next' in self.config and isinstance(self.config['predict_next'], str):
			self.test_path = self.config['predict_next']
		elif 'debug2' in self.config and isinstance(self.config['debug2'], str):
			self.test_path = self.config['debug2']
		elif 'other_test' in self.config:
			self.test_path = self.config['other_test']
		if 'valid_as_test' in self.config:
			self.test_path = self.valid_path

		self.PADDING_SYMBOL = '@'

	def read_items(self, filename):
		'''
		Returns a list of all WORDS in filename.
		'''

		with tf.gfile.GFile(filename, "r") as f:
			# Wikitext: more than 1 sentence per line, also introduce <eos> at ' . '
			# add here other datasets that contain more than 1 sentence per line
			if "WikiText" in self.config['data_path']:
				data = f.read().decode(self.encoding).replace("\n", " <eos> ").replace(" . "," <eos> ").split()
			elif 'no_eos' in self.config:
				data = f.read().decode(self.encoding).replace("\n", " ").split()
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
		'''
		Returns an item-to-id and id-to-item mapping for all words (or characters) in filename.
		Arguments:
			filename: name of file for which the mapping will be built
		Returns:
			item_to_id mapping and id_to_item mapping
		'''

		data = self.read_items(filename)

		counter = collections.Counter(data)

		# counter.items() = list of the words in data + their frequencies, then sorted according to decreasing frequency
		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		# words = list of all the words (in decreasing frequency)
		items, _ = list(zip(*count_pairs))

		# make a dictionary with a mapping from each word to an id; word with highest frequency gets lowest id etc.
		item_to_id = dict(zip(items, range(len(items))))

		# remove empty element and space
		if '' in item_to_id:
			item_to_id.pop('')
		if ' ' in item_to_id and not 'char' in self.config:
			item_to_id.pop(' ')

		# reverse dictionary
		id_to_item = dict(zip(range(len(items)), items))

		# make sure there is a special token for unknown words
		if not self.unk in item_to_id:
			item_to_id[self.unk] = len(item_to_id)
			id_to_item[len(id_to_item)] = self.unk

		# add <bos>: used for sentence-level batches, or
		# for discourse-level models that are use for e.g. rescoring
		item_to_id['<bos>'] = len(item_to_id)
		id_to_item[len(id_to_item)] = '<bos>'

		return item_to_id, id_to_item


	def extend_vocab(self, filename):
		'''
		If there already is a vocabulary, this function extends the vocabulary with words
		found in the data file 'filename'.
		'''

		data = self.read_items(filename)
		vocab_curr = set(data)

		for word in vocab_curr:
			if word not in self.item_to_id:
				print(u'word {0} not yet seen'.format(word).encode(self.encoding))
				self.item_to_id[word] = len(self.item_to_id)
				self.id_to_item[len(self.id_to_item)] = word

	def add_padding_symbol(self):
		'''
		Add the correct padding symbol to the vocabulary
		'''

		if self.PADDING_SYMBOL not in self.item_to_id:
			self.item_to_id[self.PADDING_SYMBOL] = len(self.item_to_id)
			self.id_to_item[len(self.id_to_item)] = self.PADDING_SYMBOL

		# if the default symbol for padding is already in the vocabulary
		else:

			# another symbol should be specified in the config file
			if not 'padding_symbol' in self.config:
				raise ValueError("{0} used as padding symbol but occurs in text. " \
					"Specify another padding symbol with 'padding_symbol' in the config file.".format(
					self.PADDING_SYMBOL))

			else:
				self.PADDING_SYMBOL = self.config['padding_symbol']

				# check whether the padding symbol specified in the config file occurs in the data or not
				if self.PADDING_SYMBOL not in self.item_to_id:
					self.item_to_id[self.PADDING_SYMBOL] = len(self.item_to_id)
					self.id_to_item[len(self.id_to_item)] = self.PADDING_SYMBOL
				else:
					raise ValueError("The padding symbol specified in the config file ({0}) " \
						"already occurs in the text.".format(self.PADDING_SYMBOL))

	@abstractmethod
	def build_ngram_vocab(self, filename):
		raise NotImplementedError("Abstract class.")

	@abstractmethod
	def build_skipgram_vocab(self, filename, skip):
		raise NotImplementedError("Abstract class.")

	def file_to_item_ids(self, filename, item_to_id=None):
		'''
		Returns list of all words/characters (mapped to their ids) in the file,
		either one long list or a list of lists per sentence.
		Arguments:
			filename: name of file for which the words should be mapped to their ids
			Optional:
			item_to_id: dictionary that should be used for the mapping (otherwise self.item_to_id is used)
		'''

		if item_to_id == None:
			item_to_id = self.item_to_id

		data = self.read_items(filename)
		tmp_l = []
		for w in data:
			if w in item_to_id:
				tmp_l.append(item_to_id[w])
			else:
				print(u'{0} not in item_to_id'.format(w).encode('utf-8'))

		return [item_to_id[item] if item in item_to_id else item_to_id[self.unk] for item in data]

	@abstractmethod
	def file_to_ngram_ids(self, filename):
		raise NotImplementedError("Abstract class.")

	@abstractmethod
	def file_to_skipgram_ids(self, filename):
		raise NotImplementedError("Abstract class.")

	def read_data(self):
		'''
		Makes sure there is a vocabulary and reads all necessary data.
		Returns:
			all_data: tuple of three lists : train_data, valid_data and test_data
		'''

		if 'read_vocab_from_file' in self.config:
			# read vocabulary mapping from file
			self.item_to_id, self.id_to_item = load_item_to_id(self.config['read_vocab_from_file'], self.encoding)

			# check whether the data file contains words that are not yet in the vocabulary mapping
			self.extend_vocab(self.train_path)

			if len(self.item_to_id) != self.config['vocab_size']:
				raise IOError("The vocabulary size specified by 'vocab_size' ({0}) does not correspond \
						to the size of the vocabulary file given ({1}).".format(
						self.config['vocab_size'], len(self.item_to_id)))

		else:
			# if the vocabulary mapping is not saved on disk, make one based on the training data
			self.item_to_id, self.id_to_item = self.build_vocab(self.train_path)

			# sentence-level model or model that will be used for rescoring: needs padding symbol in vocabulary
			if 'rescore_later' in self.config or 'per_sentence' in self.config:
				self.add_padding_symbol()

			# save the item_to_id mapping such that it can be re-used
			if 'save_dict' in self.config:
				save_item_to_id(self.item_to_id, '{0}.dict'.format(self.config['save_dict']), self.encoding)


		# list of all words in training data converted to their ids
		if self.TRAIN:
			train_data = self.file_to_item_ids(self.train_path)
		else:
			train_data = []

		# list of all words in validation data converted to their ids
		if self.VALID:
			valid_data = self.file_to_item_ids(self.valid_path)
		else:
			valid_data = []

		# list of all words in test data converted to their ids
		if self.TEST:
			test_data = self.file_to_item_ids(self.test_path)
		else:
			test_data = []

		all_data = (train_data, valid_data, test_data)

		return all_data

	def get_data(self):
		'''
		Retrieve the necessary data and vocabulary size.
		'''
		all_data = self.read_data()
		return all_data, len(self.id_to_item), 0

	def init_batching(self, data, test=False):
		'''
		Prepare for batching.
		'''

		if test:
			batch_size = self.eval_batch_size
			self.num_steps = self.eval_num_steps
		else:
			batch_size = self.batch_size

		# beginning of data set: set self.end_reached to False (was set to True if another data set is already processed)
		if self.iterator == 0:
			self.end_reached = False

		data_len = len(data)

		# to divide data in batch_size batches, each of length batch_len
		batch_len = data_len // batch_size

		# number of samples that can be taken from the batch_len slices
		self.num_samples = (batch_len // self.num_steps) - 1

		# remove last part of the data that doesn't fit in the batch_size x num_steps samples
		data = data[:batch_size * batch_len]

		# convert to numpy array: batch_size x batch_len
		self.data_array = np.array(data).reshape(batch_size, batch_len)

	def get_batch(self):
		'''
		Gets a single batch.
		Returns:
			x: input data
			y: target data
			end_reached: boolean marking whether the end of the data file has been reached or not
		'''

		# take slice of batch_size x num_steps
		x = self.data_array[:, self.iterator * self.num_steps :
					(self.iterator * self.num_steps) + self.num_steps]
		# targets = same slice but shifted one step to the right
		y = self.data_array[:, (self.iterator * self.num_steps) +1 :
					(self.iterator * self.num_steps) + self.num_steps + 1]

		# if iterated over the whole dataset, set iterator to 0 to start again
		if self.iterator >= self.num_samples:
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
	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(charData, self).__init__(config, eval_config, TRAIN, VALID, TEST)

	def read_items(self, filename):
		'''
		Returns a list of all CHARACTERS in filename.
		'''
		with tf.gfile.GFile(filename, "r") as f:
			data = ['<eos>' if x == '\n' else x for x in f.read().decode(self.encoding)]
			return data


class wordSentenceData(LMData):
	'''
	Feed sentence per sentence to the network,
	each sentence padded until the length of the longest sentence.
	'''
	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(wordSentenceData, self).__init__(config, eval_config, TRAIN, VALID, TEST)

		self.sentence_iterator = 0

	def read_sentences(self, filename):
		'''
		Returns a list with all sentences in filename, each sentence is split in words.
		'''

		with tf.gfile.GFile(filename, "r") as f:
			if "WikiText" in self.config['data_path']:
				all_sentences = f.read().decode(self.encoding).replace("\n", "<eos>").replace(" . "," <eos> ").split("<eos>")
			# this assumes that all other datasets contain 1 sentence per line
			else:
				all_sentences = f.read().decode(self.encoding).replace("\n", "<eos>").split("<eos>")

			# remove empty element at the end
			if all_sentences[-1] == '':
				all_sentences = all_sentences[:-1]

			# split sentence in words
			for i in xrange(len(all_sentences)):
				all_sentences[i] = all_sentences[i].split()

			return all_sentences


	def calc_longest_sent(self, all_data):
		'''
		Returns length of longest sentence occurring in all_data.
		'''

		max_length = 0
		for dataset in all_data:
			for sentence in dataset:
				if len(sentence) > max_length:
					max_length = len(sentence)

		return max_length

	def padding(self, dataset, total_length):
		'''
		Add <bos> and <eos> to each sentence in dataset + pad until max_length.
		'''

		seq_lengths = []
		for sentence in dataset:
			#seq_lengths.append(len(sentence)+1) # +1 ONLY <eos>
			seq_lengths.append(len(sentence)+2) # +2 <bos> + <eos>
			sentence.insert(0, self.item_to_id['<bos>']) # CHANGED
			# end of sentence symbol
			sentence.append(self.item_to_id['<eos>'])
			# pad rest of sentence until maximum length
			num_pads = total_length - len(sentence)
			for pos in xrange(num_pads):
				if 'not_trained_with_padding' in self.config:
					sentence.append(self.item_to_id[self.unk])
				else:
					try:
						sentence.append(self.item_to_id[self.PADDING_SYMBOL])
					except KeyError:
						print("No padding symbol ({0}) in the dictionary. Either add 'not_trained_with_padding' " \
							"in the config file if the model is trained without padding or " \
							"specify the correct symbol with 'padding_symbol' in the config.".format(
							self.PADDING_SYMBOL))
						sys.exit(1)
		return dataset, seq_lengths

	def pad_data(self, all_data, max_length):
		'''
		Pad each dataset in all_data.
		'''

		# + 2 because <bos> and <eos> should be added
		# + 1 for extra padding symbol to avoid having target sequences
		# which end on the beginning of the next sentence
		#total_length = max_length + 2
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

		all_data = self.read_data()

		self.item_to_id['<bos>'] = len(self.item_to_id)
		self.id_to_item[len(self.id_to_item)] = '<bos>'

		max_length = self.calc_longest_sent(all_data)

		# + 2 for <eos> and extra padding symbol at the end
		#self.num_steps = max_length + 2
		self.num_steps = max_length + 3
		padded_data, seq_lengths = self.pad_data(all_data, max_length)

		# return max_length+1 and not +2 because the last padding symbol is only there
		# to make sure that the target sequence does not end with the beginning of the next sequence
		#return padded_data, len(self.id_to_item), max_length+1, seq_lengths
		return padded_data, len(self.id_to_item), max_length+2, seq_lengths

	def init_batching(self, data, test=False):
		if test:
			self.batch_size = self.eval_batch_size
			self.num_steps = self.eval_num_steps

		length_sentence = self.num_steps

		print('self.num_steps: {0}'.format(self.num_steps))

		if self.iterator == 0:
			self.end_reached = False

		self.test = test

		words = data[0]
		seq_lengths = data[1]

		if not self.test:

			data_len = len(words)*len(words[0])

			# to divide data in batch_size batches, each of length batch_len
			batch_len = data_len // self.batch_size

			# number of sentences that fit in 1 batch_len
			self.num_sentences_batch = batch_len // (length_sentence+1)

			# we want batch_len to be a multiple of num_steps (=size of padded sentence)
			batch_len = self.num_sentences_batch * (length_sentence+1)

			# remove last part of the data that doesn't fit in the batch_size x num_steps samples
			words = words[:self.batch_size * self.num_sentences_batch]

			# convert to numpy array: batch_size x batch_len*num_steps
			self.data_array = np.array(words).reshape(
				self.batch_size, self.num_sentences_batch*length_sentence)

			# convert seq_lengths to numpy array
			self.seql_array = np.array(seq_lengths)

		else:
			# only for testing, this assumes that batch_size and num_steps are 1!

			self.len_data = len(words)*len(words[0])
			self.len_sentence = len(words[0])
			self.data_array = np.array(words).reshape(len(words), len(words[0]))


	def get_batch(self):

		if not self.test:

			# take slice of batch_size x num_steps
			x = self.data_array[:, self.iterator * self.num_steps :
						(self.iterator * self.num_steps) + self.num_steps - 1]
			# targets = same slice but shifted one step to the right
			y = self.data_array[:, (self.iterator * self.num_steps) +1 :
						(self.iterator * self.num_steps) + self.num_steps ]

			# take slice of sequence lengths for all elements in the batch
			seql = self.seql_array[self.iterator * self.batch_size : (self.iterator+1) * self.batch_size]

			# if iterated over the whole dataset, set iterator to 0 to start again
			if self.iterator >= self.num_sentences_batch:
				self.iterator = 0
				self.end_reached = True
			# otherwise, increase count
			else:
				self.iterator += 1

		else:

			x = self.data_array[self.sentence_iterator, self.iterator: self.iterator + 1]
			y = self.data_array[self.sentence_iterator, self.iterator + 1 : self.iterator + 2]

			# num_steps = 1 so no sequence length needed
			seql = [1]

			if self.sentence_iterator == self.len_data / self.len_sentence and self.iterator == self.len_sentence - 1:
				self.end_reached = True

			# otherwise, increase count
			else:
				self.iterator += 1

			if self.iterator == self.len_sentence - 1:
				# end of file reached
				if self.sentence_iterator >= (self.len_data / self.len_sentence) - 1:
					self.end_reached = True
				# end of sentence reached
				else:
					self.iterator = 0
					self.sentence_iterator += 1

			x = [x]
			y = [y]

		return x, y, self.end_reached, seql

class charSentenceData(wordSentenceData):
	'''
	Same as wordSentenceData, except that the input unit is a character.
	'''
	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(charSentenceData, self).__init__(config, eval_config, TRAIN, VALID, TEST)

	def read_sentences(self, filename):
		'''Returns a list with all sentences in filename, each sentence is split in words.'''
		with tf.gfile.GFile(filename, "r") as f:
			if "WikiText" in self.config['data_path']:
				all_sentences = [x for x in f.read().decode(self.encoding).replace("\n", "<eos>").replace(
					" . "," <eos> ").split("<eos>")]
			else:
				all_sentences = f.read().decode(self.encoding).replace("\n", "<eos>").split("<eos>")
			# remove empty element at the end
			if all_sentences[-1] == '':
				all_sentences = all_sentences[:-1]
			# split sentence in words
			for i in xrange(len(all_sentences)):
				all_sentences[i] = [x for x in all_sentences[i]]

			return all_sentences

	def read_items(self, filename):
		'''
		Returns a list of all CHARACTERS in filename.
		'''
		with tf.gfile.GFile(filename, "r") as f:
			data = ['<eos>' if x == '\n' else x for x in f.read().decode(self.encoding)]
			return data

class wordSentenceDataStream(wordSentenceData):
	'''
	Same as wordSentenceData but reads the data batch per batch instead of all at once.
	'''

	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(wordSentenceDataStream, self).__init__(config, eval_config, TRAIN, VALID, TEST)

	def calc_longest_sent(self, list_files):
		'''
		Calculates longest sentence based on list of files instead of already read data.
		'''
		max_length = 0
		for f in list_files:
			if os.path.isfile(f):
				for line in codecs.open(f, 'r', self.encoding):
					curr_length = len(line.strip().split(' '))
					if curr_length > max_length:
						max_length = curr_length

		return max_length

	def get_batch(self, f, test=False):
		if test:
			self.batch_size = self.eval_batch_size

		curr_batch = []
		seq_lengths = []
		for i in xrange(self.batch_size):
			curr_sentence = f.readline().strip()

			# if end of file is reached
			if curr_sentence == '':
				end_reached = True
				f.close()
				return None, None, end_reached, None

			# input batch: words
			curr_sentence_idx = [self.item_to_id[word] if word in self.item_to_id \
				else self.item_to_id[self.unk] for word in curr_sentence.split(' ')]
			# length of sentence (for dynamic rnn)
			seq_lengths.append(len(curr_sentence_idx))
			number_pads = self.max_length - len(curr_sentence_idx) + 1
			padding = [self.item_to_id[self.PADDING_SYMBOL]]*number_pads
			curr_sentence_idx.extend(padding)
			curr_batch.append(curr_sentence_idx)

		curr_batch_array = np.array(curr_batch)
		x = curr_batch_array[:,:-1]
		y = curr_batch_array[:,1:]

		seq_lengths_array = np.array(seq_lengths)

		return x, y, False, seq_lengths_array


	def prepare_data(self):

		if 'read_vocab_from_file' in self.config:
			# read vocabulary mapping and maximum sentence length from file
			self.item_to_id, self.id_to_item = load_item_to_id(self.config['read_vocab_from_file'], self.encoding)

			if len(self.item_to_id) != self.config['vocab_size']:
				raise IOError("The vocabulary size specified by 'vocab_size' ({0}) does not correspond \
						to the size of the vocabulary file given ({1}).".format(
						self.config['vocab_size'], len(self.item_to_id)))

			self.max_length = int(open(os.path.join(self.config['data_path'], "max_sentence_length")).readlines()[0].strip())
		else:
			# build input vocabulary
			self.item_to_id, self.id_to_item = self.build_vocab(self.train_path)

			# get maximum length of sentence in all files
			self.max_length = self.calc_longest_sent([self.train_path, self.valid_path, self.test_path])

		# padding symbol needed
		self.add_padding_symbol()

		return (self.train_path, self.valid_path, self.test_path), len(self.item_to_id), self.max_length

	def init_batching(self, data_path):

		self.end_reached = False
		data_file = codecs.open(data_path,"r", self.encoding)

		return data_file


class charWordData(wordSentenceData):
	'''
	Character-level data, but per word (padded until the maximum word length).
	Used for lm_char_rnn.
	'''
	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(charWordData, self).__init__(config, eval_config, TRAIN, VALID, TEST)

	def read_items(self, filename):
		'''
		Returns a list of all CHARACTERS in filename.
		'''

		with tf.gfile.GFile(filename, "r") as f:
			# Wikitext: more than 1 sentence per line, also introduce <eos> at ' . '
			if "WikiText" in self.config['data_path']:
				data = [list(x) if (x != '<eos>' and x != self.unk) else x for x in f.read().decode(
					self.encoding).replace("\n", " <eos> ").replace(" . "," <eos> ").split(" ")]

			else:
				data = [list(x) if (x != '<eos>' and x != self.unk) else x for x in f.read().decode(
					self.encoding).replace("\n", " <eos> ").split(" ")]

			data = flattern(data)

			return data # single list with all characters in the file

	def read_sentences(self, filename):
		'''
		Returns a list with all words in filename, each words is split in characters.
		'''
		with tf.gfile.GFile(filename, "r") as f:
			if "WikiText" in self.config['data_path']:
				all_words = [list(word) if (word != self.unk and word != '<eos>') else word for word in f.read().decode(
					self.encoding).replace("\n", " <eos> ").replace(" . "," <eos> ").split(" ")]
			else:
				# split word in characters if it is not <unk> or <eos>
				all_words = [list(word) if (word != self.unk and word != '<eos>') else word for word in f.read().decode(
					self.encoding).replace("\n", " <eos> ").split(" ")]

			# remove empty elements
			all_words = [word for word in all_words if word != []]

			return all_words

	def padding(self, dataset, total_length):
		'''
		Pad until max_length without adding <eos> symbol first.
		'''
		seq_lengths = []

		# total_length = max_length + 2 (inherited from wordSentenceData),
		# but since we did not add <eos> in addition to the padding symbols, the actual length is -1
		# if no extra padding symbol is used to ensure the last padding symbol still has a 'target', -2
		total_length = total_length - 2

		for word in dataset:
			seq_lengths.append(len(word))

			# pad rest of word until maximum length
			num_pads = total_length - len(word)
			for pos in xrange(num_pads):
				word.append(self.item_to_id[self.PADDING_SYMBOL])
		return dataset, seq_lengths

	def file_to_item_ids(self, filename):

		data = self.read_sentences(filename)
		data_ids = []
		for word in data:
			if word == '<eos>' or word == self.unk:
				data_ids.append([self.item_to_id[word]])
			else:
				data_ids.append([self.item_to_id[char] for char in word if char in self.item_to_id])
		return data_ids

	def get_data(self):
		all_data = self.read_data()
		self.add_padding_symbol()

		max_length = self.calc_longest_sent(all_data)
		self.max_length = max_length
		#self.num_steps = max_length
		#self.eval_num_steps = max_length

		padded_data, seq_lengths = self.pad_data(all_data, max_length)

		# return max_length+1 and not +2 because the last padding symbol is only there
		# to make sure that the target sequence does not end with the beginning of the next sequence
		return padded_data, len(self.id_to_item), max_length, seq_lengths

	def init_batching(self, data, test=False):
		if test:
			self.batch_size = self.eval_batch_size
			self.num_steps = self.eval_num_steps
		#else:
			#batch_size = self.batch_size
			#num_steps = self.num_steps

		if self.iterator == 0:
			self.end_reached = False

		self.test = test

		words = data[0]
		seq_lengths = data[1]

		data_len = len(words)*len(words[0])

		# to divide data in batch_size batches, each of length batch_len
		batch_len = data_len // self.batch_size

		# number of items in 1 batch_len = self.max_length (length of word) * self.num_steps (number of words)
		# subtract one because there is not target for the last word
		self.num_words_batch = batch_len // (self.max_length*self.num_steps) - 1

		# we want batch_len to be a multiple of num_steps (=size of padded sentence)
		#batch_len = self.num_words_batch * self.num_steps #v1
		batch_len = self.num_words_batch * self.max_length * self.num_steps

		# only batch_size x batch_len words fit,
		# divide by self.max_length because 'words' = list of lists (each max.length long)
		words = words[:(self.batch_size * batch_len)/self.max_length]

		# convert to numpy array
		#self.data_array = np.array(words).reshape(self.batch_size, self.num_words_batch*self.num_steps) #v1
		self.data_array = np.array(words).reshape(self.batch_size, batch_len)

		# convert seq_lengths to numpy array
		seq_lengths = seq_lengths[:(self.batch_size * batch_len)/self.max_length]
		self.seql_array = np.array(seq_lengths).reshape(self.batch_size, self.num_steps*self.num_words_batch)


	def get_batch(self):

		# take slice of batch_size x num_steps
		x = self.data_array[:, self.iterator * self.num_steps : (self.iterator * self.num_steps) + (self.num_steps*self.max_length)]
		x = x.reshape(self.batch_size, self.num_steps, self.max_length)

		y = self.data_array[:, (self.iterator * self.num_steps)+1 : (self.iterator * self.num_steps) + (self.num_steps*self.max_length) +1]
		y = y.reshape(self.batch_size, self.num_steps, self.max_length)
		# !!! TO DO: last element of each word is first character of next word --> correct this

		# take slice of sequence lengths for all elements in the batch
		seql = self.seql_array[:, self.iterator * self.num_steps : (self.iterator+1) * self.num_steps]

		# if iterated over the whole dataset, set iterator to 0 to start again
		if self.iterator >= self.num_words_batch:
			self.iterator = 0
			self.end_reached = True
		# otherwise, increase count
		else:
			self.iterator += 1

		return x, y, self.end_reached, seql


class wordSentenceDataRescore(wordSentenceData):
	'''
	Rescore N-best lists with model trained across sentence boundaries.
	'''

	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(wordSentenceDataRescore, self).__init__(config, eval_config, TRAIN, VALID, TEST)

	def file_to_item_ids(self, filename):
		data = self.read_sentences(filename)
		data_ids = []
		for sentence in data:
			# difference: words not in vocabulary are mapped to the unk symbol
			data_ids.append([self.item_to_id[item] if item in self.item_to_id \
				else self.item_to_id[self.unk] for item in sentence])
		return data_ids

	def get_data(self):
		all_data = self.read_data()

		max_length = self.config['num_steps'] - 3
		padded_data, _ = self.pad_data(all_data, max_length)

		# return max_length+2 and not +3 because the last padding symbol is only there
		# to make sure that the target sequence does not end with the beginning of the next sequence
		return padded_data, len(self.id_to_item), max_length+2

class charSentenceDataRescore(charSentenceData, wordSentenceDataRescore):
	'''
	Same as wordSentenceDataRescore but on character level.
	'''

	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(charSentenceDataRescore, self).__init__(config, eval_config, TRAIN, VALID, TEST)

	def file_to_item_ids(self, filename):
		return wordSentenceDataRescore.file_to_item_ids(self, filename)

	def get_data(self):
		return wordSentenceDataRescore.get_data(self)


class charNGramData(LMData):
	'''
	Feed character n-grams to the network (but still predict words).
	'''
	def __init__(self, config, eval_config, TRAIN, VALID, TEST):

		super(charNGramData, self).__init__(config, eval_config, TRAIN, VALID, TEST)

		if not isinstance(self.config['char_ngram'],int):
			raise IOError("Specify what n should be used for the character n-grams.")
		else:
			self.n = self.config['char_ngram']

		self.special_symbols = ['<UNK>', '<unk>', '<eos>']
		self.ngram_to_id = {}
		self.id_to_ngram = {}

		#if 'add_word' in self.config and 'input_vocab' in self.config:
		if 'add_word' in self.config:

			if not 'word_size' in self.config:
				raise IOError("Specify the size that should be assigned to the word input (word_size).")
			if not 'input_vocab_size' in self.config:
				raise IOError("Specify the size of the word input vocabulary (input_vocab_size).")

			self.input_item_to_id = {}
			self.input_id_to_item = {}

	def find_ngrams(self, data):
		'''
		Finds all ngrams in data.
		Arguments:
			data: list of all words in the training file
		Returns:
			freq_ngrams: dictionary containing all n-grams found + their frequency
		'''

		freq_ngrams = dict()

		for word in data:
			# add the special symbols as 1
			if word in self.special_symbols:
				if word in freq_ngrams:
					freq_ngrams[word] += 1
				else:
					freq_ngrams[word] = 1

			else:

				# first ngram: append <bow> to the beginning of the word
				first_ngram = '<bow>'+word[:self.n-1]
				if 'capital' in self.config:
					first_ngram = first_ngram.lower()

				if first_ngram in freq_ngrams:
					freq_ngrams[first_ngram] += 1
				else:
					freq_ngrams[first_ngram] = 1

				# n-grams in the middle of the word
				for pos in xrange(len(word)):

					# only add the ngram if it is long enough (end of the word: not long enough)
					if len(word[pos:pos+self.n]) == self.n:

						curr_ngram = word[pos:pos+self.n]

						# if special marker for capital: only use lower case n-grams
						if 'capital' in self.config:
							curr_ngram = curr_ngram.lower()

						# add ngram if not yet in set
						if curr_ngram in freq_ngrams:
							freq_ngrams[curr_ngram] += 1
						else:
							freq_ngrams[curr_ngram] = 1


				# last n-gram: append '<eow>' to end of word
				last_ngram = word[-1-self.n+2:]+'<eow>'
				if 'capital' in self.config:
					last_ngram = last_ngram.lower()

				if last_ngram in freq_ngrams:
					freq_ngrams[last_ngram] += 1
				else:
					freq_ngrams[last_ngram] = 1

		return freq_ngrams

	def find_skipgrams(self, data, skip):
		'''
		Finds all skipgrams in data.
		Arguments:
			data: list of all words in the training file
			skip: number of characters that should be skipped
		Returns:
			freq_ngrams: dictionary containing all skipgrams found + their frequency
		'''

		freq_skipgrams = dict()

		for word in data:
			# add the special symbols as 1
			if word in self.special_symbols:
				if word in freq_skipgrams:
					freq_skipgrams[word] += 1
				else:
					freq_skipgrams[word] = 1

			elif len(word) > 1:

				# first skipgram: append '<bow>' to beginning of word
				first_skipgram = '<bow>'+word[skip]
				if 'capital' in self.config:
					first_skipgram = first_skipgram.lower()

				if first_skipgram in freq_skipgrams:
					freq_skipgrams[first_skipgram] += 1
				else:
					freq_skipgrams[first_skipgram] = 1

				for pos in xrange(len(word)):

					# only add the skipgram if it is long enough (end of the word: not long enough)
					if len(word[pos:]) >= skip+2:

						curr_skipgram = word[pos] + word[pos+1+skip]

						# if special marker for capital: only use lower case n-grams
						if 'capital' in self.config:
							curr_skipgram = curr_skipgram.lower()

						if curr_skipgram in freq_skipgrams:
							freq_skipgrams[curr_skipgram] += 1
						else:
							freq_skipgrams[curr_skipgram] = 1


				# append '<eow>' to end of word
				last_skipgram = word[-1-skip]+'<eow>'
				if 'capital' in self.config:
					last_skipgram = last_skipgram.lower()

				if last_skipgram in freq_skipgrams:
					freq_skipgrams[last_skipgram] += 1
				else:
					freq_skipgrams[last_skipgram] = 1

		return freq_skipgrams

	def build_ngram_vocab(self, filename, skip=None):
		'''
		Reads the data and builds ngram-to-id mapping and id-to-ngram mapping.
		Arguments:
			filename: data file from which the vocbulary is read
			skip: if None, n-grams are read; if not None, 'skip' characters are skipped
		'''

		data = self.read_items(filename)

		# find all n-grams/skipgram + their frequency
		if skip != None:
			freq_ngrams = self.find_skipgrams(data, skip)
		else:
			freq_ngrams = self.find_ngrams(data)

		# for words that consist of only 1 character: add unigrams
		# possible TO DO: if n > 2, what to do with words of 2 characters?
		all_chars = set(''.join(data))
		for word in data:
			if word in all_chars:
				if word in freq_ngrams:
					freq_ngrams[word] += 1
				else:
					freq_ngrams[word] = 1

		# if only ngrams with a frequency > ngram_cutoff have to be kept
		if 'ngram_cutoff' in self.config:
			if not isinstance(self.config['ngram_cutoff'],int):
				raise ValueError("Specify what cutoff frequency should be used for the character n-grams.")
			else:
				freq_ngrams = {ngram:freq for ngram, freq in freq_ngrams.items() if freq > self.config['ngram_cutoff']}

		ngrams = freq_ngrams.keys()

		if 'capital' in self.config:
			# special symbol to indicate whether the word contains (a) capital(s) or not
			ngrams.append('<cap>')
			# remove ngrams with capitals from vocabulary
			ngrams = [gram for gram in ngrams if gram.islower()]

		# unknown n-gram symbol
		ngrams.append('<UNKngram>')

		self.ngram_to_id = dict(zip(ngrams, range(len(ngrams))))
		self.id_to_ngram = dict(zip(range(len(ngrams)), ngrams))

		print('Size of n-gram vocabulary: {0}'.format(len(self.ngram_to_id)))


	"""def file_to_ngram_ids(self, filename):
		data = self.read_items(filename)
		ngrams = []

		for word in data:
			# initialize zero vector of size of the ngram vocabulary
			ngram_repr = np.zeros(len(self.ngram_to_id), dtype=np.float32)

			# do not cut word in n-grams if it contains only 1 character or is a special symbol
			if len(word) == 1 or word in self.special_symbols:
				if word in self.ngram_to_id:
					ngram_repr[self.ngram_to_id[word]] += 1
				else:
					ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

			else:

				# if special marker for capital, check how many capitals the word has
				if 'capital' in self.config:
					num_capitals = sum(1 for char in word if char.isupper())
					if num_capitals > 0:
						# increase count at index of special capital marker
						ngram_repr[self.ngram_to_id['<cap>']] += num_capitals

				for pos in xrange(len(word)):

					# not yet at the end of the word (otherwise the subword might be shorter than n)
					if len(word[pos:pos+self.n]) == self.n:

						curr_ngram = word[pos:pos+self.n]

						# if special marker for capital: only use lower case n-grams
						if 'capital' in self.config:
							lower_version = curr_ngram.lower()
							curr_ngram = lower_version

						# increase count at index of character ngram
						if curr_ngram in self.ngram_to_id:
							ngram_repr[self.ngram_to_id[curr_ngram]] += 1
						else:
							ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

				# append '<eow>' to end of word
				last_ngram = word[-1-self.n+2:]+'<eow>'

				# increase count at index of character ngram
				if last_ngram in self.ngram_to_id:
					ngram_repr[self.ngram_to_id[last_ngram]] += 1
				else:
					ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

			ngrams.append(ngram_repr)

		return ngrams

	def file_to_skipgram_ids(self, filename, skip):
		data = self.read_items(filename)
		skipgrams = []

		for word in data:

			# initialize zero vector of size of the skipgram vocabulary
			skipgram_repr = np.zeros(len(self.ngram_to_id), dtype=np.float32)

			# do not cut word in n-grams if it contains only 1 character or is a special symbol
			if len(word) == 1 or word in self.special_symbols:
				if word in self.ngram_to_id:
					skipgram_repr[self.ngram_to_id[word]] += 1
				else:
					skipgram_repr[self.ngram_to_id['<UNKskipgram>']] += 1

			else:

				# append '<bow>' to beginning of word
				first_skipgram = '<bow>'+word[skip]
				if first_skipgram in self.ngram_to_id:
					skipgram_repr[self.ngram_to_id[first_skipgram]] += 1
				else:
					skipgram_repr[self.ngram_to_id['<UNKskipgram>']] += 1

				# if special marker for capital, check how many capitals the word has
				if 'capital' in self.config:
					num_capitals = sum(1 for char in word if char.isupper())
					if num_capitals > 0:
						# increase count at index of special capital marker
						skipgram_repr[self.ngram_to_id['<cap>']] += num_capitals

				for pos in xrange(len(word)):


					# not yet at the end of the word (otherwise the subword might be shorter than n)
					if len(word[pos:]) >= skip+2:

						curr_skipgram = word[pos] + word[pos+1+skip]
						#print(u'curr_skipgram: {0}'.format(curr_skipgram).encode(self.encoding))

						# if special marker for capital: only use lower case n-grams
						if 'capital' in self.config:
							lower_version = curr_skipgram.lower()
							curr_skipgram = lower_version

						# increase count at index of character skipgram
						if curr_skipgram in self.ngram_to_id:
							skipgram_repr[self.ngram_to_id[curr_skipgram]] += 1
							#print('skipgram in vocab: index = {0}'.format(self.ngram_to_id[curr_skipgram]))
						else:
							skipgram_repr[self.ngram_to_id['<UNKskipgram>']] += 1
							#print('skipgram not in vocab: index = {0}'.format(self.ngram_to_id['<UNKskipgram>']))

				# WHAT ABOUT THE END OF THE WORD?
				last_skipgram = word[-1-skip]+'<eow>'
				#print(u'last_skipgram: {0}'.format(last_skipgram).encode(self.encoding))

				if last_skipgram in self.ngram_to_id:
					skipgram_repr[self.ngram_to_id[last_skipgram]] += 1
				#	print('skipgram in vocab: index = {0}'.format(self.ngram_to_id[last_skipgram]))
				else:
					skipgram_repr[self.ngram_to_id['<UNKskipgram>']] += 1
				#	print('skipgram not in vocab: index = {0}'.format(self.ngram_to_id['<UNKskipgram>']))

			#print('skipgram_repr: {0}'.format(skipgram_repr))
			skipgrams.append(skipgram_repr)

		return skipgrams"""

	def map_ngrams_to_ids(self, ngram_repr, word):
		'''
		Maps all n-grams in the word to a count on the input vector.
		Arguments:
			ngram_repr: input vector
			word: word that should be mapped to n-grams
		Returns:
			ngram_repr: input vector, with counts for all n-grams in 'word' added
		'''

		# first ngram
		first_ngram = '<bow>'+word[:self.n-1]
		if 'capital' in self.config:
			first_ngram = first_ngram.lower()

		# increase count at index of character ngram
		if first_ngram in self.ngram_to_id:
			ngram_repr[self.ngram_to_id[first_ngram]] += 1
		else:
			ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

		for pos in xrange(len(word)):

			# not yet at the end of the word (otherwise the subword might be shorter than n)
			if len(word[pos:pos+self.n]) == self.n:

				curr_ngram = word[pos:pos+self.n]

				# if special marker for capital: only use lower case n-grams
				if 'capital' in self.config:
					curr_ngram = curr_ngram.lower()

				# increase count at index of character ngram
				if curr_ngram in self.ngram_to_id:
					ngram_repr[self.ngram_to_id[curr_ngram]] += 1
				else:
					ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

		# append '<eow>' to end of word
		last_ngram = word[-1-self.n+2:]+'<eow>'
		if 'capital' in self.config:
			last_ngram = last_ngram.lower()

		# increase count at index of character ngram
		if last_ngram in self.ngram_to_id:
			ngram_repr[self.ngram_to_id[last_ngram]] += 1
		else:
			ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

		return ngram_repr

	def map_skipgrams_to_ids(self, ngram_repr, word, skip):
		'''
		Maps all skipgrams in the word to a count on the input vector.
		Arguments:
			ngram_repr: input vector
			word: word that should be mapped to skipgrams
			skip: number of characters that should be skipped
		Returns:
			ngram_repr: input vector, with counts for all skipgrams in 'word' added
		'''

		# first skipgram
		first_skipgram = '<bow>'+word[skip]
		if 'capital' in self.config:
			first_skipgram = first_skipgram.lower()

		# increase count at index of character skipgram
		if first_skipgram in self.ngram_to_id:
			ngram_repr[self.ngram_to_id[first_skipgram]] += 1
		else:
			ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

		for pos in xrange(len(word)):

			# not yet at the end of the word (otherwise the subword might be shorter than n)
			if len(word[pos:]) >= skip+2:

				curr_skipgram = word[pos] + word[pos+1+skip]

				# if special marker for capital: only use lower case n-grams
				if 'capital' in self.config:
					curr_skipgram = curr_skipgram.lower()

				# increase count at index of character skipgram
				if curr_skipgram in self.ngram_to_id:
					ngram_repr[self.ngram_to_id[curr_skipgram]] += 1
				else:
					ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

		# append '<eow>' to end of word
		last_skipgram = word[-1-skip]+'<eow>'
		if 'capital' in self.config:
			last_skipgram = last_skipgram.lower()

		# increase count at index of character skipgram
		if last_skipgram in self.ngram_to_id:
			ngram_repr[self.ngram_to_id[last_skipgram]] += 1
		else:
			ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

		return ngram_repr

	def file_to_ngram_ids(self, filename):
		'''
		Generates occurrence vectors for all words in the file.
		Arguments:
			filename: name of data file
		Returns:
			ngrams: a list of ngram_repr, which are numpy arrays containing the counts of each n-gram
		'''

		data = self.read_items(filename)
		ngrams = []

		for word in data:
			# initialize zero vector of size of the ngram vocabulary
			ngram_repr = np.zeros(len(self.ngram_to_id), dtype=np.float32)

			# do not cut word in n-grams if it contains only 1 character or is a special symbol
			if len(word) == 1 or word in self.special_symbols:
				if word in self.ngram_to_id:
					ngram_repr[self.ngram_to_id[word]] += 1
				else:
					ngram_repr[self.ngram_to_id['<UNKngram>']] += 1

			else:

				# if special marker for capital, check how many capitals the word has
				if 'capital' in self.config:
					num_capitals = sum(1 for char in word if char.isupper())
					if num_capitals > 0:
						# increase count at index of special capital marker
						ngram_repr[self.ngram_to_id['<cap>']] += num_capitals

				if not 'skipgram' in self.config:
					ngram_repr = self.map_ngrams_to_ids(ngram_repr, word)
				else:
					ngram_repr = self.map_skipgrams_to_ids(ngram_repr, word, self.config['skipgram'])

			ngrams.append(ngram_repr)

		return ngrams


	def read_data(self):
		# n-gram input: use data with full vcoabulary, where words are not converted to <UNK>
		train_path_full_vocab = os.path.join(self.config['data_path'], "train.txt")
		valid_path_full_vocab = os.path.join(self.config['data_path'], "valid.txt")
		test_path_full_vocab = os.path.join(self.config['data_path'], "test.txt")

		if 'skipgram' in self.config:
			if self.config['char_ngram'] != 2 or self.config['skipgram'] != 1:
				raise NotImplementedError("Skipgrams have only been implemented for char_ngram = 2 and skipgram = 1.")
			self.build_ngram_vocab(train_path_full_vocab, self.config['skipgram'])
		else:
			self.build_ngram_vocab(train_path_full_vocab)

		# output vocabulary: use reduced vocabulary
		self.item_to_id, self.id_to_item = self.build_vocab(self.train_path)

		# combine character n-grams with word input
		if 'add_word' in self.config:
			# if input vocabulary is different from output vocabulary
		 	if 'input_vocab' in self.config:
				train_file = "train_" + str(self.config['input_vocab']) + "k-unk.txt"
				valid_file = "valid_" + str(self.config['input_vocab']) + "k-unk.txt"
				test_file = "test_" + str(self.config['input_vocab']) + "k-unk.txt"
			else:
				train_file = "train.txt"
				valid_file = "valid.txt"
				test_file = "test.txt"
			input_train_path = os.path.join(self.config['data_path'], train_file)
			input_valid_path = os.path.join(self.config['data_path'], valid_file)
			input_test_path = os.path.join(self.config['data_path'], test_file)

			# build vocab for input word representation
			self.input_item_to_id, self.input_id_to_item = self.build_vocab(input_train_path)

		# lists of all ngrams/words in training data converted to their ids
		if self.TRAIN:
			#if 'skipgram' in self.config:
			#	train_ngrams = self.file_to_skipgram_ids(train_path_full_vocab, self.config['skipgram'])
			#else:
			train_ngrams = self.file_to_ngram_ids(train_path_full_vocab)

			if 'add_word' in self.config and 'input_vocab' in self.config:
				train_input_words = self.file_to_item_ids(input_train_path, item_to_id=self.input_item_to_id)

			train_words = self.file_to_item_ids(self.train_path)
		else:
			train_ngrams = []
			train_words = []
			train_input_words = []

		# lists of all ngrams/words in validation data converted to their ids
		if self.VALID:
			#if 'skipgram' in self.config:
			#	valid_ngrams = self.file_to_skipgram_ids(valid_path_full_vocab, self.config['skipgram'])
			#else:
			valid_ngrams = self.file_to_ngram_ids(valid_path_full_vocab)

			if 'add_word' in self.config and 'input_vocab' in self.config:
				valid_input_words = self.file_to_item_ids(input_valid_path, item_to_id=self.input_item_to_id)

			valid_words = self.file_to_item_ids(self.valid_path)
		else:
			valid_ngrams = []
			valid_words = []
			valid_input_words = []

		# lists of all ngrams/words in test data converted to their ids
		if self.TEST:
			#if 'skipgram' in self.config:
			#	test_ngrams = self.file_to_skipgram_ids(test_path_full_vocab, self.config['skipgram'])
			#else:
			test_ngrams = self.file_to_ngram_ids(test_path_full_vocab)

			if 'add_word' in self.config and 'input_vocab' in self.config:
				test_input_words = self.file_to_item_ids(input_test_path, item_to_id=self.input_item_to_id)

			test_words = self.file_to_item_ids(self.test_path)
		else:
			test_ngrams = []
			test_words = []
			test_input_words = []

		if 'add_word' in self.config and 'input_vocab' in self.config:
			train_words = (train_words, train_input_words)
			valid_words = (valid_words, valid_input_words)
			test_words = (test_words, test_input_words)

		all_data = ((train_ngrams,train_words), (valid_ngrams,valid_words),(test_ngrams,test_words))

		return all_data

	def get_data(self):
		all_data = self.read_data()
		lengths = (len(self.id_to_ngram), len(self.id_to_item))

		return all_data, lengths, 0

	def init_batching(self, data, test=False):
		if test:
			batch_size = self.eval_batch_size
			self.num_steps = self.eval_num_steps
		else:
			batch_size = self.batch_size
			#self.num_steps = self.num_steps

		ngram_data, word_data = data

		if 'add_word' in self.config and 'input_vocab' in self.config:
			word_data, input_word_data = word_data

		input_size = self.config['input_size']

		if self.iterator == 0:
			self.end_reached = False

		data_len = len(word_data)
		# to divide data in batch_size batches, each of length batch_len
		batch_len = data_len // batch_size
		# number of samples that can be taken from the batch_len slices
		if self.num_steps != 1:
			self.num_samples = batch_len // self.num_steps
		else:
			self.num_samples = (batch_len // self.num_steps) - 1

		# remove last part of the data that doesn't fit in the batch_size x num_steps samples
		ngram_data = ngram_data[:batch_size * batch_len]
		word_data = word_data[:batch_size * batch_len]

		print('batch_len: {0}'.format(batch_len))
		print('input_size: {0}'.format(input_size))

		# for n-gram inputs: convert to numpy array: batch_size x batch_len x input_size
		self.data_array_ngrams = np.array(ngram_data).reshape(batch_size, batch_len, input_size)

		# for word outputs: convert to numpy array: batch_size x batch_len
		self.data_array_words = np.array(word_data).reshape(batch_size, batch_len)

		# if word representation is added to the input and input and output vocabulary are not the same
		if 'add_word' in self.config and 'input_vocab' in self.config:
			input_word_data = input_word_data[:batch_size * batch_len]
			self.data_array_input_words = np.array(input_word_data).reshape(batch_size, batch_len)

		self.batching_initialized = True

	def get_batch(self):

		if not self.batching_initialized:
			raise ValueError("Batching is not yet initialized.")

		# inputs = ngrams (take slice of batch_size x num_steps)
		x = self.data_array_ngrams[:, self.iterator * self.num_steps : (self.iterator * self.num_steps) + self.num_steps]

		if 'add_word' in self.config:

			# different size for input and output vocabulary
			if 'input_vocab' in self.config:
				x_words = self.data_array_input_words[:, self.iterator * self.num_steps :
								(self.iterator * self.num_steps) + self.num_steps]
			# same size for input and output vocabulary
			else:
				x_words = self.data_array_words[:, self.iterator * self.num_steps :
								(self.iterator * self.num_steps) + self.num_steps]

			x = (x, x_words)

		# targets = words (same slice but shifted one step to the right)
		y = self.data_array_words[:, (self.iterator * self.num_steps) +1 :
						(self.iterator * self.num_steps) + self.num_steps + 1]

		self.iterator += 1

		# if iterated over the whole dataset, set iterator to 0 to start again
		if self.iterator >= self.num_samples:
			self.iterator = 0
			self.end_reached = True
		# otherwise, increase count
		#else:
		#	self.iterator += 1

		return x, y, self.end_reached
