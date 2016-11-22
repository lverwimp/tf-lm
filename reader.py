# based on ptb tutorial (27/10/16)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os, sys

import tensorflow as tf
import numpy as np

# turn this switch on for debugging
DEBUG = True

def debug(string):
	if DEBUG:
		sys.stderr.write('DEBUG: {0}'.format(string))

def _read_words(filename):
	'''Returns a list of all words in filename.'''
	with tf.gfile.GFile(filename, "r") as f:
		return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _read_sentences(filename):
	'''Returns a list with all sentences in filename, each sentence split in words.'''
	with tf.gfile.GFile(filename, "r") as f:
		all_sentences = f.read().decode("utf-8").replace("\n", "<eos>").split("<eos>")
		# remove empty element at the end
		if all_sentences[-1] == '':
			all_sentences = all_sentences[:-1]
		for i in xrange(len(all_sentences)):
			all_sentences[i] = all_sentences[i].split()
		return all_sentences


def _build_vocab(filename, config):
	'''Returns a word-to-id and id-to-word mapping for all words in filename.'''
	data = _read_words(filename)

	counter = collections.Counter(data)

	# counter.items() = list of the words in data + their frequencies, then sorted according to decreasing frequency
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	
	# words = list of all the words (in decreasing frequency)
	words, _ = list(zip(*count_pairs))

	# make a dictionary with a mapping from each word to an id; word with highest frequency gets lowest id etc.
	word_to_id = dict(zip(words, range(len(words))))

	# reverse dictionary
	id_to_word = dict(zip(range(len(words)), words))

	# if processing per sentence: add beginning of sentence symbol + padding symbol
	if 'per_sentence' in config:
		word_to_id['<bos>'] = len(word_to_id)
		id_to_word[len(id_to_word)] = '<bos>'

		if '@' not in word_to_id:
			word_to_id['@'] = len(word_to_id)
			id_to_word[len(id_to_word)] = '@'
		else:
			raise ValueError("@ used as padding symbol but occurs in text.")

	# if n-best rescoring: we need a symbol for OOV-words
	if 'nbest' in config:
		if not '<UNK>' in word_to_id and not '<unk>' in word_to_id:
			word_to_id['<UNK>'] = len(word_to_id)
			id_to_word[len(id_to_word)] = '<UNK>'
				

	return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id, config):
	'''Returns list of all words in the file, either one long list or a list of lists per sentence.'''
	if 'per_sentence' in config:
		data = _read_sentences(filename)
		data_ids = []
		for sentence in data: 
			data_ids.append([word_to_id[word] for word in sentence if word in word_to_id])
		return data_ids
	else:
		data = _read_words(filename)
		return [word_to_id[word] for word in data if word in word_to_id]

def calc_longest_sent(all_data):
	'''Returns length of longest sentence occurring in all_data.'''
	max_length = 0
	for dataset in all_data:
		for sentence in dataset:
			if len(sentence) > max_length:
				max_length = len(sentence)
	return max_length


def padding(dataset, total_length, word_to_id):
	'''Add <bos> and <eos> to each sentence in dataset + pad until max_length.'''
	for sentence in dataset:
		# beginning of sentence symbol
		sentence.insert(0, word_to_id['<bos>'])
		# end of sentence symbol
		sentence.append(word_to_id['<eos>'])
		# pad rest of sentence until maximum length
		num_pads = total_length - len(sentence)
		for pos in xrange(num_pads):
			sentence.append(word_to_id['@'])
	return dataset


def pad_data(all_data, max_length, word_to_id):
	'''Pad each dataset in all_data.'''
				
	# <bos> and <eos> should be added 
	# + 1 extra padding symbol to avoid having target sequences which end on the beginning of the next sentence
	total_length = max_length + 3
 
	if isinstance(all_data, tuple):
		padded_all = ()
		for dataset in all_data:
			padded_all += (padding(dataset, total_length, word_to_id),)
	else:
		padded_all = padding(all_data, total_length, word_to_id)

	return padded_all

def ptb_raw_data(config):
	'''Reads all data. Returns 
		- a tuple with each dataset converted to a list of word ids
		- the id to word mapping
		- the total length of a sentence in case of padding
		- the n-best hypotheses in case of n-best rescoring. '''

	# if words not in vocabulary are converted to UNK
	if config['vocab']:
		train_file = "train_" + str(config['vocab']) + "k-unk.txt"
		valid_file = "valid_" + str(config['vocab']) + "k-unk.txt"
		test_file = "test_" + str(config['vocab']) + "k-unk.txt"
		train_path = os.path.join(config['data_path'], train_file)
		valid_path = os.path.join(config['data_path'], valid_file)
		test_path = os.path.join(config['data_path'], test_file)
	else: 
		train_path = os.path.join(config['data_path'], "train.txt")
		valid_path = os.path.join(config['data_path'], "valid.txt")
		test_path = os.path.join(config['data_path'], "test.txt")

	# dictionary containing word to id mapping
	word_to_id, id_to_word = _build_vocab(train_path, config)

	# list of all words in training data converted to their ids
	train_data = _file_to_word_ids(train_path, word_to_id, config)

	# list of all words in validation data converted to their ids
	valid_data = _file_to_word_ids(valid_path, word_to_id, config)

	# list of all words in test data converted to their ids
	test_data = _file_to_word_ids(test_path, word_to_id, config)

	all_data = (train_data, valid_data, test_data)

	# for n-best rescoring: read n-best hypotheses
	# only for sentence-level language model
	if 'nbest' in config:
		max_length = calc_longest_sent(all_data)
		hypotheses = _file_to_word_ids(config['nbest'], word_to_id, config)
		padded_hypotheses = pad_data(hypotheses, max_length, word_to_id)
		return all_data, id_to_word, max_length+2, padded_hypotheses


	# if data is processed per sentence, calculate length of longest sentence + pad all other sentences
	if 'per_sentence' in config:
		max_length = calc_longest_sent(all_data)
		padded_data = pad_data(all_data, max_length, word_to_id)

		# return max_length+2 and not +3 because the last padding symbol is only there 
		# to make sure that the target sequence does not end with the beginning of the next sequence
	 	return padded_data, id_to_word, max_length+2
	
	else:
		return all_data, id_to_word, 0


def ptb_producer(raw_data, config, name=None):
	'''Returns mini-batches of raw_data.'''
	with tf.name_scope("PTBProducer"):
		batch_size = config['batch_size']
		num_steps = config['num_steps']

		if 'per_sentence' in config:
			raw_data = [word for sentence in raw_data for word in sentence] # flatten list of lists

		data_len = len(raw_data)

		# data should be divided in batch_size pieces of size batch_len
		batch_len = data_len // batch_size

		# if processing per sentence: we don't want to cut in the middle of a sentence, 
		# so make sure that batch_len is a multiple of num_steps (= length of each padded sentence)
		if 'per_sentence' in config:
			num_sentences = batch_len // (num_steps+1)
			batch_len = num_sentences * (num_steps+1)

		# Tensor containing all words in the text (in 1 dimension)
		raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

		# convert to 2-dimensional Tensor of batch_size x batch_len
		# the last words are thrown away (only up till batch_size * batch_len is kept)
		# first row of resulting tensor = first batch_len words in text, second row contain next batch_len words etc.
		data = tf.reshape(raw_data[0 : batch_size * batch_len],[batch_size, batch_len])

		# epoch_size = how many times you can extract a sample of num_steps words from the data
		if num_steps == 1:
			epoch_size = (batch_len - 1) // num_steps
		else:
			epoch_size = (batch_len - 1) // (num_steps+1)

		epoch_size = tf.identity(epoch_size, name="epoch_size")

		# tf.train.range_input_producer produces the integers from 0 to epoch_size - 1 in a queue
		# .dequeue() dequeues one element from the queue: each time the lowest (element) is dequeued
		# this makes sure that every time a new sample of data is provided
		i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

		# inputs: extract slice of batch_size x num_steps; first batch begins at index 0, second at index num_steps(*1), third at index num_steps*2 etc.
		# one row in the batch can contain only part of a sentence or multiple sentences!
		#x = tf.slice(data, [0, i * (num_steps+1)], [batch_size, num_steps])
		x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
		
		# targets: same but shift everything 1 step to the right
		#y = tf.slice(data, [0, i * (num_steps+1) + 1], [batch_size, num_steps])
		y = tf.slice(data, [0, i * num_steps+1], [batch_size, num_steps])

		return x, y

