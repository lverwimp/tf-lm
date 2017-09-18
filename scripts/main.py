#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import time, os, sys, random

import numpy as np
import tensorflow as tf

from writer import writer
import configuration, lm_data, multiple_lm_data, trainer, lm, run_epoch

print('TensorFlow version: {0}'.format(tf.__version__))

# command line arguments
flags = tf.flags
flags.DEFINE_string("config", None,"Configuration file")
flags.DEFINE_boolean("train", True,"Train the model or not.")
flags.DEFINE_boolean("valid", True,"Validate the model or not.")
flags.DEFINE_boolean("test", True,"Test the model or not.")
flags.DEFINE_string("device", None,"Specify 'cpu' if you want to run on cpu.")
FLAGS = flags.FLAGS

# TO ADAPT: directory for log files
LOG_DIR = '/esat/spchtemp/scratch/lverwimp/tensorflow/logs/'


def read_data(config, eval_config, (TRAIN, VALID, TEST)):
	'''
	Reads data from file.
	Inputs:
		config: dictionary containing configuration options (for training and validation)
		eval_config: dictionary containing configuration options (for testing)
		(TRAIN, VALID, TEST): tuple of booleans indicating whether we should train, validate and/or test
	Returns:
		config: dictionary containing configuration options (for training and validation)
		eval_config: dictionary containing configuration options (for testing)
		data: data object
		train_data: training data mapped to indices (can be single list or tuple of lists depending on the type of model)
		valid_data: validation data mapped to indices
		test_data: test data mapped to indices
		(TRAIN, VALID, TEST): tuple of booleans indicating whether we should train, validate and/or test
	'''

	if 'predict_next' in config or 'rescore' in config or 'debug2' in config:
		TRAIN = False
		VALID = False

	if 'bidirectional' in config and not 'per_sentence' in config:
		raise IOError("Training of bidirectional models is only supported for sentence-level models.")
		sys.exit(1)

	# character-level training, in batches (across sentence boundaries)
	if 'char' in config:
		print('Character-level data')
		if 'per_sentence' in config:
			print('Sentence per sentence')
			data = lm_data.charSentenceData(config, eval_config, TRAIN, VALID, TEST)
			all_data, vocab_size, total_length, seq_lengths = data.get_data()

			config['num_steps'] = total_length

			if not 'rescore' in config and not 'predict_next' in config:
				eval_config['num_steps'] = total_length

			# write maximum sentence length to file
			max_length_f = open('{0}max_length'.format(config['save_path']), 'w')
			max_length_f.write('{0}\n'.format(total_length))
			max_length_f.close()

		elif 'rescore' in config or 'predict_next' in config:
			# use trained model for rescoring
			print('For rescoring: sentence per sentence')
			data = lm_data.charSentenceDataRescore(config, eval_config, TRAIN, VALID, TEST)
			all_data, vocab_size, _ = data.get_data()
		elif 'debug2' in config:
			raise NotImplementedError("Generating a debug2 file with a character-level \
				model is not implemented.")
		else:
			data = lm_data.charData(config, eval_config, TRAIN, VALID, TEST)
			all_data, vocab_size, _ = data.get_data()


	# word-level training, on sentence level (sentences are padded until maximum sentence length)
	elif 'per_sentence' in config:

		if 'char_ngram' in config:
			raise NotImplementedError("Models with character n-gram input are only "
				"implemented on discourse level.")
		elif 'word_char_concat' in config:
			raise NotImplementedError("Models with concatenated word and character embeddings "
				"as input are only implemented at discourse level.")

		# do not read all data at once (for large datasets/small memory)
		elif 'stream_data' in config:
			print('Sentence-level data, stream data instead or reading all at once')

			data = lm_data.wordSentenceDataStream(config, eval_config, TRAIN, VALID, TEST)

			# !!! all_data = tuple containing strings specifying the paths of the data rather than the data itself
			all_data, vocab_size, total_length = data.prepare_data()

			config['num_steps'] = total_length

			if not 'rescore' in config and not 'predict_next' in config:
				eval_config['num_steps'] = total_length

			# write maximum sentence length to file
			max_length_f = open('{0}max_length'.format(config['save_path']), 'w')
			max_length_f.write('{0}\n'.format(total_length))
			max_length_f.close()

		else:
			print('Sentence-level data')

			if 'rescore' in config and 'bidirectional' in config:
				raise NotImplementedError("Rescoring with a bidirectional model is not (yet) implemented.")
			elif 'predict_next' in config and 'bidirectional' in config:
				raise NotImplementedError("Predicting the next word with a bidirectional model is not (yet) implemented.")
			elif 'debug2' in config and 'bidirectional' in config:
				raise NotImplementedError("Generating a debug2 file with a bidirectional model is not (yet) implemented.")

			data = lm_data.wordSentenceData(config, eval_config, TRAIN, VALID, TEST)
			all_data, vocab_size, total_length, seq_lengths = data.get_data()

			# set num_steps = total length of each (padded) sentence
			config['num_steps'] = total_length

			print('Write max length of sentence to {0}max_length'.format(config['save_path']))

			# write maximum sentence length to file
			max_length_f = open('{0}max_length'.format(config['save_path']), 'w')
			max_length_f.write('{0}\n'.format(total_length))
			max_length_f.close()

	# rescoring with non-sentence-level LMs: prepare data sentence-level
	# except when it is explicitily specified not to (across_sentence)
	elif ('rescore' in config or 'debug2' in config or 'predict_next' in config) and not 'across_sentence' in config:
		print('Data for rescoring (not on sentence level trained), for generating debugging file or for predicting next word')

		if 'char_ngram' in config:
			raise NotImplementedError('Rescoring/generating a debug file/predicting the next word is '
				'not (yet) implemented for models with character n-grams as input.')
		elif 'word_char_concat' in config:
			raise NotImplementedError('Rescoring/generating a debug file/predicting the next word is '
				'not (yet) implemented for models with concatenated word and character embeddings as input.')

		if 'debug2' in config and not TEST:
			config['valid_as_test'] = True
			TEST = True

		data = lm_data.wordSentenceDataRescore(config, eval_config, TRAIN, VALID, TEST)
		all_data, vocab_size, _ = data.get_data()

	# train an LM with as input a concatenation of word and character embeddings
	elif 'word_char_concat' in config:
		print('Data for concatenated word and character embeddings')

		# word data
		word_data = lm_data.LMData(config, eval_config, TRAIN, VALID, TEST)
		all_word_data, vocab_size, _ = word_data.get_data()

		# character data
		char_data = multiple_lm_data.multipleLMDataChar(config, eval_config, TRAIN, VALID, TEST)
		all_char_data, char_vocab_size = char_data.get_data()

		config['char_vocab_size'] = char_vocab_size
		eval_config['char_vocab_size'] = char_vocab_size

	# train character n-grams as input and words as output
	elif 'char_ngram' in config:
		print('Character n-gram data')
		data = lm_data.charNGramData(config, eval_config, TRAIN, VALID, TEST)
		all_data, (input_size, vocab_size), _ = data.get_data()

		# input_size = size of character n-gram vocabulary
		config['input_size'] = input_size
		eval_config['input_size'] = input_size


	# word-level training, in batches (goes across sentence boundaries)
	else:
		if 'stream_data' in config:
			raise NotImplementedError("Streaming data is only implemented for sentence-level batching.")

		print('Word-level data, across sentence boundaries')
		data = lm_data.LMData(config, eval_config, TRAIN, VALID, TEST)
		all_data, vocab_size, _ = data.get_data()

	# update vocab_size
	config['vocab_size'] = vocab_size
	eval_config['vocab_size'] = vocab_size

	# rescoring/printing predictions/generating debug2 file: only test_data needed
	if 'rescore' in config or 'predict_next' in config or 'debug2' in config:
		test_data = all_data[2]
		return config, eval_config, data, None, None, test_data, (TRAIN, VALID, TEST)

	# concatenation of word an character embeddings: tuple of word data and character data
	elif 'word_char_concat' in config:
		train_data = (all_word_data[0], all_char_data[0])
		valid_data = (all_word_data[1], all_char_data[1])
		test_data = (all_word_data[2], all_char_data[2])
		return config, eval_config, (word_data, char_data), train_data, valid_data, test_data, (TRAIN, VALID, TEST)

	else:
		# sentence-level: sentence lengths needed
		if 'per_sentence' in config and not 'stream_data' in config:
			train_data = (all_data[0], seq_lengths[0])
			valid_data = (all_data[1], seq_lengths[1])
			test_data = (all_data[2], seq_lengths[2])

		# all other cases
		else:
			train_data = all_data[0]
			valid_data = all_data[1]
			test_data = all_data[2]

		return config, eval_config, data, train_data, valid_data, test_data, (TRAIN, VALID, TEST)

def create_lm(config, is_training, reuse, test=False):
	'''
	Creates language model.
	'''
	if 'word_char_concat' in config:
		print('Model with concatenated word and character embeddings')
		lm_obj = lm.lm_charwordconcat(config, is_training=is_training, reuse=reuse)
	elif 'char_ngram' in config:
		print('Model with character n-grams as input')
		lm_obj = lm.lm_ngram(config, is_training=is_training, reuse=reuse)
	else:
		print('Standard LM')
		lm_obj = lm.lm(config, is_training=is_training, reuse=reuse)

	return lm_obj


def main(_):
	# process command line arguments and configuration
	if FLAGS.config == None:
		raise ValueError("Please specify a configuration file (with --config).")
	else:
		config = configuration.get_config(FLAGS.config)

	TRAIN = FLAGS.train
	VALID = FLAGS.valid
	TEST = FLAGS.test

	if TRAIN and not VALID:
		raise ValueError("Training and validation are always combined. Set both TRAIN = True and VALID = True.")

	device = FLAGS.device
	if device == 'cpu':
		os.environ['CUDA_VISIBLE_DEVICES']="" # if you don't want to use GPU

	eval_config = config.copy() # same parameters for evaluation, except for:

	eval_config['batch_size'] = 1 # batch_size

	# num_steps for testing model = 1, except for:
	# bidirectional model used for predicting: still feed full length of sentence because
	# otherwise the backwards model will have no extra input
	if 'bidirectional' in config and 'predict_next' in config:
		pass
	else:
		eval_config['num_steps'] = 1

	try:
		os.makedirs(config['save_path'])
	except OSError:
		pass

	if 'log' in config:
		log_file = LOG_DIR + os.path.basename(os.path.normpath(config['log'])) + '.log'
	else:
		log_file = LOG_DIR + os.path.basename(os.path.normpath(config['save_path'])) + '.log'
	# if log file already exists, make a new version by adding a random number (to avoid overwriting)
	if os.path.isfile(log_file):
		rand_num = round(random.random(),3)
		log_file = log_file.strip('.log') + str(rand_num) + '.log'

	fout = file(log_file,'w',0)
	# write both to standard output and log file
	sys.stdout = writer(sys.stdout, fout)

	print('configuration:')
	for par,value in config.iteritems():
		print('{0}\t{1}'.format(par, value))

	# read data in appropriate format + adapt configs if necessary
	config, eval_config, data, train_data, valid_data, test_data, (TRAIN, VALID, TEST) = read_data(
		config, eval_config, (TRAIN, VALID, TEST))

	with tf.Graph().as_default():

		# TO DO: CHECK WHETHER THIS WORKS!!!
		if not 'random' in config:
			# use the same seed for random initialization (to better compare models)
			tf.set_random_seed(1)
		initializer = tf.random_uniform_initializer(minval=-config['init_scale'], maxval=config['init_scale'], seed=1)

		if TRAIN:

			reuseOrNot = True # valid and test models: reuse the graph

			print('Create training model...')
			with tf.name_scope("Train"):
				with tf.variable_scope("Model", reuse=None, initializer=initializer):
					train_lm = create_lm(config, is_training=True, reuse=False)

		else:
			reuseOrNot = None

		if VALID:
			print('Create validation model...')
			with tf.name_scope("Valid"):
				with tf.variable_scope("Model", reuse=reuseOrNot, initializer=initializer):
					valid_lm = create_lm(config, is_training=False, reuse=reuseOrNot)

			if reuseOrNot == None:
				reuseOrNot = True


		if TEST:

			print('Create testing model...')
			with tf.name_scope("Test"):
				with tf.variable_scope("Model", reuse=reuseOrNot, initializer=initializer):
					test_lm = create_lm(eval_config, is_training=False, reuse=reuseOrNot, test=True)

		sv = tf.train.Supervisor(logdir=config['save_path'])

		# allow_soft_placement: automatically choose device
		with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

			if TRAIN and VALID:

				# create a trainer object based on config file
				class_name = 'trainer.{0}'.format(config['trainer'])
				train_obj = eval(class_name)(sv, session, config, train_lm, valid_lm, data, train_data, valid_data)

				# train + validate the model
				train_obj.train()

			if VALID and not TRAIN:

				#validator = run_epoch.run_epoch(session, valid_lm, data, valid_data)
				validator = run_epoch.run_epoch(session, valid_lm, data, valid_data,
						eval_op=None, test=False)

				valid_perplexity = validator()
				print('Valid Perplexity: {0}'.format(valid_perplexity))

			if TEST:

				# test the model
				print('Start testing...')

				if 'rescore' in config or 'debug2' in config or 'predict_next' in config:

					# read sentence per sentence from file
					if 'per_sentence' in config and 'stream_data' in config:
						tester = run_epoch.rescore(session, test_lm, data, test_data, eval_op=None, test=True)

						data_file = data.init_batching(test_data)

						end_reached = False

						while True:

							if 'bidirectional' in config:
								length_batch = test_lm.num_steps + 1
								x, y, end_reached, seq_lengths = data.get_batch(data_file, test=True, num_steps=length_batch)
							else:
								length_batch = test_lm.num_steps
								x, y, end_reached, seq_lengths = data.get_batch(data_file, test=True)

							if end_reached:
								break

							tester(np.reshape(x, [length_batch]))

					# normal sentence-level rescoring
					# test_data already contains all data
					else:

						tester = run_epoch.rescore(session, test_lm, data, test_data, eval_op=None, test=True)

						len_test = len(test_data)
						counter = 0
						for line in test_data:
							tester(line)
							counter += 1
							if counter % 100 == 0:
								print('{0} sentences processed'.format(counter))

				else:

					tester = run_epoch.run_epoch(session, test_lm, data, test_data,
						eval_op=None, test=True)

					test_perplexity = tester()

					print('Test Perplexity: {0}'.format(test_perplexity))



if __name__ == "__main__":
	tf.app.run()

