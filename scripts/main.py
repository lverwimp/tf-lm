#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import time, os, sys, time, random  
       
import numpy as np
import tensorflow as tf

from writer import writer
import configuration, lm_data, trainer, lm, run_epoch

# command line arguments 
flags = tf.flags
flags.DEFINE_string("config", None,"Configuration file")
flags.DEFINE_boolean("train", True,"Train the model or not.")
flags.DEFINE_boolean("valid", True,"Validate the model or not.")
flags.DEFINE_boolean("test", True,"Test the model or not.")
flags.DEFINE_string("device", None,"Specify 'cpu' if you want to run on cpu.")
FLAGS = flags.FLAGS

# directory for log files
LOG_DIR = '/esat/spchtemp/scratch/lverwimp/tensorflow/logs/'

print('TensorFlow version: {0}'.format(tf.__version__))

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
	
	# character-level training, in batches (cross sentence boundaries)
	if 'char' in config:
		data = lm_data.charData(config, eval_config)
		all_data, vocab_size, _ = data.get_data()

	# word-level training, on sentence level (sentences are padded until maximum sentence length)
	elif 'per_sentence' in config:

		if 'rescore' in config:
			max_length = int(open('{0}max_length'.format(config['trained_model'])).readlines()[0].strip())
			# set num_steps = total length of each (padded) sentence
			config['num_steps'] = max_length

			data = lm_data.wordSentenceDataRescore(config, eval_config)
			all_data, vocab_size, _ = data.get_data()

		else:
			data = lm_data.wordSentenceData(config, eval_config)
			all_data, vocab_size, total_length, seq_lengths = data.get_data()

			# set num_steps = total length of each (padded) sentence
			config['num_steps'] = total_length

			print('Write max length of sentence to {0}max_length'.format(config['save_path']))
	
			# write maximum sentence length to file
			max_length_f = open('{0}max_length'.format(config['save_path']), 'w')
			max_length_f.write('{0}\n'.format(total_length))
			max_length_f.close()
		
	# rescoring with non-sentence-level LMs	
	elif 'rescore' in config:

		data = lm_data.wordSentenceDataRescore(config, eval_config)
		all_data, vocab_size, _ = data.get_data()
		
		print('all_data len: {0}'.format(len(all_data)))

	# word-level training, in batches (goes across sentence boundaries)
	else:
		data = lm_data.LMData(config, eval_config)
		all_data, vocab_size, _ = data.get_data()
		
	# update vocab_size
	config['vocab_size'] = vocab_size
	eval_config['vocab_size'] = vocab_size
		
	# get the data sets you need and update TRAIN/VALID if necessary
	
	# rescoring: only test_data needed
	if 'rescore' in config:
		test_data = all_data
		TRAIN = False
		VALID = False
		return config, eval_config, data, _, _, test_data, (TRAIN, VALID, TEST)
	
	else:
		# sentence-level: sentence lengths needed
		if 'per_sentence' in config:
			train_data = (all_data[0], seq_lengths[0])
			valid_data = (all_data[1], seq_lengths[1])
			test_data = (all_data[2], seq_lengths[2])
			
		# all other cases
		else:
			train_data = all_data[0]
			valid_data = all_data[1]
			test_data = all_data[2]
			
		return config, eval_config, data, train_data, valid_data, test_data, (TRAIN, VALID, TEST)
			
	


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
	eval_config['num_steps'] = 1 # and number of steps

	try:
		os.makedirs(config['save_path'])
	except OSError:
		pass

	final_model = config['save_path'] + os.path.basename(os.path.normpath(config['save_path'])) + '.final'

	if os.path.isfile(final_model) and TRAIN:
		raise OSError("{0} already exists. If you want to re-train the model, remove the model file and its checkpoints.".format(
			final_model))

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
	
		initializer = tf.random_uniform_initializer(minval=-config['init_scale'], maxval=config['init_scale'])

		if TRAIN:

			reuseOrNot = True # valid and test models: reuse the graph

			print('Create training model...')
			with tf.name_scope("Train"):
				with tf.variable_scope("Model", reuse=None, initializer=initializer):
					train_lm = lm.lm(config, is_training=True, reuse=False)

		else:
			reuseOrNot = None
			
		if VALID:
			print('Create validation model...')
			with tf.name_scope("Valid"):
				with tf.variable_scope("Model", reuse=reuseOrNot, initializer=initializer):
					valid_lm = lm.lm(config, is_training=False, reuse=reuseOrNot)

			if reuseOrNot == None:
				reuseOrNot = True
				

		if TEST:
		
			print('Create testing model...')
			with tf.name_scope("Test"):
				with tf.variable_scope("Model", reuse=reuseOrNot, initializer=initializer):
					test_lm = lm.lm(eval_config, is_training=False, reuse=reuseOrNot)


		sv = tf.train.Supervisor(logdir=config['save_path'])

		with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

			if TRAIN and VALID:
	
				# create a trainer object based on config file
				class_name = 'trainer.{0}'.format(config['trainer'])
				train_obj = eval(class_name)(sv, session, config, train_lm, valid_lm, data, train_data, valid_data)
				
				# train + validate the model
				train_obj.train()

			if VALID and not TRAIN:

				validator = run_epoch.run_epoch(session, valid_lm, data, valid_data, eval_op=None, test=False, valid=True)

				valid_perplexity = validator()
				print('Valid Perplexity: {0}'.format(valid_perplexity))

			if TEST:

				# n-best rescoring
				if 'rescore' in config:
					print('Start rescoring.')

					tester = run_epoch.rescore(session, test_lm, data, test_data, eval_op=None, test=True)
	
					len_test = len(test_data)
					counter = 0
					for line in test_data:
						tester(line)
						counter += 1
						if counter % 100 == 0:
							print(counter)
					
					print('Done rescoring.')

				# test the model 
				else:
						
					tester = run_epoch.run_epoch(session, test_lm, data, test_data, eval_op=None, test=True)	
					
					test_perplexity = tester()
					
					print('Test Perplexity: {0}'.format(test_perplexity))

if __name__ == "__main__":
	tf.app.run()
