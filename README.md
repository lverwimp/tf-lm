# tf-languagemodel
**New version** of the scripts: different structure + now compatible with TensorFlow v1.1. If you want to continue using the old version, go to branch v0.

This repository contains scripts for recurrent neural network language modeling with TensorFlow (based on the [TensorFlow tutorial] (https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html)).

For now, scripts for training a word- or character-level LSTM language model and rescoring n-best hypotheses are included.

# Installation

* Python version used: 2.7.5. 
* Install [TensorFlow] (https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup). These scripts are compatible with version 1.1.
* Modify the config files in config/: change the pathnames and optionally the parameters.

# Options

* Word-level and character-level language models.
* Early stopping based on comparison of previous *n* validation perplexities.
* Optimizers: stochastic gradient descent and adam (with learning rate decay).
* Train on mini-batches (that can contain multiple sentences/parts of sentences) or on padded sentences.
* Other features from the [TensorFlow tutorial] (https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html).

# Code overview

Main script:

* main.py:
  * --config: configuration file specifying all options
  * --train: boolean marking whether you want to train the model or not (default = True)
  * --valid: boolean marking whether you want to validate the model or not (default = True)
  * --test: boolean marking whether you want to test the model or not (default = True)
  * --device: use with 'cpu' if you want to explicitly run on cpu, otherwise it will try to run on gpu
  

Other scripts:

* configuration.py: handles configuration files
* reader.py: reads the text data and generates mini-batches for training/testing
* writer.py: for writing to multiple output streams

# Example commands

For these examples, you can download the [Penn Treebank] (https://catalog.ldc.upenn.edu/ldc99t42) or use you own dataset. The data should be divided in a train.txt, valid.txt and test.txt and the correct data path should be specified in the configuration file.

Train a small language model on Penn Treebank (sentence-level):

python train_lm.py --config ../config/ptb_word_small_sentence.config

Rescore a list of (N-best) hypotheses:

python lm_rescore_nbest.py --config ../config/ptb_word_small_sentence.config.nbest_test

Train + rescore with the trained model:

./train+rescore.sh ../config/ptb_word_small_sentence.config ../nbest/test

# Contact

If you have any questions, mail to lyan.verwimp@esat.kuleuven.be.
