# tf-languagemodel
**New version** of the scripts: different structure + now compatible with TensorFlow v1.1. If you want to continue using the old version, go to branch v0.

This repository contains scripts for recurrent neural network language modeling with TensorFlow.

For now, scripts for training a word- or character-level LSTM language model and rescoring n-best hypotheses are included.

# Installation

* Python version used: 2.7.5. 
* Install [TensorFlow](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup). These scripts are compatible with version 1.1.
* Modify the config files in config/: change the pathnames and optionally the parameters.

# Options

* Word-level and character-level language models.
* Early stopping based on comparison of previous *n* validation perplexities.
* Optimizers: stochastic gradient descent and adam (with learning rate decay).
* Train on mini-batches (that can contain multiple sentences/parts of sentences) or on padded sentences.
* Other features from the [TensorFlow tutorial](https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html).

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
* lm.py: class for language model
* lm_data.py: contains several classes for handling the language model data in different ways (as words, as characters, sentence-level or not)
* run_epoch.py: calls lm_data to get batches of data, feeds the batches to the language model and calculates the probability/perplexity
* trainer.py: several classes for training the model: with a fixed schedule (decaying learning rate), early stopping with (fixed) decaying learning rate or early stopping with a learning rate adapted depending on the validation perplexity
* writer.py: for writing to multiple output streams


# Example commands

For these examples, you can download the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42) or use you own dataset. The data should be divided in a train.txt, valid.txt and test.txt and the correct data path should be specified in the configuration file ('data_path').

Train and evaulate a small language model on Penn Treebank (sentence-level):

python main.py --config ../config/ptb_word_small_sentence.config

# Contact

If you have any questions, mail to lyan.verwimp@esat.kuleuven.be.
