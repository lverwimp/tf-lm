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
* Train on sentence-level (with all sentences padded until the length of the longest sentence in the dataset) or train on batches that may contain multiple sentences. 
  * e.g. across sentence boundaries (default): "owned by \<unk\> & \<unk\> co. was under contract with <unk> to make the cigarette filters \<eos\> the finding probably"
  * e.g. sentence-level: 
    * "\<bos\> the plant which is owned by \<unk\> & \<unk\> co. was under contract with \<unk\> to make the cigarette filters \<eos\> @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @"
    * "\<bos\> the finding probably will support those who argue that the u.s. should regulate the class of asbestos including <\unk\> more \<unk\> than the common kind of asbestos \<unk\> found in most schools and other buildings dr. \<unk\> said \<eos\> @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @"
* Training schedules:
  * Fixed training schedule with a fixed learning rate decay schedule
  * Early stopping based on comparison with previous *n* validation perplexities, but with fixed learning rate decay schedule
  * Early stopping based on comparison of previous validation perplexity, learning rate is halved each time no improvement is seen (until it has been halved *n* times)
* Optimizers: stochastic gradient descent, adam, adagrad.
* Full softmax or sampled softmax. 

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

Train and evualate a small language model on Penn Treebank:

python main.py --config ../config/ptb_word_small_dropout0.75.config

This should give you a train perplexity of (approximately) 65, validation perplexity of 101 and a test perplexity of 97.

# Contact

If you have any questions, mail to lyan.verwimp [at] esat.kuleuven.be.
