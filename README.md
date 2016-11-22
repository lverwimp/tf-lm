# tf-languagemodel
Language modeling scripts based on TensorFlow.

This repository contains scripts for recurrent neural network language modeling with TensorFlow (based on the [TensorFlow tutorial] (https://www.tensorflow.org/versions/r0.11/tutorials/recurrent/index.html)).

For now, scripts for training a word-level LSTM language model and rescoring n-best hypotheses are included.

# Installation

* Install [TensorFlow] (https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup)
* Modify the config files in config/: change the pathnames and optionally the parameters.

# Overview

Main scripts:

* word_lm.py: train a language model
* word_lm_rescore_nbest.py: use a trained language model for n-best rescoring
* train+rescore.sh: combines the above 2 scripts: trains a language model and then uses the same model for rescoring

Other scripts:

* configuration.py: handles configuration files
* reader.py: reads the text data and generates mini-batches for training/testing\
* writer.py: for writing to multiple output streams

# Example commands

Train a small language model (training per sentence):

python word_lm.py --config ../config/ptb_word_small_sentence.config

Rescore a list of hypotheses:

python word_lm_rescore_nbest.py --config ../config/ptb_word_small_sentence.config.nbest_test

Train + rescore:

./train+rescore.sh ../config/ptb_word_small_sentence.config ../nbest/test
