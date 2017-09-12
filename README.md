# tf-languagemodel
**New version** of the scripts: different structure, more options + pre-trained language models.

These scripts are compatible with TensorFlow **v1.1**. 

This repository contains scripts for recurrent neural network language modeling with TensorFlow.

!!! Disclaimer: This project is still under development and not everything has been tested very thoroughly yet.

# Installation and setup

* Python version used: 2.7.5. 
* Install [TensorFlow](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup). These scripts are compatible with version 1.1.
* Modify the config files in config/: change the pathnames and optionally the parameters.
* Modify the log directory in main.py (LOG_DIR=...).

# Options

For more information on how to specify these options in a configuration file, see the README in config/.

* Input units: words, characters, character n-gram or concatenated word and characters [1].
* Train on sentence level ('sentence'), with all sentences padded until the length of the longest sentence in the dataset, or train on batches that may contain multiple sentences ('discourse'). 
  * e.g. across sentence boundaries (default): "owned by \<unk\> & \<unk\> co. was under contract with <unk> to make the cigarette filters \<eos\> the finding probably"
  * e.g. sentence-level: 
    * "\<bos\> the plant which is owned by \<unk\> & \<unk\> co. was under contract with \<unk\> to make the cigarette filters \<eos\> @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @"
    * "\<bos\> the finding probably will support those who argue that the u.s. should regulate the class of asbestos including <\unk\> more \<unk\> than the common kind of asbestos \<unk\> found in most schools and other buildings dr. \<unk\> said \<eos\> @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @"
* Training schedules:
  * Fixed training schedule
  * Early stopping based on comparison with previous *n* validation perplexities
  * Learning rate decay
* Optimizers: stochastic gradient descent, adam, adagrad.
* Full softmax or sampled softmax. 
* Testing options:
  * Perplexity (of the standard validation set or test set: same configuration file as for training, but set --train False and --test False or --valid False respectively)
  * Re-scoring: log probabilities per sentence
  * Predicting the next word(s) given a prefix
  * Generate debugging file similar to SRILM's -debug 2 option: can be used to calculate interpolation weights
* Reading the data all at once or streaming sentence per sentence.
 

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
* lm.py: classes for language models
* lm_data.py: contains several classes for handling the language model data in different ways
* multiple_lm_data.py: class that handles several lm_data classes (for models with concatenated word and character embedding [1])
* run_epoch.py: calls lm_data to get batches of data, feeds the batches to the language model and calculates the probability/perplexity
* trainer.py: classes for training the model
* writer.py: for writing to multiple output streams

# Possible combinations

| Input unit | Batching | Model | Testing options | Example (arguments only)
| --- | --- | --- | --- | --- |
| Word | Discourse | Unidirectional | Perplexity | --config ../config/en-ptb_word_discourse.config (--train False --valid False)
| Word | Discourse | Unidirectional | Rescore | --config ../config/en-ptb_word_discourse_rescore.config
| Word | Discourse | Unidirectional | Predict next word(s) | --config ../config/en-ptb_word_discourse_predict.config
| Word | Discourse | Unidirectional | Generate debug file | --config ../config/en-ptb_word_discourse_debug2.config
| Word | Sentence | Unidirectional | Perplexity | --config ../config/en-ptb_word_sentence.config (--train False --valid False)
| Word | Sentence | Unidirectional | Rescore | --config ../config/en-ptb_word_sentence_rescore.config
| Word | Discourse | Unidirectional | Predict next word(s) | --config ../config/en-ptb_word_sentence_predict.config
| Word | Sentence | Unidirectional | Generate debug file | --config ../config/en-ptb_word_sentence_debug2.config
| Word | Sentence | Bidirectional | Perplexity | --config ../config/en-ptb_word_sentence_bidir.config (--train False --valid False)
| Character | Discourse | Unidirectional | Perplexity | --config en-ptb_char_discourse.config (--train False --valid False)
| Character | Discourse | Unidirectional | Rescore |
| Character | Discourse | Unidirectional | Predict next word(s) |
| Word-Character | Discourse | Unidirectional | Perplexity |
| Character n-gram | Discourse | Unidirectional | Perplexity |



# Example commands

For these examples, you can download the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42), [WikiText](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) or use you own dataset. The data should be divided in a train.txt, valid.txt and test.txt and the correct data path should be specified in the configuration file ('data_path').

Train and evaluate a word-level language model on Penn Treebank:

python main.py --config ../config/en-ptb_word_discourse.config

# Contact

If you have any questions, mail to lyan.verwimp [at] esat.kuleuven.be.

[1] Verwimp, L., Pelemans, J., hamme, H. V., and Wambacq, P. (2017). Character-Word LSTM Language Models. *Proceedings of the European Chapter of the Association for Computational Linguistics (EACL)*, pages 417–427.
