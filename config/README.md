This directory contains examples of configuration files that are used as input for the scripts.

The pathnames in the config files should be adapted to your local pathnames.

Options in the configuration file:

# Files/directories

* **data_path**: path where the training/validation/test sets are located, should contain files named 'train.txt', 'valid.txt' and 'test.txt' (or train_<size>k-unk.txt etc. if you are using a restricted vocabulary)

* **save_path**: path where the model will be saved

* **log** (optional): name for the log file
  * change LOG_DIR in main.py to a directory of your choice
  * if no log is specified, the log file will be LOG_DIR/\<basename of save_path\>

# Size of the model

* **vocab**: can be switched on if you want to train with a smaller vocabulary than in the original data
  * False if you want to use the full vocabulary (data files are named 'train.txt' etc.) 
  * size of vocab divided by 1000 (e.g. if you want to reduce the vocabulary of Penn Treebank to 5k instead of 10k, use '5' as value, replace all words in the data with the \<unk\> symbol and rename the files like this: train.txt -> train_5k-unk.txt)
  
  !!! by default, out-of-vocabulary words are assumed to be replaced by \<unk\>, if this is for example \<UNK\> in your dataset, adapt the \_\_init\_\_ function in the lm_data.LMData class (self.unk and self.replace_unk).
* **layer**: type of cell (only LSTM implemented so far, but can be easily changed)
* **num_layers**: number of hidden layers
* **size**: size of the word (or character) embedding = size of the hidden layer
* **batch_size**: size of the mini-batch used in training
* **num_steps**: number of steps used for enrolling for training with backpropagation through time

# Initialization

* **init_scale**: the weights of the model will be randomly initialized, with a uniform distribution and values between -init_scale and init_scale

* **forget_bias**: initial bias of the forget gate in the LSTM cell

# Optimization

* **optimizer**: type of optimizer (stochastic gradient descent ('sgd'), 'adam' or 'adagrad')

* **softmax**: 'full' or 'sampled'

# Regularization

* **max_grad_norm**: clip the gradients if their norm exceeds max_grad_norm

* **dropout**: probability at which neurons are dropped (e.g. 0.75: 75% of neurons are dropped)

# Training schedules

* **trainer**:
  * 'trainer': Fixed training schedule with (optionally) a fixed learning rate decay schedule. Used in combination with
    * **learning_rate**: initial learning rate
    * **max_epoch**: number of epochs during which the initial learning rate should be used
    * **lr_decay**: determines how fast the learning rate decays
    * **max_max_epoch**: the total number of epochs to train
  * 'earlyStopping': Early stopping based on comparison with previous *n* validation perplexities, but with (optionally) fixed learning rate decay schedule. Used in combination with:
    * same parameters as for trainer
    * **early_stop** = *n*: compare with *n* previous epochs, if the validation perplexity has not improved for *n* times, stop training
    
# Input/output unit 

Default: input and output unit = word.

* **char** (optional): by default, the data is read as words, but if this option is set to 'True', the model will train on character level (both input and output)
* **char_ngram**: specify *n*, input consists of a vector with counts for all *n*-grams in the current word, output is still words
* **word_char_concat**: inputs consists of the concatenation of the word embedding and embeddingss of (part of) the characters occurring in the word, outpt is still words. Used in combination with:
  * **num_char**: number of characters to add (if the current word is shorter than *num_char*, padding symbols are used; if it is longer than *num_char*, only part of the characters in the word are added)
  * **char_size**: size assigned to each character embedding (size for the word embedding = *size* - *num_char* * *char_size*)
  * **order**: 
    * *begin_first*: start adding characters from the beginning of the word (e.g. 4 characters from 'pineapple': p, i, n, e)
    * *end_first*: start adding characters from the end of the word (e.g. 4 characters from 'pineapple': e, l, p, p)
    * *both*: add *num_char* characters starting from the beginning and *num_char* starting from the end (e.g. 4 characters from 'pineapple': p, i, n, e; e, l, p, p)

# Batching

Default: each batch is of length *num_steps* and may contain multiple (parts of) sentences.

* **per_sentence** (optional): if this option is set to 'True', each sentence individually is processed (padded until the length of the longest sentence in the data)

# Testing options

* **rescore** (optional): the data file that should be rescored, containing 1 hypothesis per line

* **result** (optional): file in which the results for n-best rescoring will be written



