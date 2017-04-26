This directory contains examples of configuration files that are used as input for the scripts.

The pathnames in the config files should be adapted to your local pathnames.

Options in the configuration file:

# Files/directories

* **log** (optional): name for the log file
  * change LOG_DIR in main.py to a directory of your choice
  * if no log is specified, the log file will be LOG_DIR/<basename of save_path>
* **save_path**: path where the model will be saved
* **data_path**: path where the training/validation/test sets are located, should contain files named 'train.txt', 'valid.txt' and 'test.txt' (or train_<size>k-unk.txt etc. if you are using a restricted vocabulary)

# Size of the model

* **vocab**: can be switched on if you want to train with a smaller vocabulary than in the original data
  * False if you want to use the full vocabulary (data files are named 'train.txt' etc.) 
  * size of vocab divided by 1000 (e.g. if you want to reduce the vocabulary of Penn Treebank to 5k instead of 10k, use '5' as value, replace all words in the data with the <UNK> symbol and rename the files like this: train.txt -> train_5k-unk.txt)
* **layer**: type of cell (only LSTM implemented so far, but can be easily changed)
* **num_layers**: number of hidden layers
* **size**: size of the word (or character) embedding = size of the hidden layer
* **batch_size**: size of the mini-batch used in training
* **num_steps**: number of steps used for enrolling for training with backpropagation through time

# Initialization

* **init_scale**: the weights of the model will be randomly initialized, with a uniform distribution and values between -init_scale and init_scale
* **forget_bias**: initial bias of the forget gate in the LSTM cell

# Optimization

* **optimizer**: type of optimizer (stochastic gradient descent (sgd) or adam)
* **softmax**: full or sampled

# Regularization

* **max_grad_norm**: clip the gradients if their norm exceeds max_grad_norm
* **dropout**: probability at which neurons are dropped (e.g. 0.75: 75% of neurons are dropped)

# Training schedules

* **trainer**:
  * trainer: Fixed training schedule with a fixed learning rate decay schedule. Used in combination with
    * **learning_rate**: initial learning rate
    * **max_epoch**: number of epochs during which the initial learning rate should be used
    * **lr_decay**: determines how fast the learning rate decays
    * **max_max_epoch**: the total number of epochs to train
  * earlyStopping: Early stopping based on comparison with previous *n* validation perplexities, but with fixed learning rate decay schedule. Used in combination with:
    * same parameters as for trainer
    * **early_stop** = *n*: compare with *n* previous epochs, if the validation perplexity has not improved for *n* times, stop training
  * validHalve: Early stopping based on comparison of previous validation perplexity, learning rate is halved each time no improvement is seen (until it has been halved *n* times). Used in combination with:
    * **learning_rate**: initial learning rate
    * **valid_halve** = *n*: the number of times the learning rate can be halved before training is stopped
    
# Type of data

* **per_sentence** (optional): by default, the network trains on batches that contain parts of sentences/multiple sentences; if this option is set to True, each sentence individually is processed (padded until the length of the longest sentence in the data)
* **char** (optional): by default, the data is read as words, but if this option is set to True, the model will train on character level

# Rescoring

* **rescore** (optional): the data file that should be rescored, containing 1 hypothesis per line
* **result** (optional): file in which the results for n-best rescoring will be written



