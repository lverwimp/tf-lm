This directory contains examples of configuration files that are used as input for the scripts.

The pathnames in the config files should be adapted to your local pathnames.

For every option, we specify between brackets whether it is optional and what type of value we expect. E.g. **log** (optional, string) means that 'log' is an optional variable, and if it is specified, its value should be a string.

Options in the configuration file:

# Files/directories

* **data_path** (string): path where the training/validation/test sets are located, should contain files named 'train.txt', 'valid.txt' and 'test.txt' (or train_<size>k-unk.txt etc. if you are using a restricted vocabulary)

* **save_path** (string): path where the model will be saved

* **log** (optional, string): name for the log file
  * change LOG_DIR in main.py to a directory of your choice
  * if no log is specified, the log file will be LOG_DIR/\<basename of save_path\>
  
# Reading data

Default: all data is read at once and kept in memory. If the dataset is too large for this, use:

* **stream_data** (optional, True): Read data sentence by sentence. This assumes that the data file contains one sentence per line and that batching is sentence-level!
  * **read_vocab_from_file** (optional, string): To speed up more, read the vocabulary from file. The vocabulary file should contain 2 columns: the words and their ids. The *data_path* should also contain a file *max_sentence_length* with the length of the longest sentence (will be the length of every batch).

# Size of the model

* **vocab** (False/integer): can be switched on if you want to train with a smaller vocabulary than in the original data
  * False if you want to use the full vocabulary (data files are named 'train.txt' etc.) 
  * size of vocab divided by 1000 (e.g. if you want to reduce the vocabulary of Penn Treebank to 5k instead of 10k, use '5' as value, replace all words in the data with the \<unk\> symbol and rename the files like this: train.txt -> train_5k-unk.txt)
  
  !!! by default, out-of-vocabulary words are assumed to be replaced by \<unk\>, if this is for example \<UNK\> in your dataset, adapt the \_\_init\_\_ function in the lm_data.LMData class (self.unk and self.replace_unk).
* **layer** (string): type of cell (only LSTM implemented so far, but can be easily changed)
* **num_layers** (integer): number of hidden layers
* **size** (integer): size of the word (or character) embedding = size of the hidden layer
* **batch_size** (integer): size of the mini-batch used in training
* **num_steps** (integer): number of steps used for enrolling for training with backpropagation through time

# Type of model

Default = unidirectional.

* **bidirectional** (optional, True): if set to True, the model is trained as a bidirectional LM

# Initialization

* **init_scale** (float): the weights of the model will be randomly initialized, with a uniform distribution and values between -init_scale and init_scale

* **forget_bias** (float/integer): initial bias of the forget gate in the LSTM cell

# Optimization

* **optimizer** (string): type of optimizer (stochastic gradient descent ('sgd'), 'adam' or 'adagrad')

* **softmax** (string): 'full' or 'sampled'

# Regularization

* **max_grad_norm** (integer): clip the gradients if their norm exceeds max_grad_norm

* **dropout** (float): probability at which neurons are dropped (e.g. 0.75: 75% of neurons are dropped)

# Training schedules

* **trainer** (string):
  * 'trainer': Fixed training schedule with (optionally) a fixed learning rate decay schedule. Used in combination with
    * **learning_rate** (float): initial learning rate
    * **max_epoch** (integer): number of epochs during which the initial learning rate should be used
    * **lr_decay** (float): determines how fast the learning rate decays
    * **max_max_epoch** (integer): the total number of epochs to train
  * 'earlyStopping': Early stopping based on comparison with previous *n* validation perplexities, but with (optionally) fixed learning rate decay schedule. Used in combination with:
    * same parameters as for trainer
    * **early_stop** (integer) = *n*: compare with *n* previous epochs, if the validation perplexity has not improved for *n* times, stop training
    
# Input/output unit 

Default: input and output unit = word.

* **char** (optional, True): by default, the data is read as words, but if this option is set to 'True', the model will train on character level (both input and output)

* **char_ngram** (optional, integer): specify *n*, input consists of a vector with counts for all *n*-grams in the current word, output is still words

  * **ngram_cutoff** (optional, integer): to reduce the input vocabulary size, set a frequency cutoff for the character n-grams
  
  * **capital** (optional, True): another option to reduce the input vocabulary: map all characters to lowercase and add a special symbol to mark the frequency of uppercase characters in the word
  
  * **skipgram** (optional, integer): use skipgrams instead of n-grams, where the integer value of 'skipgram' specifies how many characters should be skipped
  
  * **add_word** (optional, True): if you want to feed a concatenation of character n-gram input and the word embedding to the LSTM
  
    * **word_size** (integer): size of the LSTM that should be assigned to the word embedding; the size assigned to character n-gram input is size - word_size
    * **input_vocab_size** (integer): size of word input vocabulary
    * **input_vocab** (optional, integer): size of word input vocab divided by 1000, in case you want to read the vocabulary from a dataset with a reduced vocabulary (e.g. train_1k-unk.txt)
    
* **word_char_concat** (optional, True): inputs consists of the concatenation of the word embedding and embeddingss of (part of) the characters occurring in the word, output is still words. Character feature files can be generated with aux_scripts/make_char_feat_files.py. Used in combination with:

  * **num_char** (integer): number of characters to add (if the current word is shorter than *num_char*, padding symbols are used; if it is longer than *num_char*, only part of the characters in the word are added)
  
  * **char_size** (integer): size assigned to each character embedding (size for the word embedding = *size* - *num_char* * *char_size*)
  
  * **order** (string): 
    * *begin_first*: start adding characters from the beginning of the word (e.g. 4 characters from 'pineapple': p, i, n, e)
    * *end_first*: start adding characters from the end of the word (e.g. 4 characters from 'pineapple': e, l, p, p)
    * *both*: add *num_char* characters starting from the beginning and *num_char* starting from the end (e.g. 4 characters from 'pineapple': p, i, n, e; e, l, p, p)

# Batching

Default: each batch is of length *num_steps* and may contain multiple (parts of) sentences ('discourse').

* **per_sentence** (optional, True): if this option is set to 'True', each sentence individually is processed and padded until the length of the longest sentence in the data ('sentence').

* **padding_symbol** (optional, string): the default padding symbol is '@'; if your dataset already contains this symbol, choose another one that doesn't occur in the data.

* **not_trained_with_padding** (optional, True): if a 'discourse' model is used for rescoring, generating a -debug 2 file or predicting the next words, this option should be added to make sure the \<unk\> symbol is used as padding symbol.

# Testing options

For all 3 options below: if set to 'True', the default test set will be evaluated, if a file is specified, that file will be evaluated.

* **valid_as_test** (optional, True): calculate the validation perplexity in the 'testing mode': with a batch_size and num_steps of 1
* **other_test** (optional, string): calculate the perplexity of a new dataset

* **rescore** (optional, string): rescore a set of sentences/hypotheses and write their log probabilities to file

* **debug2** (optional, string): generate a file similar to the output of SRILM's -debug 2 option, that can be used for calculating interpolation weights

* **predict_next** (optional, string): generate the most likely sequence for every prefix in a given file
  * **max_num_predictions** (optional, integer): generate only *max_num_predictions* (default = until <eos> is predicted or until 100 words are generated)
  * **sample_multinomial** (optional, True): do not generate the most likely sequence, but sample from a multinomial distribution specified by the softmax probabilities
 * **interactive** (optional, True): do not read seed word(s) from file but ask the user for input

All 3 options should be combined with:
* **result** (optional, string): results file



