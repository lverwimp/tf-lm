This directory contains examples of configuration files that are used as input for the scripts.

The pathnames in the config files should be adapted to your local pathnames.

Options in the configuration file:

Obligatory:
* **name**: prefix for the name of the final model
* **log**: name for the log file
* **save_path**: path where the checkpoint files will be saved
* **data_path**: path where the training/validation/test sets are located, should contain files named 'train.txt', 'valid.txt' and 'test.txt'
* **vocab_size**: size of the vocabulary
* **vocab**: can be switched on if you want to train with a smaller vocabulary than in the original data (e.g. if you want to reduce the vocabulary of Penn Treebank to 5k instead of 10k, use '5' as value, replace all words in the data with the <UNK> symbol and rename the files like this: train.txt -> train_5k-unk.txt)
* **layer**: type of cell (only LSTM implemented so far, but can be easily changed)
* **num_layers**: number of hidden layers
* **word_size**: size of the word embedding = size of the hidden layer
* **num_steps**: number of steps used for enrolling when training with backpropagation through time
* **init_scale**: the weights of the model will be randomly initialized, with a uniform distribution and values between -init_scale and init_scale
* **learning_rate**: learning rate of the model
* **max_grad_norm**: clip the gradients if their norm exceeds max_grad_norm
* **batch_size**: size of the mini-batch used in training
* **max_epoch**: determines when the learning rate will start decaying
* **max_max_epoch**: after max_max_epoch epochs, training will be stopped (in case of early stopping, it might be stopped earlier)
* **dropout**: probability at which neurons are dropped (e.g. 0.75: 75% of neurons are dropped)
* **lr_decay**: factor that determines the learning rate decay (the smaller lr_decay, the faster the decay)
* **forget_bias**: initial bias of the forget gate in the LSTM cell
* **optimizer**: type of optimizer (stochastic gradient descent (sgd) or adam)

Optional:
* **per_sentence**: by default, the network trains on batches that contain parts of sentences/multiple sentences; if this option is set, each sentence individually is processed (padded until the length of the longest sentence in the data)
!!! do not compare perplexities of language models trained per sentence with perplexities of language models trained on batches (the padding symbols seriously decrease the perplexity because they are easy to predict) !!!
* **early_stop**: e.g. 2; if a value for early_stop is given, training will be stopped if the validation perplexity has not improved compared to the last 2 epochs
* **nbest**: file with (n-best) hypotheses that should be rescored (the model should already be trained and be specified as 'lm'); can only be used with sentence-level language models!
* **lm**: trained model that can be used for n-best rescoring
* **result**: file in which the results for n-best rescoring will be written


