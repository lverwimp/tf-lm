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
* **max_epoch**: 
* **max_max_epoch**
* **dropout**:
* **lr_decay**
* **forget_bias**
* **optimizer**

Optional:
* **early_stop**
* **nbest**


