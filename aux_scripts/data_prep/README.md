# Data preparation scripts

These scripts can be used to prepare your data.
!! Some of them require [SRILM] (http://www.speech.sri.com/projects/srilm/).

* make_vocab_ids.sh: creates a vocabulary file with words and indices
	* Ex: ./make_vocab_ids.sh "training_text" "size_of_vocabulary" "vocabulary_name"
* map_oov_to_unk.sh: for a given vocabulary and text file, map all words in the text file not in the vocubulary to an unknown-token
	* Ex: ./map_oov_to_unk.sh "text_file" "vocabular_file" "new_text_file" "unk-token"
* reduce_vocab.sh: construct vocabulary + map words not in vocabulary to unk-token
	* Ex: ./reduce_vocab.sh "vocabulary_size" "training_text" "validation_text"
	* If you also have test set besides a validation set, you can uncomment the lines concerning the test set
	* The default unk-token is <unk>; this can changed in the script
* split_data.sh

