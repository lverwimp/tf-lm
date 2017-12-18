#! /usr/bin/bash
# use reduced vocabulary: make vocabulary + convert words to <unk> if necessary

# default = lowercase unk
unk="<unk>"

size=$1
train=$2
valid=$3
# test=$4 #uncomment if you have a test set

train_base="${train%\.txt}"
vocab="${train_base}_${size}.wlist"

echo "$vocab"

train_unk="${train_base}_${size}-unk.txt"
valid_unk="${valid%\.txt}_${size}-unk.txt"
# test_unk="${test%\.txt}_${size}-unk.txt" #uncomment if you have a test set

echo "$train_unk $valid_unk"

# make vocabulary of size $size based on training text
echo "Extracting vocabulary to $vocab..."
./make_vocab_ids.sh $train $size $vocab

# map OOV words to unk in training text
echo "Mapping OOV words to $unk..."
./map_oov_to_unk.sh $train $vocab $train_unk $unk

# map OOV words to unk in validation text
./map_oov_to_unk.sh $valid $vocab $valid_unk $unk

# map OOV words to unk in test text
#./map_oov_to_unk.sh $test $vocab $test_unk $unk #uncomment if you have a test set
