#! /bin/bash
# splits large data file in pieces

train=$1
dir_split=$2
valid=$3
# test=$4 #uncomment if you have a test set

# default = 100k lines per file
numlines="100000"

base_train=`basename $train`

echo $base_train

mkdir -p $dir_split

# split in pieces
split -l $numlines -d $train $dir_split/train_

cd $dir_split

for f in train_* ; do
	# make separate directory and move file to new directory
	mkdir ${f#train_}
	mv $f "${f#train_}/${base_train}"
	
	# add symbolic link to validation set
	cd ${f#train_}
	ln -s $valid
	#ln -s $test #uncomment if you have a test set
	cd ..
	
done


