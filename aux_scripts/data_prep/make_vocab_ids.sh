#! /bin/bash

training_text=$1
size=$2
vocab=$3

# make unigram model
nice -19 ngram-count -text $training_text -write tmp -write-order 1 -order 1 -unk

# sort based on frequency
nice -19 sort tmp -k 2 -n -r > tmp2

# take $size most frequent words + add ids
nice -19 head tmp2 -n $size | awk  '{print $1,NR}' > $vocab

rm -f tmp tmp2
