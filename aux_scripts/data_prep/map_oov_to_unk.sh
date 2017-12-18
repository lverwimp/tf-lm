#! /bin/bash
# map words not in the vocab to <UNK>
# + remove <s> and </s> added by SRILM

text=$1
vocab=$2
output=$3
unk=$4

replace-unk-words vocab=$vocab $text | sed "s/<unk>/${unk}/g" | sed "s/<s> //" | sed "s/ <\/s>//" > $output

