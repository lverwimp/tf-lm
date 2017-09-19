#! /usr/bin/env python
# creates character feature files

import sys

f = sys.argv[1] # data file containing words
prefix = str(sys.argv[2]) # prefix for data file with characters
it = int(sys.argv[3]) # position in the word that should be written to file, starting from 0
invert = str(sys.argv[4]) # 'normal' or 'invert': order of the characters

print('invert: {0}'.format(invert))

if invert != 'normal' and invert != 'invert':
	raise IOError("Specify in which order the characters should be added (4th argument): \
		starting from the beginning of the word ('normal') or from the ending of the \
		word ('invert').")

PADDING_SYMBOL = '#' # change this if this symbol already occurs in your dataset

out_f = open('{0}{1}'.format(prefix,it), 'w')

for line in open(f, 'r'):
	l = line.split(' ')
	for word in l[:-1]: # for each word in the sentence except the last one
		if word != '' and word != ' ' and word != '\n': # skip empty elements and newlines
			word_l = list(word)

			if invert == 'invert':
				word_l = word_l[::-1]

			# if e.g. length of word = 3 and it = 4 --> PADDING_SYMBOL
			if len(word_l) < it+1:
				out_f.write('{0} '.format(PADDING_SYMBOL))

			# otherwise, print correct letter
			else:
				out_f.write(word_l[it] + ' ')

	# last word: do the same, except add a newline instead of a space
	word_l = list(l[-1].strip('\n'))
	if ''.join(word_l) != '' and ''.join(word_l) != ' ':

		if invert == 'invert':
			word_l = word_l[::-1]

		# if e.g. length of word = 3 and it = 4 --> hashtag
		if len(word_l) < it+1:
			out_f.write('{0}\n'.format(PADDING_SYMBOL))

		# otherwise, print correct letter
		else:
			out_f.write(word_l[it] + '\n') # word is NOT inverted

	else:
		out_f.write('\n')
out_f.close()
