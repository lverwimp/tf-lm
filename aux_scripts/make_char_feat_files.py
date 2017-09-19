#! /usr/bin/env python
# creates character feature files

import sys, argparse

PADDING_SYMBOL = '#' # change this if this symbol already occurs in your dataset

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('file', help='data file containing words')
	parser.add_argument('prefix', help='prefix for data file with characters')
	parser.add_argument('position', type=int, help='position in the word that should be written to file, starting from 0')
	parser.add_argument('--invert', help='invert the order of the characters', action='store_true')

	args = parser.parse_args()

	if args.invert:
		invert = True
	else:
		invert = False

	out_f = open('{0}{1}'.format(args.prefix, args.position), 'w')

	for line in open(args.file, 'r'):
		l = line.split(' ')
		for word in l[:-1]: # for each word in the sentence except the last one
			if word != '' and word != ' ' and word != '\n': # skip empty elements and newlines
				word_l = list(word)

				if invert:
					word_l = word_l[::-1]

				# if e.g. length of word = 3 and it = 4 --> PADDING_SYMBOL
				if len(word_l) < args.position+1:
					out_f.write('{0} '.format(PADDING_SYMBOL))

				# otherwise, print correct letter
				else:
					out_f.write(word_l[args.position] + ' ')

		# last word: do the same, except add a newline instead of a space
		word_l = list(l[-1].strip('\n'))
		if ''.join(word_l) != '' and ''.join(word_l) != ' ':

			if invert:
				word_l = word_l[::-1]

			# if e.g. length of word = 3 and it = 4 --> hashtag
			if len(word_l) < args.position+1:
				out_f.write('{0}\n'.format(PADDING_SYMBOL))

			# otherwise, print correct letter
			else:
				out_f.write(word_l[args.position] + '\n') # word is NOT inverted

		else:
			out_f.write('\n')
	out_f.close()
