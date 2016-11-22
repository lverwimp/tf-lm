#! /usr/bin/env python

class writer: # NEW
	'''Copied from https://groups.google.com/forum/#!topic/comp.lang.python/0lqfVgjkc68'''
	def __init__(self, *writers):
		self.writers = writers

	def write(self, text):
		for w in self.writers:
			w.write(text)
