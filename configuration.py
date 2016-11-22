#! /usr/bin/env python
	
def boolify(s):
	'''Copied from http://stackoverflow.com/questions/7019283/automatically-type-cast-parameters-in-python'''
	if s == 'True':
		return True
	if s == 'False':
		return False
	raise ValueError("huh?")

def autoconvert(s):
	'''Copied from http://stackoverflow.com/questions/7019283/automatically-type-cast-parameters-in-python'''
	for fn in (boolify, int, float):
		try:
			return fn(s)
		except ValueError:
			pass
	return s

def get_config(f):
	config = {}
	for line in open(f,'r'):
		l = line.split()
		if len(l) > 1:
			config[l[0]] = autoconvert(l[-1])
	return config
	
