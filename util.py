import theano
import theano.tensor as T 
import numpy as np
import cPickle, gzip

def load_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	train,valid,test = cPickle.load(f)
	x_tr,y_tr = train
	x_valid,y_valid = valid
	x_test,y_test = test
	return x_tr,y_tr,x_valid,y_valid,x_test,y_test

