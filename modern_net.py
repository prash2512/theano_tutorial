import theano
import theano.tensor as T 
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle, gzip
from util import *

srng = RandomStreams()

def floatX(X):
	return np.asarray(X,dtype = theano.config.floatX)

def init_weights(shape):
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def relu(X):
	return T.maximum(X,0.)

def softmax(X):
	e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
	return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def sgd(cost,params,lr = 0.05):
	grads = T.grad(cost = cost,wrt = params)
	updates = []
	for p,g in zip(params,grads):
		updates.append([p,p-g*lr])
	return updates

def dropout(X,p = 0.):
	if p>0:
		retain = 1-p
		X = X*srng.binomial(X.shape,p= retain,dtype=theano.config.floatX)
		X = X/retain
	return X

def model(X,w_h,w_o,prob_hidden,prob_input):
	X = dropout(X,prob_input)
	h = relu(T.dot(X,w_h))
	h_p = dropout(h,prob_hidden)
	pred = softmax(T.dot(h,w_o))
	return pred

n_epochs = 200

X = T.fmatrix()
Y = T.fmatrix()
#load data
x_train,y_train,x_valid,y_valid,x_test,y_test = load_mnist()

#set one_hot label
y_train = one_hot(y_train,10)
y_test = one_hot(y_test,10)

w_h = init_weights((784,625))
w_o = init_weights((625,10))

predy_x = model(X,w_h,w_o,0.5,0.2)
y_pred = T.argmax(predy_x,axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(predy_x,Y))
params = [w_h,w_o]
updates = sgd(cost,params)

train = theano.function(inputs = [X,Y] , outputs =cost, updates = updates, allow_input_downcast = True)
predict = theano.function(inputs = [X], outputs = y_pred ,allow_input_downcast = True)

for i in range(n_epochs):
	for start, end in zip(range(0, len(x_train), 128), range(128, len(x_train), 128)):
        	cost = train(x_train[start:end], y_train[start:end])
	print "accuracy = ",100*np.mean(np.argmax(y_test,axis=1)==predict(x_test)),"%"