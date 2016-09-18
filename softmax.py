import theano
import theano.tensor as T 
import numpy as np
import cPickle, gzip
from util import *

def floatX(X):
	return np.asarray(X,dtype = theano.config.floatX)

def init_weights(shape):
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X,w):
	return T.nnet.softmax(T.dot(X,w))

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

n_epochs = 100

X = T.fmatrix()
Y = T.fmatrix()
#load data
x_train,y_train,x_valid,y_valid,x_test,y_test = load_mnist()

#set one_hot label
y_train = one_hot(y_train,10)
y_test = one_hot(y_test,10)
w = init_weights((784,10))

predy_x = model(X,w)
y_pred = T.argmax(predy_x,axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(predy_x,Y))
gradient = T.grad(cost = cost, wrt = w)
update = [(w,w-gradient*0.05)]

train = theano.function(inputs = [X,Y] , outputs =cost, updates = update, allow_input_downcast = True)
predict = theano.function(inputs = [X], outputs = y_pred ,allow_input_downcast = True)

for i in range(n_epochs):
	cost = train(x_train,y_train)
	print cost

print "accuracy = ",100*np.mean(np.argmax(y_test,axis=1)==predict(x_test)),"%"