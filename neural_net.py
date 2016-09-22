import theano
import theano.tensor as T 
import numpy as np
import cPickle, gzip
from util import *

#optimizers
def sgd(cost,params,lr = 0.05):
	grads = T.grad(cost = cost,wrt = params)
	updates = []
	for p,g in zip(params,grads):
		updates.append([p,p-g*lr])
	return updates

def rmsprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def adam(cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
	updates = []
	grads = T.grad(cost, params)
	i = theano.shared(floatX(0.))
	i_t = i + 1.
	fix1 = 1. - (1. - b1)**i_t
	fix2 = 1. - (1. - b2)**i_t
	lr_t = lr * (T.sqrt(fix2) / fix1)
	for p, g in zip(params, grads):
	    m = theano.shared(p.get_value() * 0.)
	    v = theano.shared(p.get_value() * 0.)
	    m_t = (b1 * g) + ((1. - b1) * m)
	    v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
	    g_t = m_t / (T.sqrt(v_t) + e)
	    p_t = p - (lr_t * g_t)
	    updates.append((m, m_t))
	    updates.append((v, v_t))
	    updates.append((p, p_t))
	updates.append((i, i_t))
	return updates

def floatX(X):
	return np.asarray(X,dtype = theano.config.floatX)

def init_weights(shape):
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X,w_h,w_o):
	h = T.nnet.sigmoid(T.dot(X,w_h))
	pred = T.nnet.softmax(T.dot(h,w_o))
	return pred

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

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

predy_x = model(X,w_h,w_o)
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