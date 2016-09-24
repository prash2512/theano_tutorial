import theano
import theano.tensor as T 
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle, gzip
from util import *
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d


srng = RandomStreams()

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

def dropout(X,p = 0.):
	if p>0:
		retain = 1-p
		X = X*srng.binomial(X.shape,p= retain,dtype=theano.config.floatX)
		X = X/retain
	return X

def model(X,w1,w2,w3,w4,w_o,prob_hidden,prob_input):
	X = dropout(X,prob_input)

	l1_temp = relu(conv2d(X,w1,border_mode='full'))
	l1 = max_pool_2d(l1_temp,(2,2))
	l1 = dropout(l1,prob_hidden)

	l2_temp = relu(conv2d(l1,w2))
	l2 = max_pool_2d(l2_temp,(2,2))
	l2 = dropout(l2,prob_hidden)

	l3_temp = relu(conv2d(l2,w3))
	l3_temp1 = max_pool_2d(l3_temp,(2,2),)
	l3 = T.flatten(l3_temp1,outdim=2)
	l3 = dropout(l3,prob_hidden)

	l4 = relu(T.dot(l3,w4))
	l4 = dropout(l4,prob_hidden)

	predy_x = softmax(T.dot(l4,w_o))

	return l1,l2,l3,l4,predy_x

n_epochs = 100


#load X as a tensor
X = T.ftensor4()
Y = T.fmatrix()

#load data
x_train,y_train,x_valid,y_valid,x_test,y_test = load_mnist()

#reshape data into form of tensor for conv operations
x_train = x_train.reshape(-1,1,28,28)
x_test = x_test.reshape(-1,1,28,28)

#set one_hot label
y_train = one_hot(y_train,10)
y_test = one_hot(y_test,10)

#init weights
w1 = init_weights((16,1,3,3))
w2 = init_weights((32,16,3,3))
w3 = init_weights((64,32,3,3))
w4 = init_weights((64*3*3,625))
w_o = init_weights((625,10))

noiseh1,noiseh2,noiseh3,noiseh4,noisey_x = model(X,w1,w2,w3,w4,w_o,0.5,0.2)
h1,h2,h3,h4,predy_x = model(X,w1,w2,w3,w4,w_o,0.,0.)
y_pred = T.argmax(predy_x,axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noisey_x,Y))
params = [w1,w2,w3,w4,w_o]
updates = rmsprop(cost,params,lr=0.001)

train = theano.function(inputs = [X,Y] , outputs =cost, updates = updates, allow_input_downcast = True)
predict = theano.function(inputs = [X], outputs = y_pred ,allow_input_downcast = True)

for i in range(n_epochs):
	for start, end in zip(range(0, len(x_train), 128), range(128, len(x_train), 128)):
            cost = train(x_train[start:end], y_train[start:end])
            print cost
	print "accuracy = ",100*np.mean(np.argmax(y_test,axis=1)==predict(x_test)),"%"