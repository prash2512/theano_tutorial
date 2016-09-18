import theano
import theano.tensor as T 
import numpy as np
import cPickle, gzip
from util import load_mnist


a = T.scalar()
b = T.scalar()
y = a*b
product = theano.function(inputs=[a,b],outputs=y)
print product(2,3)


x_train,y_train,x_valid,y_valid,x_test,y_test = load_mnist()
y_pred = np.zeros(y_test.shape)
count=0
for i in range(len(x_test)):
	distances = np.sum(np.abs(x_train - x_test[i,:]), axis = 1)
	minindex = np.argmin(distances)
	y_pred[i] = y_train[minindex]
	if y_test[i] == y_pred[i]:
		count=count+1
	print count
print 'accuracy: %f' % ( count/100 )
