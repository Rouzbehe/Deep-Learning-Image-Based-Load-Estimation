# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 01:31:57 2017

@author: Rouzbeh Davoudi
"""

import pandas, keras
import numpy , scipy.io
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation , Dropout
from keras.layers import advanced_activations
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
# load dataset
dataframe = pandas.read_csv("without_rein.dat", delim_whitespace=False, header=None)
dataset = dataframe.values


# split into input (X) and output (Y) variables
features=numpy.array([13,20,21,24,26])-1
testfeatNo=numpy.array([43])-1
X = dataset[:,features]
Y = dataset[:,testfeatNo]

# preprocessing step  

#X = preprocessing.normalize(X)
# convert the data into -1 to +1 range
#maxY=numpy.amax(Y)
#minY=numpy.amin(Y)
#Y=-1+2*(Y-minY)/(maxY-minY)

# define base mode

#def baseline_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(10, input_dim=10, init='normal', activation='relu'))
#	model.add(Dense(1, init='normal'))
#	# Compile model
#	model.compile(loss='mean_squared_error', optimizer='adam')
#	return model
# # fix random seed for reproducibility
seed = 1
#numpy.random.seed(seed)
## evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
#
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)
#
#
#
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#
## evaluate model with standardized dataset
#numpy.random.seed(seed)
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)
#print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#act=keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
#act=K.elu(x, alpha=1.0)


#def vector_mse(y_true, y_pred):
#    from theano import tensor as T
#    diff2 = T.sum((y_pred - T.mean(y_true))**2)
#    diff1 = T.sum((y_true - T.mean(y_true))**2)
#    return diff1/diff2

#def vector_mse(y, x):
#    
#    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.eval(), y.eval())
#    
#    return r_value**2
 #	act = advanced_activations.LeakyReLU(alpha=0.5)
#def l1_reg(weight_matrix):
#    return 0.01 * K.sum(K.abs(weight_matrix)) 
def larger_model():
	# create model
	model = Sequential()
 
#	model.add(Dense(input_dim = 10, output_dim = 20))
##	model.add(BatchNormalization())
#	model.add(Activation('relu'))
#	model.add(Dense(input_dim = 20, output_dim = 20))
##	model.add(BatchNormalization())
#	model.add(Activation('relu'))
#
#	model.add(Dense(input_dim = 20, output_dim = 1))
##	model.add(BatchNormalization())
#	model.add(Activation('relu'))

	model.add(Dense(20, input_dim=5,  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None) , activation='elu')) #, W_regularizer=regularizers.l1(0.02)
                
#	model.add(BatchNormalization())
#	model.add(Activation('sigmoid'))
#	model.add(Dropout(0.25))
#	model.add(Dense(10, init='normal'))
#	model.add(BatchNormalization())
#	model.add(Activation('elu'))
##	model.add(Dropout(0.25))
	model.add(Dense(7, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), activation='elu'))
#	model.add(Dense(5, init='normal', activation='elu'))

#	model.add(BatchNormalization())
#	model.add(Activation('sigmoid'))
#	model.add(act)
#	model.add(act)
#	model.add(Dense(10, init='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))
	# Compile model
	opt=keras.optimizers.SGD(lr=0.05, momentum=0., decay=0, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer='sgd')
	return model
 
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=5000, batch_size=1000, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs = 1)
predicted = cross_val_predict(pipeline, X, Y, cv=kfold, n_jobs = 1)

# change -1 to +1 range back to raw dataw
#predicted=(predicted+1)*(maxY-minY)/2+minY
#Y=(Y+1)*(maxY-minY)/2+minY


fig, ax = plt.subplots()
ax.scatter(Y, predicted)
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

arr=numpy.c_[Y, predicted]
scipy.io.savemat('C:/Users/Rouzbeh Davoudi/Dropbox/Research/Data/Beam without shear reinforcement/Combined Study_All/Deep Neural Network_Python/arrdata.mat', mdict={'arr': arr})
#def wider_model():
#	# create model
#	model = Sequential()
#	model.add(Dense(20, input_dim=10, init='normal', activation='relu'))
#	model.add(Dense(1, init='normal'))
#	# Compile model
#	model.compile(loss='mean_squared_error', optimizer='adam')
#	return model
# 
#numpy.random.seed(seed)
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=wider_model, nb_epoch=100, batch_size=5, verbose=0)))
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(pipeline, X, Y, cv=kfold)
#print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


 