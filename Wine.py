"""Regression problem: predicting the value of quality of wine (target value is a integer from 1 to 10) based on features (all are float numbers).
value of quality is not continus so can be solved as classification as well but we just practicing different ways.
This is includes deep learning with different layers and settings)
"""

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


data = np.array(pd.read_csv('/Users/Hoda/Desktop/Insight/wine/winequality-red.csv'))
df = pd.DataFrame(data)
#data.info()
#data.describe()
#plt.hist(data['quality']) #shows the distribution of data corresponded to quality of wine
#plt.show()
early_stopping_monitor = EarlyStopping(patience = 2) 
columns = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]] #a data frame of data excluding the target column 
labels = df.iloc[:,11] #a data frame of data only including target cloumn
model = Sequential()
n_cols = columns.shape[1] #number of predictive values

"""#deep learning with two hidden leayer, each 100 nodes
model.add(Dense(100, activation ='relu', input_shape=(n_cols, )))
model.add(Dense(100, activation ='relu'))
model.add(Dense(1))"""


"""#deep learning with two hidden leayer, each 50 nodes
model.add(Dense(50, activation ='relu', input_shape=(n_cols, )))
model.add(Dense(50, activation ='relu'))
model.add(Dense(1))"""

"""
#deep learning with one hidden leayer, each 100 nodes
model.add(Dense(100, activation ='relu', input_shape=(n_cols, )))
model.add(Dense(1))
"""

"""#deep learning with one hidden leayer, each 50 nodes
model.add(Dense(50, activation ='relu', input_shape=(n_cols, )))
model.add(Dense(1))
"""

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(pd.DataFrame.as_matrix(columns),pd.DataFrame.as_matrix(labels), validation_split = 0.3, nb_epoch = 20, callbacks = [early_stopping_monitor])



"""Classificatio problem: predicting the value of quality of wine (target value is a integer from 1 to 10) based on features (all are float numbers).
This is includes deep learning with different layers and settings)
"""

#classification problem
#columns = data.drop(['quality'],axis = 1).as_matrix() #axis =1 means drop a column (i.e. quality is a column)
#labels = np_utils.to_categorical(data.quality)
#model.add(Dense(10, activation = 'softmax'))
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
