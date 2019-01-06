import numpy as np
import pandas as pd

dataset = pd.read_csv('trumpet.csv')
X = dataset.iloc[:, 0:19].values
y = dataset.iloc[:, -1].values
y_not_encoded = y

from keras.utils import np_utils
y = np_utils.to_categorical(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=12, init='uniform', activation='relu', input_dim=19))
classifier.add(Dense(output_dim=12, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=4, init='uniform', activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)


#To Predict
y_pred = classifier.predict(X_test)

#To convert y pred in right format
y_pred_modified=np.zeros_like(y_pred)
y_pred_modified[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

#testing accuracy
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test.argmax(axis=1), y_pred_modified.argmax(axis=1))

#use to predict new values & use above function....
y_predict4 = classifier.predict(sc.transform(np.array([[0.004552907,0.00896062,7.6480045,-72.009514,-14.29964,8.205175,33.47892,31.563509999999997,-16.947468,-31.122694,2.6064602999999997,41.751205,42.386013,-15.203971,1715.9824219999998,2793998.75,-6.959999999999999e-09,0.074429087,1.835897446
]])))


