import numpy as np
import pandas as pd

dataset = pd.read_csv('trumpet.csv')
X = dataset.iloc[:, 0:4].values
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

#Evaluaing ANN not working----check in Sid's PC
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=4, init='uniform', activation='relu', input_dim=4))
    classifier.add(Dense(output_dim=4, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=4, init='uniform', activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

#Tuning the neural network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=4, init='uniform', activation='relu', input_dim=4))
    classifier.add(Dense(output_dim=4, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=4, init='uniform', activation='softmax'))
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier) 
parameters = {'batch_size' : [25, 32],
              'nb_epoch' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']
             }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 100)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_