import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to pandas dataframe
sonar_data = pd.read_csv('./Copy of sonar data.csv', header = None)
sonar_data.head()
#detecting number of rows and cols
sonar_data.shape
sonar_data.describe(include= 'all')#to get statistical measures of the data
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()
#seperating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.1, random_state= 1)
print(X.shape, X_train.shape, X_test.shape)
print(X_train)
print(Y_train)
model = LogisticRegression()
model.fit(X_train, Y_train)
#model evaluation
#finding accuracy of our model on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print((training_data_accuracy)*100)
#finding accuracy of our model on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print((test_data_accuracy)*100)
#Making a predictive system
input_data = (0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.0080,0.0107,0.0161,0.0133)
#converting input_data numpy array to make process faster
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array for predicting one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=="R"):
  print ('The Object Scanned by the sonar is a Rock')
else:
  print ('The object Scanned by the sonar is a Mine')

