## **Importing Modules and Libraries**
import os
# Suppress TensorFlow GPU-related warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# importing required libraries
import numpy as np
import pandas as pd
import time

import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from os import path


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten , Activation, SimpleRNN, LSTM, GRU, Dropout, TimeDistributed, Reshape, Input, Lambda, Add
from keras import Sequential

from tensorflow.keras import backend as K
from keras.models import Model
from keras.optimizers import Adam



import sklearn.discriminant_analysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import class_weight
from sklearn.neighbors import NearestCentroid
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier


from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical
import warnings
warnings.simplefilter("ignore")

# requried to change for each dataset
method = "NSL_KDD_train_BBA_Smote_ENN"

# dont change this
method = method + "_Train_Test_70_30_Metrics"

## **Importing Datasets**
# # 1.NO_Opt
# train_data = pd.read_csv('NSL_KDD_train_No_Opt_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_No_Opt_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_No_Opt_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_No_Opt.csv')

# # 2.BBA
# train_data = pd.read_csv('NSL_KDD_train_BBA_No_Smote.csv')
train_data = pd.read_csv('NSL_KDD_train_BBA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_BBA_Smote_IPF.csv')

test_data = pd.read_csv('NSL_KDD_test_BBA.csv')

# # 3.CS
# train_data = pd.read_csv('NSL_KDD_train_CS_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_CS_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_CS_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_CS.csv')

# # 4.NO_Opt
# train_data = pd.read_csv('NSL_KDD_train_DE_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_DE_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_DE_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_DE.csv')

# # 5.EO
# train_data = pd.read_csv('NSL_KDD_train_EO_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_EO_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_EO_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_EO.csv')

# # 6.FA
# train_data = pd.read_csv('NSL_KDD_train_FA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_FA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_FA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_FA.csv')

# # 7.FPA
# train_data = pd.read_csv('NSL_KDD_train_FPA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_FPA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_FPA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_FPA.csv')

# # 8.GA
# train_data = pd.read_csv('NSL_KDD_train_GA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_GA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_GA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_GA.csv')

# # 9.GSA
# train_data = pd.read_csv('NSL_KDD_train_GSA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_GSA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_GSA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_GSA.csv')

# # 10.GWO
# train_data = pd.read_csv('NSL_KDD_train_GWO_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_GWO_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_GWO_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_GWO.csv')

# # 11.HHO
# train_data = pd.read_csv('NSL_KDD_train_HHO_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_HHO_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_HHO_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_HHO.csv')


# # 12.HS
# train_data = pd.read_csv('NSL_KDD_train_HS_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_HS_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_HS_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_HS.csv')

# # 13.JA
# train_data = pd.read_csv('NSL_KDD_train_JA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_JA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_JA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_JA.csv')

# # 14.MA
# train_data = pd.read_csv('NSL_KDD_train_MA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_MA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_MA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_MA.csv')

# # 15.PSO
# train_data = pd.read_csv('NSL_KDD_train_PSO_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_PSO_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_PSO_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_PSO.csv')

# # 16.RDA
# train_data = pd.read_csv('NSL_KDD_train_RDA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_RDA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_RDA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_RDA.csv')

# # 17.SCA
# train_data = pd.read_csv('NSL_KDD_train_SCA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_SCA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_SCA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_SCA.csv')

# # 18.SSA
# train_data = pd.read_csv('NSL_KDD_train_SSA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_SSA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_SSA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_SSA.csv')

# # 19.WOA
# train_data = pd.read_csv('NSL_KDD_train_WOA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_WOA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_WOA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_WOA.csv')
# print(train_data.shape)
# print(test_data.shape)



# **MULTI-CLASS CLASSIFICATION**
## **Data Splitting**
X_train = train_data.drop(columns=['label'],axis=1)
X_test = test_data.drop(columns=['label'],axis=1)
y_train = train_data['label']
y_test = test_data['label']

#feature_list = ['service','dpkts','dbytes','sttl','sload','dloss','sjit','smean','dmean','trans_depth','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','ct_ftp_cmd','ct_flw_http_mthd','ct_srv_dst','is_sm_ips_ports']
#X = X[feature_list]
fname = method + "_output.csv"
file = open(fname, 'w')
file.write("algo vs matrices,time to train(sec),time to predict(sec),accuracy_score,precision_score,recall_score,f1_score,fbeta_score,matthews_corrcoef,jaccard_score,cohen_kappa_score,hamming_loss,zero_one_loss,mean_absolute_error,mean_squared_error,mean_squared_error,balanced_accuracy_score,explained_variance_score\n")
def format_decimal(number):
    return f"{number:.{3}f}"
def result(y_pred,y_test,algo,start,end_train,end_predict):
    file.write(algo+",")
    file.write(str(format_decimal(end_train-start))+",")
    file.write(str(format_decimal(end_predict-end_train))+",")
    file.write(str(format_decimal(metrics.accuracy_score(y_test,y_pred)))+",")
    file.write(str(format_decimal(metrics.precision_score(y_test, y_pred, average='weighted')))+",")
    file.write(str(format_decimal(metrics.recall_score(y_test, y_pred, average='weighted')))+",")
    file.write(str(format_decimal(metrics.f1_score(y_test, y_pred, average='weighted')))+",")
    file.write(str(format_decimal(metrics.fbeta_score(y_test, y_pred,average='weighted', beta=0.5)))+",")
    file.write(str(format_decimal(metrics.matthews_corrcoef(y_test, y_pred)))+",")
    file.write(str(format_decimal(metrics.jaccard_score(y_test, y_pred, average='weighted')))+",")
    file.write(str(format_decimal(metrics.cohen_kappa_score(y_test, y_pred)))+",")
    file.write(str(format_decimal(metrics.hamming_loss(y_test, y_pred)))+",")
    file.write(str(format_decimal(metrics.zero_one_loss(y_test, y_pred)))+",")
    file.write(str(format_decimal(metrics.mean_absolute_error(y_test, y_pred)))+",")
    file.write(str(format_decimal(metrics.mean_squared_error(y_test, y_pred)))+",")
    file.write(str(format_decimal(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))+",")
    file.write(str(format_decimal(metrics.balanced_accuracy_score(y_test, y_pred)))+",")
    file.write(str(format_decimal(metrics.explained_variance_score(y_test, y_pred)*100))+"\n")




## **1.Decision Tree**
dt_multi = DecisionTreeClassifier(random_state=24)
start = time.time()
dt_multi.fit(X_train,y_train)
end_train = time.time()
y_pred = dt_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"DT",start,end_train,end_predict)
pname = method +"_Decision_Tree_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **2.Linear Regression**
lr_multi = LinearRegression()
start = time.time()
lr_multi.fit(X_train, y_train)
end_train = time.time()
y_pred = lr_multi.predict(X_test)
end_predict = time.time()

for i in range(len(y_pred)):
  y_pred[i] = int(np.round_(y_pred[i]))
result(y_pred,y_test,"Linear Regression",start,end_train,end_predict)
pname = method +"_"+"Linear_Regression"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **3.Logistic Regression**

logr_multi = LogisticRegression(random_state=123, max_iter=5000,solver='newton-cg',multi_class='multinomial')
start = time.time()
logr_multi.fit(X_train,y_train)
end_train = time.time()
y_pred = logr_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"Logistic Regression",start,end_train,end_predict)
pname = method +"_"+"Logistic_Regression"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **4.K Nearest Neighbor Classifier**
knn_multi = KNeighborsClassifier(n_neighbors=5)
start = time.time()
knn_multi.fit(X_train,y_train)
end_train = time.time()
y_pred = knn_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"KNN",start,end_train,end_predict)
pname = method +"_"+"KNeighborsClassifie"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **5.Random Forest Classifier**

rf_multi = RandomForestClassifier(random_state=50)
start = time.time()
rf_multi.fit(X_train,y_train)
end_train = time.time()
y_pred = rf_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"Random Forest",start,end_train,end_predict)
pname = method +"_"+"RandomForestClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **6.Multi Layer Perceptron**

mlp_multi = MLPClassifier(random_state=123, solver='adam', max_iter=8000)
start = time.time()
mlp_multi.fit(X_train,y_train)
end_train = time.time()
y_pred = mlp_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"MLP",start,end_train,end_predict)
pname = method +"_"+"MLPClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **7.Bagging**
# Create a base classifier (Decision Tree)
base_classifier = DecisionTreeClassifier(random_state=42)

# Create a bagging classifier
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
start = time.time()
# Train the bagging classifier
bagging_classifier.fit(X_train, y_train)
end_train = time.time()
# Predict on the test set
y_pred = bagging_classifier.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"Bagging",start,end_train,end_predict)
pname = method +"_"+"Bagging"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **8.J48**

# Initialize and train the J48 (Decision Tree) classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
start = time.time()
classifier.fit(X_train, y_train)
end_train = time.time()
# Make predictions
y_pred = classifier.predict(X_test)
end_predict = time.time()

result(y_pred,y_test,"J48",start,end_train,end_predict)
pname = method +"_"+"J48"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **9.ANN**
multi_ann = Sequential()
# Adding the input layer and the first hidden layer
multi_ann.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
multi_ann.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
multi_ann.add(Dense(units = len(np.unique(y_train)), kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN | means applying SGD on the whole ANN
multi_ann.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
start = time.time()
multi_ann.fit(X_train, y_train,epochs=10, batch_size=50,verbose=2)
end_train = time.time()
# Predicting the Test set results
y_pred = np.argmax(multi_ann.predict(X_test), axis=1)
# y_pred = (y_pred > 0.5)
end_predict = time.time()
result(y_pred,y_test,"ANN",start,end_train,end_predict)
pname = method +"_"+"ANN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **10.DNN**
multi_dnn = Sequential()
# Adding the input layer and the first hidden layer
multi_dnn.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
multi_dnn.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
multi_dnn.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
multi_dnn.add(Dense(units = len(np.unique(y_train)), kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN | means applying SGD on the whole ANN
multi_dnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
start = time.time()
multi_dnn.fit(X_train, y_train,epochs=10, batch_size=50,verbose=2)
end_train = time.time()
y_pred = np.argmax(multi_dnn.predict(X_test), axis=1)
end_predict = time.time()

result(y_pred,y_test,"DNN",start,end_train,end_predict)
pname = method +"_"+"DNN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **11.CNN**
multi_cnn = Sequential()
multi_cnn.add(Reshape((-1,1), input_shape=(X_train.shape[1],)))
multi_cnn.add(Conv1D(32, 3, activation='relu', padding='causal'))
multi_cnn.add(Conv1D(64, 3, activation='relu', padding='causal'))
multi_cnn.add(MaxPooling1D(pool_size=2))
multi_cnn.add(Flatten())
multi_cnn.add(Dense(len(np.unique(y_train)),activation='softmax'))
multi_cnn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
multi_cnn.fit(X_train, y_train, epochs=10, batch_size=50)
end_train = time.time()
y_pred = np.argmax(multi_cnn.predict(X_test), axis=1)
end_predict = time.time()
result(y_pred,y_test,"CNN",start,end_train,end_predict)
pname = method +"_"+"CNN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **12.Gradient Boosting Classifier**

start = time.time()
multi_gb = GradientBoostingClassifier().fit(X_train,y_train)
end_train = time.time()
y_pred= multi_gb.predict(X_test) # These are the predictions from the test data.
end_predict = time.time()
result(y_pred,y_test,"Gradient Boosting",start,end_train,end_predict)
pname = method +"_"+"GradientBoostingClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **13.xgboost**

#xgb_model = XGBClassifier('n_estimators': 500,'max_depth': 16)
xgb_model = XGBClassifier()
start = time.time()
xgb_model.fit(X_train, y_train)
end_train = time.time()
y_pred = xgb_model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"XGBoost",start,end_train,end_predict)
pname = method +"_"+"XGBClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **14.Gaussian Naive Bayes**
NB_model = GaussianNB()
start = time.time()
NB_model.fit(X_train, y_train)
end_train = time.time()
y_pred = NB_model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"Gaussian Naive Bayes",start,end_train,end_predict)
pname = method +"_"+"Gaussian_Naive_Bayes"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **15.Adaptive Gradient Boosting**
weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
n_estimators = 300

AB_model = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=n_estimators,
    algorithm="SAMME",
    random_state=42,
)
start = time.time()
AB_model.fit(X_train, y_train)
end_train = time.time()
y_pred = AB_model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"ADaboost",start,end_train,end_predict)
pname = method +"_"+"Adaptive_Gradient_Boosting"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **16.QuadraticDiscriminantAnalysis**
qda_multi = QuadraticDiscriminantAnalysis()

start = time.time()
qda_multi.fit(X_train,y_train)
end_train = time.time()
y_pred = qda_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"QDA",start,end_train,end_predict)
pname = method +"_"+"QDA"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **17.shallow neural network**

num_classes =  len(np.unique(y_train))


snn_multi = Sequential()
snn_multi.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
snn_multi.add(Dense(32, activation='relu'))
snn_multi.add(Dense(20, activation='relu'))
snn_multi.add(Dense(num_classes, activation='softmax'))


snn_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
snn_multi.fit(X_train, y_train, epochs=10, batch_size=50)
end_train = time.time()
y_pred = np.argmax(snn_multi.predict(X_test), axis=1)
end_predict = time.time()
result(y_pred,y_test,"SNN",start,end_train,end_predict)
pname = method +"_"+"SNN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **18.restricted boltzmann machine**
num_classes = len(np.unique(y_train))

# Define the RBM class
class RBM(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, name="rbm", **kwargs):
        super(RBM, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer='uniform', trainable=True, name='weights')
        self.h_bias = self.add_weight(shape=(self.hidden_dim,), initializer='zeros', trainable=True, name='h_bias')
        self.v_bias = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name='v_bias')

    def call(self, inputs):
        hidden_prob = tf.nn.sigmoid(tf.matmul(inputs, self.W) + self.h_bias)
        hidden_state = self._sample_prob(hidden_prob)
        visible_prob = tf.nn.sigmoid(tf.matmul(hidden_state, tf.transpose(self.W)) + self.v_bias)
        return visible_prob, hidden_state

    def _sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

# Assuming class 6 corresponds to index 5 (Python uses 0-based indexing)
input_data = Input(shape=(X_train.shape[1],))
rbm1_visible, rbm1_hidden = RBM(hidden_dim=128, name="rbm1")(input_data)
rbm2_visible, rbm2_hidden = RBM(hidden_dim=64, name="rbm2")(rbm1_hidden)
rbm3_visible, rbm3_hidden = RBM(hidden_dim=32, name="rbm3")(rbm2_hidden)

# Specific RBM for class 6 with more hidden units
rbm6_visible, rbm6_hidden = RBM(hidden_dim=64, name="rbm6")(rbm3_hidden)

# Classifier model
classifier_output = Dense(num_classes, activation='softmax', name='classifier')(rbm6_hidden)

# Create the combined model
model = tf.keras.Model(inputs=input_data, outputs=classifier_output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the stacked RBM + classifier
start = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=50)
end_train = time.time()
y_pred = np.argmax(model.predict(X_test), axis=1)
end_predict = time.time()
result(y_pred,y_test,"RBM",start,end_train,end_predict)
pname = method +"_"+"RBM"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **19.LSTM**
num_classes =  len(np.unique(y_train))

# Convert DataFrame to NumPy array
X_train_array_multi = X_train.to_numpy()

# Reshape input data for LSTM
X_train_reshaped_multi = X_train_array_multi.reshape((X_train_array_multi.shape[0], X_train_array_multi.shape[1], 1))

# Define the recuurent neural network model
rnn_multi = Sequential()
rnn_multi.add(LSTM(128, input_shape=(X_train_reshaped_multi.shape[1], X_train_reshaped_multi.shape[2])))
rnn_multi.add(Dense(32, activation='relu'))
rnn_multi.add(Dense(num_classes, activation='softmax'))
rnn_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
rnn_multi.fit(X_train, y_train, epochs=10, batch_size=50)
end_train = time.time()

y_pred = np.argmax(rnn_multi.predict(X_test), axis=1)
end_predict = time.time()
result(y_pred,y_test,"LSTM",start,end_train,end_predict)
pname = method +"_"+"LSTM"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()

## **20.reconstruction neural networks**

num_classes = len(np.unique(y_train))

# Assuming y_train_multi is one-hot encoded
y_train_multi_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_multi_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Define model architecture
input_dim = X_train.shape[1]
encoding_dim = 32  # Choose appropriate dimensionality
latent_dim = 2  # Dimensionality of the latent space

# Encoder
input_layer = tf.keras.layers.Input(shape=(input_dim,))
hidden = tf.keras.layers.Dense(64, activation='relu')(input_layer)
z_mean = tf.keras.layers.Dense(latent_dim)(hidden)
z_log_var = tf.keras.layers.Dense(latent_dim)(hidden)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = tf.keras.layers.Lambda(sampling,output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_hidden = tf.keras.layers.Dense(64, activation='relu')(z)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(decoder_hidden)

# Define VAE model
vae = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile model
vae.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
vae.fit(X_train, y_train_multi_onehot, epochs=10, batch_size=50, shuffle=True)
end_train = time.time()
y_pred = np.argmax(vae.predict(X_test), axis=1)
end_predict = time.time()
result(y_pred,y_test,"reconstruction_NN",start,end_train,end_predict)
pname = method +"_"+"reconstruction_NN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **21.ELM**
class ELMClassifier:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # Initialize random weights and biases
        self.input_weights = np.random.randn(hidden_layer_size, input_size)
        self.bias = np.random.randn(hidden_layer_size, 1)
        self.output_weights = None

    def _softmax(self, x):
       exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
       return exp_values / np.sum(exp_values, axis=1, keepdims=True)


    def _one_hot_encode(self, y):
        classes = np.unique(y)
        encoded = np.zeros((len(y), len(classes)))

        for i, cls in enumerate(classes):
            encoded[y == cls, i] = 1

        return encoded

    def fit(self, X_train, y_train):
        # Calculate hidden layer output
        hidden_layer_output = self._softmax(np.dot(self.input_weights, X_train.T) + self.bias)

        # Calculate output weights using Moore-Penrose pseudoinverse
        self.output_weights = np.dot(np.linalg.pinv(hidden_layer_output.T), self._one_hot_encode(y_train))

    def predict(self, X_test):
        # Calculate hidden layer output for test data
        hidden_layer_output = self._softmax(np.dot(self.input_weights, X_test.T) + self.bias)

        # Make predictions using the output weights
        raw_predictions = np.dot(hidden_layer_output.T, self.output_weights)
        predictions = np.argmax(raw_predictions, axis=1)

        return predictions


# Specify the number of hidden layer neurons and output classes
input_size = X_train.shape[1]
hidden_layer_size = 500  # You can adjust this parameter
output_size = len(np.unique(y_train))

# Create and train the ELM classifier
elm_classifier = ELMClassifier(input_size, hidden_layer_size, output_size)
start = time.time()
elm_classifier.fit(X_train, y_train)
end_train = time.time()
# Make predictions on the test set
y_pred = elm_classifier.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"ELM",start,end_train,end_predict)
pname = method +"_"+"ELM"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **22.DANN**

def build_dann_model(input_shape, num_classes, lambda_val=1e-3):
    input_layer = Input(shape=input_shape, name='input_layer')

    # Feature extractor
    shared_layer = Dense(128, activation='relu')(input_layer)
    shared_layer = Dropout(0.5)(shared_layer)

    # Source classifier
    source_classifier = Dense(num_classes, activation='softmax', name='source_classifier')(shared_layer)

    # Domain classifier
    domain_classifier = Dense(1, activation='sigmoid', name='domain_classifier')(shared_layer)

    # Combined model
    model = Model(inputs=input_layer, outputs=[source_classifier, domain_classifier])

    # Domain adversarial loss
    def domain_adversarial_loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred))

    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss={'source_classifier': 'categorical_crossentropy', 'domain_classifier': domain_adversarial_loss},
                  loss_weights={'source_classifier': 1.0, 'domain_classifier': lambda_val},
                  metrics={'source_classifier': 'accuracy'})

    return model

# Convert class vectors to binary class matrices
num_classes = len(np.unique(y_train))
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)

# Build and train DANN model
input_shape = (X_train.shape[1],)
lambda_val = 1e-3  # Trade-off parameter for domain adversarial loss

dann_model = build_dann_model(input_shape, num_classes, lambda_val)
start = time.time()
dann_model.fit(X_train, {'source_classifier': y_train_categorical, 'domain_classifier': np.zeros((len(X_train), 1))},
               epochs=10, batch_size=64, validation_data=(X_test, {'source_classifier': y_test_categorical, 'domain_classifier': np.ones((len(X_test), 1))}))

end_train = time.time()
predictions = dann_model.predict(X_test)
# Extracting the predicted class probabilities and domain predictions
source_classifier_predictions = predictions[0]
y_pred = np.argmax(source_classifier_predictions, axis=1)
end_predict = time.time()
result(y_pred,y_test,"DANN",start,end_train,end_predict)
pname = method +"_"+"DANN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **23.Deep brief networks (DBNs)**

# Number of classes
num_classes = len(np.unique(y_train))

# Create a pipeline with BernoulliRBM and MLPClassifier
rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=20, random_state=42, verbose=True)
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=52)
start = time.time()
dbn_model = Pipeline(steps=[('rbm', rbm), ('mlp', mlp)])

# Train the classifier
dbn_model.fit(X_train, y_train)
end_train = time.time()
## Predict on the test set
y_pred = dbn_model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"DBN",start,end_train,end_predict)
pname = method +"_"+"DBN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **24.Deep Boltzmann machines (DBMs)**

# Convert data to float32 to ensure consistent data types
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train = X_train.values
X_test = X_test.values

# Build a simple Restricted Boltzmann Machine (RBM) using TensorFlow
class RBM(tf.Module):
    def __init__(self, visible_dim, hidden_dim, learning_rate=0.01):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W = tf.Variable(tf.random.normal([visible_dim, hidden_dim], stddev=0.01, dtype=tf.float32))
        self.b_visible = tf.Variable(tf.zeros([visible_dim], dtype=tf.float32))
        self.b_hidden = tf.Variable(tf.zeros([hidden_dim], dtype=tf.float32))

    def _softmax(self, x):
        exp_x = tf.exp(x)
        return exp_x / tf.reduce_sum(exp_x, axis=1, keepdims=True)

    def sample_hidden(self, visible_prob):
        hidden_prob = self._softmax(tf.matmul(visible_prob, self.W) + self.b_hidden)
        return tf.nn.relu(tf.sign(hidden_prob - tf.random.uniform(tf.shape(hidden_prob))))

    def sample_visible(self, hidden_prob):
        visible_prob = self._softmax(tf.matmul(hidden_prob, tf.transpose(self.W)) + self.b_visible)
        return tf.nn.relu(tf.sign(visible_prob - tf.random.uniform(tf.shape(visible_prob))))

    def contrastive_divergence(self, x, k=1):
        visible = x
        for _ in range(k):
            hidden = self.sample_hidden(visible)
            visible = self.sample_visible(hidden)

        positive_hidden = self._softmax(tf.matmul(x, self.W) + self.b_hidden)
        negative_hidden = self._softmax(tf.matmul(visible, self.W) + self.b_hidden)

        # Update weights and biases
        self.W.assign_add(self.learning_rate * (tf.matmul(tf.transpose(x), positive_hidden) -
                                                tf.matmul(tf.transpose(visible), negative_hidden)))
        self.b_visible.assign_add(self.learning_rate * tf.reduce_mean(x - visible, axis=0))
        self.b_hidden.assign_add(self.learning_rate * tf.reduce_mean(positive_hidden - negative_hidden, axis=0))

# Number of visible and hidden units
visible_dim = X_train.shape[1]
hidden_dim1 = 64
hidden_dim2 = 32

# Create RBMs for each layer
rbm1 = RBM(visible_dim, hidden_dim1)
rbm2 = RBM(hidden_dim1, hidden_dim2)

# Training RBMs
num_epochs = 5
batch_size = 32
start = time.time()
# Training first RBM
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_data = X_train[i:i+batch_size]
        rbm1.contrastive_divergence(batch_data)

# Getting hidden layer representation from the first RBM
hidden1_representation = tf.nn.relu(tf.sign(rbm1.sample_hidden(X_train)))

# Training second RBM using the hidden layer representation from the first RBM
for epoch in range(num_epochs):
    for i in range(0, len(hidden1_representation), batch_size):
        batch_data = hidden1_representation[i:i+batch_size]
        rbm2.contrastive_divergence(batch_data)

# Getting hidden layer representation from the second RBM
hidden2_representation = tf.nn.relu(tf.sign(rbm2.sample_hidden(hidden1_representation)))

# Fine-tuning for classification
num_classes = len(np.unique(y_train))  # Replace with the actual number of classes
dbm_model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_dim1, activation='relu'),
    tf.keras.layers.Dense(hidden_dim2, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

dbm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
start = time.time()
dbm_model.fit(X_train, y_train, epochs=10, batch_size=50, shuffle=True, validation_data=(X_test, y_test))
end_train = time.time()
## Predict on the test set
y_pred_probabilities = dbm_model.predict(X_test)
end_predict = time.time()
y_pred = np.argmax(y_pred_probabilities, axis=1)
result(y_pred,y_test,"DBM",start,end_train,end_predict)
pname = method +"_"+"DBM"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **25.DEEP AUTO ENCODERS(DA)**
num_classes = len(np.unique(y_train))

# Define the autoencoder model
autoencoder = Sequential()

# Encoder
autoencoder.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dense(32, activation='relu'))

# Decoder
autoencoder.add(Dense(64, activation='relu'))
autoencoder.add(Dense(128, activation='relu'))
autoencoder.add(Dense(X_train.shape[1], activation='linear'))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=10, batch_size=50, shuffle=True, validation_data=(X_test, X_test))

# Add a classification head on top of the trained autoencoder
da_model = Sequential()
da_model.add(autoencoder.layers[0])  # Add encoder layers
da_model.add(autoencoder.layers[1])
da_model.add(autoencoder.layers[2])
da_model.add(Dense(num_classes, activation='softmax'))  # Adjust output layer for multiple classes

# Compile the classification model
da_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=num_classes)
y_test_onehot = to_categorical(y_test, num_classes=num_classes)

# Train the classification model using the encoded representations
start = time.time()
da_model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test, y_test_onehot))
end_train = time.time()
## Predict on the test set
y_pred_probabilities = da_model.predict(X_test)
end_predict = time.time()
y_pred = np.argmax(y_pred_probabilities, axis=1)
result(y_pred,y_test,"DA",start,end_train,end_predict)
pname = method +"_"+"DA"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **26.PassiveAggressiveClassifier**

model =  PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)

start = time.time()
model.fit(X_train, y_train)
end_train = time.time()

y_pred = model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"PassiveAggressiveClassifier",start,end_train,end_predict)
pname = method +"_"+"PassiveAggressiveClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **27.RidgeClassifier**

model =  RidgeClassifier()

start = time.time()
model.fit(X_train, y_train)
end_train = time.time()

y_pred = model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"RidgeClassifier",start,end_train,end_predict)
pname = method +"_"+"RidgeClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()


## **29.NearestCentroid**
model = NearestCentroid()

start = time.time()
model.fit(X_train, y_train)
end_train = time.time()

y_pred = model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"NearestCentroid",start,end_train,end_predict)
pname = method +"_"+"NearestCentroid"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **30.Cost Sensitive Logistic Regression(CSLR)**
# Gemini


def get_sample_weight(cost_matrix, y_tru):
  y_true = np.array(y_tru)
  num_samples = len(y_true)
  sample_weights = np.zeros(num_samples)
  for i in range(num_samples):
    true_class = y_true[i]
    for j in range(len(cost_matrix)):
      if j != true_class:
        sample_weights[i] += cost_matrix[true_class, j]
  return sample_weights

cost_matrix = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],  # Costs for misclassifying class 0
                        [1, 0, 1, 2, 3, 4, 5, 6, 7],  # Costs for misclassifying class 1
                        [2, 1, 0, 1, 2, 3, 4, 5, 6],  # Costs for misclassifying class 2
                        [3, 2, 1, 0, 1, 2, 3, 4, 5],  # Costs for misclassifying class 3
                        [4, 3, 2, 1, 0, 1, 2, 3, 4],  # Costs for misclassifying class 4
                        [5, 4, 3, 2, 1, 0, 1, 2, 3],  # Costs for misclassifying class 5
                        [6, 5, 4, 3, 2, 1, 0, 1, 2],  # Costs for misclassifying class 6
                        [7, 6, 5, 4, 3, 2, 1, 0, 1],  # Costs for misclassifying class 7
                        [8, 7, 6, 5, 4, 3, 2, 1, 0]]) # Costs for misclassifying class 8

sample_weights = get_sample_weight(cost_matrix, y_train)

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

clf_cost_sensitive = LogisticRegression(solver='lbfgs')
start = time.time()
clf_cost_sensitive.fit(X_train, y_train, sample_weight=sample_weights)
end_train = time.time()

y_pred = clf_cost_sensitive.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"CSLR",start,end_train,end_predict)
pname = method +"_"+"CSLR"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **31.Cost sensitive bagging classifier(CSBC)**


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)


# Step 3: Initialize the CostSensitiveBaggingClassifier model
base_estimator = DecisionTreeClassifier(max_depth=5)
bagging_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

start = time.time()
# Step 4: Train the model on your dataset
bagging_model.fit(X_train, y_train)
end_train = time.time()


# Step 5: Evaluate the model's performance
y_pred = bagging_model.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"CSBC",start,end_train,end_predict)
pname = method +"_"+"CSBC"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **32.lightgbm**
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
start = time.time()
lgbm.fit(X_train, y_train)
end_train = time.time()
y_pred = lgbm.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"lightgbm",start,end_train,end_predict)
pname = method +"_"+"lightgbm"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **33.LinearDiscriminantAnalysis(LDA)**

lda = LinearDiscriminantAnalysis(n_components=1)
start=time.time()
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
end_train=time.time()
y_pred = classifier.predict(X_test)
end_predict=time.time()
result(y_pred,y_test,"LDA",start,end_train,end_predict)
pname = method +"_"+"LDA"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **34.GRU**
num_classes =  len(np.unique(y_train))

# Convert DataFrame to NumPy array
X_train_array_multi = X_train

# Reshape input data for LSTM
X_train_reshaped_multi = X_train_array_multi.reshape((X_train_array_multi.shape[0], X_train_array_multi.shape[1], 1))

# Define the recuurent neural network model
rnn_multi = Sequential()
rnn_multi.add(GRU(128, input_shape=(X_train_reshaped_multi.shape[1], X_train_reshaped_multi.shape[2])))
rnn_multi.add(Dense(32, activation='relu'))
rnn_multi.add(Dense(num_classes, activation='softmax'))
rnn_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start = time.time()
rnn_multi.fit(X_train, y_train, epochs=10, batch_size=50)
end_train = time.time()
y_pred = np.argmax(rnn_multi.predict(X_test), axis=1)
end_predict = time.time()
result(y_pred,y_test,"GRU",start,end_train,end_predict)
pname = method +"_"+"GRU"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **35.Stochastic gradient**


sgd_multi = make_pipeline(StandardScaler(), SGDClassifier(random_state=24))

# Measure time to train
start = time.time()
sgd_multi.fit(X_train, y_train)
end_train = time.time()

# Measure time to predict
y_pred = sgd_multi.predict(X_test)
end_predict = time.time()

result(y_pred,y_test,"Stochastic_gradient",start,end_train,end_predict)
pname = method +"_"+"Stochastic_gradient"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **36.SVM**
svm_multi = make_pipeline(StandardScaler(), SVC(random_state=24))

# Measure time to train
start = time.time()
svm_multi.fit(X_train, y_train)
end_train = time.time()
y_pred_svm = svm_multi.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"SVM",start,end_train,end_predict)
pname = method +"_"+"SVM"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **37.ExtraTreesClassifier**

# Train Extra Trees classifier and evaluate
start = time.time()
extra_trees_clf = ExtraTreesClassifier()
extra_trees_clf.fit(X_train, y_train)
end_train = time.time()
y_pred = extra_trees_clf.predict(X_test)
end_predict = time.time()
result(y_pred,y_test,"ExtraTreesClassifier",start,end_train,end_predict)
pname = method +"_"+"ExtraTreesClassifier"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **38.Feed Forward Neural Networks**
# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # This accuracy is for training monitoring, not the evaluation metric

# Train the model
start = time.time()
model.fit(X_train, y_train, epochs=10, batch_size=50, verbose=1)
end_train = time.time()
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
end_predict = time.time()
result(y_pred,y_test,"FFNN",start,end_train,end_predict)
pname = method +"_"+"FFNN"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **39.Fuzzy**

# Train the model
start = time.time()


# Generate fuzzy c-means clusters
n_clusters = 10  # Number of classes
centers, u_train, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_train.T, n_clusters, 2, error=0.005, maxiter=1000
)

# Predict cluster membership for test data
u_test, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
    X_test.T, centers, 2, error=0.005, maxiter=1000
)

end_train = time.time()

# Evaluate the model on test data
start_time = time.time()

# Assign class labels based on cluster membership
y_pred = np.argmax(u_test, axis=0)
end_predict = time.time()
result(y_pred,y_test,"Fuzzy",start,end_train,end_predict)
pname = method +"_"+"Fuzzy"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **40.Ensemble_of_DL_Networks(EDLNs)**
# Define the architecture of your neural network (example architecture)
def create_model(input_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
start = time.time()


# Define hyperparameters
num_networks = 5
epochs = 10
num_classes = len(np.unique(y_train))  # Number of unique classes in the training data

# Train multiple neural networks
models = []
for i in range(num_networks):
    model = create_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    models.append(model)


training_time = time.time() - start_time

# Evaluate the model on test data
end_train = time.time()


# Make predictions on test data using each model
predictions = np.array([model.predict(X_test) for model in models])

# Aggregate predictions by averaging
y_pred = np.argmax(np.mean(predictions, axis=0), axis=1)

end_predict = time.time()

result(y_pred,y_test,"EDLNs",start,end_train,end_predict)
pname = method +"_"+"EDLNs"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()
## **41.GMM**

# Convert data to numpy arrays if needed
X_train_multi = np.array(X_train)
X_test_multi = np.array(X_test)
# Number of classes
n_classes = len(set(y_train))

# Dictionary to store GMMs for each class
gmm_models = {}

# Train GMMs for each class
start = time.time()
for i in range(n_classes):
    # Filter data for the current class
    X_class = X_train_multi[y_train == i]
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2)  # You can adjust n_components as needed
    gmm.fit(X_class)
    # Store the trained GMM
    gmm_models[i] = gmm
end_train = time.time()



import time

start_predict = time.time()
y_pred = []
for x in X_test_multi:
    class_likelihoods = []
    # Reshape x to have the appropriate dimensions
    x_reshaped = x.reshape(1, -1)
    # Calculate likelihood for each class
    for i in range(n_classes):
        class_likelihood = gmm_models[i].score_samples(x_reshaped)
        class_likelihoods.append(class_likelihood)
    # Assign the class with the highest likelihood
    predicted_class = max(zip(class_likelihoods, range(n_classes)))[1]
    y_pred.append(predicted_class)
end_predict = time.time()
result(y_pred,y_test,"GMM",start,end_train,end_predict)
pname = method +"_"+"GMM"+"_confusion_matrix.png"
cm = multilabel_confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize']=8,8
sns.set_style("white")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(pname)
#plt.show()

## **Closing the file**
file.close()
