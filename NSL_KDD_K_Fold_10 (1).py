
n_splits_for_cv = 10 #change as required


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

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

from sklearn import metrics
import numpy as np
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical
import warnings
warnings.simplefilter("ignore")

# requried to change for each dataset
method = "NSL_KDD_train_No_Opt_No_Smote"

# dont change this
method = method + "_K_Fold_10_Metrics"

## **Importing Datasets**
# # 1.NO_Opt
train_data = pd.read_csv('NSL_KDD_train_No_Opt_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_No_Opt_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_No_Opt_Smote_IPF.csv')

test_data = pd.read_csv('NSL_KDD_test_No_Opt.csv')

# # 2.BBA
# train_data = pd.read_csv('NSL_KDD_train_BBA_No_Smote.csv')
# train_data = pd.read_csv('NSL_KDD_train_BBA_Smote_ENN.csv')
# train_data = pd.read_csv('NSL_KDD_train_BBA_Smote_IPF.csv')

# test_data = pd.read_csv('NSL_KDD_test_BBA.csv')

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

print(train_data.shape)
print(test_data.shape)

# **MULTI-CLASS CLASSIFICATION**
## **Data Splitting**
X_train = train_data.drop(columns=['label'],axis=1)
X_test = test_data.drop(columns=['label'],axis=1)
y_train = train_data['label']
y_test = test_data['label']

X_train = pd.concat([X_train, X_test], axis=0)

# Concatenate y_train and y_test along rows
y_train = pd.concat([y_train, y_test], axis=0)

# Reset indices of X_train and y_train
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

print(X_train.shape)
print(y_train.shape)

#feature_list = ['service','dpkts','dbytes','sttl','sload','dloss','sjit','smean','dmean','trans_depth','ct_state_ttl','ct_dst_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','ct_ftp_cmd','ct_flw_http_mthd','ct_srv_dst','is_sm_ips_ports']
#X = X[feature_list]

fname = method + "_output.csv"
file = open(fname, 'w')
file.write("algo vs matrices,time to train(sec),time to predict(sec),accuracy_score,precision_score,recall_score,f1_score,fbeta_score,matthews_corrcoef,jaccard_score,cohen_kappa_score,hamming_loss,zero_one_loss,mean_absolute_error,mean_squared_error,mean_squared_error,balanced_accuracy_score,explained_variance_score\n")
def format_decimal(number):
    return f"{number:.{3}f}"
def result(y_pred,y_test,algo,time_to_train,time_to_predict):
    file.write(algo+",")
    file.write(str(format_decimal(time_to_train))+",")
    file.write(str(format_decimal(time_to_predict))+",")
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


# Initialize Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits= n_splits_for_cv, random_state=42, shuffle=True)

## **1.Decision Tree**
# Initialize lists to store metrics
all_y_test = []
all_y_pred = []
start_cv = time.time()
time_to_predict = 0
time_to_train = 0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Decision Tree Classifier for this fold
    dt_multi = DecisionTreeClassifier(random_state=24)
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fit the model
    start_train = time.time()
    dt_multi.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train - start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = dt_multi.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict +=  end_predict - start_predict
    # Append metrics to lists
    all_y_test.extend(y_test_fold)
    all_y_pred.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = multilabel_confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix")
    i = str(fold_number)
    pname = method + "_fold_"+ i + "_Decision_Tree_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv = time.time()
result(all_y_pred, all_y_test, "DT", time_to_train, time_to_predict)

print(time_to_train)
print(time_to_predict)
print(end_cv-start_cv)

## **2.Linear Regression**
# Initialize lists to store metrics
all_y_test_lr = []
all_y_pred_lr = []
start_cv_lr = time.time()
time_to_predict = 0
time_to_train = 0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Linear Regression model for this fold
    lr_multi = LinearRegression()
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fit the model
    start_train = time.time()
    lr_multi.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train+=end_train - start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = lr_multi.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict +=  end_predict - start_predict
    for i in range(len(y_pred_fold)):
        y_pred_fold[i] = int(np.round_(y_pred_fold[i]))
    # Append metrics to lists
    all_y_test_lr.extend(y_test_fold)
    all_y_pred_lr.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Linear Regression")
    pname = method + "_fold_"+ str(fold_number) + "_Linear_Regression_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_lr = time.time()
result(all_y_pred_lr, all_y_test_lr, "Linear Regression", time_to_train, time_to_predict)

## **3.Logistic Regression**
# Initialize lists to store metrics
all_y_test_logreg = []
all_y_pred_logreg = []
start_cv_logreg = time.time()
time_to_predict = 0
time_to_train  = 0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Logistic Regression model for this fold
    logreg_multi =LogisticRegression(random_state=123, max_iter=5000,solver='newton-cg',multi_class='multinomial')
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fit the model
    start_train = time.time()
    logreg_multi.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train - start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = logreg_multi.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict +=  end_predict - start_predict
    # Append metrics to lists
    all_y_test_logreg.extend(y_test_fold)
    all_y_pred_logreg.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Logistic Regression")
    pname = method + "_fold_"+ str(fold_number) + "_Logistic_Regression_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_logreg = time.time()
result(all_y_pred_logreg, all_y_test_logreg, "Logistic Regression", time_to_train, time_to_predict)

## **4.K Nearest Neighbor Classifier**
# Initialize lists to store metrics
all_y_test_knn = []
all_y_pred_knn = []
start_cv_knn = time.time()
time_to_predict = 0
time_to_train = 0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize KNN model for this fold
    knn = KNeighborsClassifier(8)
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fit the model
    start_train = time.time()
    knn.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train - start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = knn.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_knn.extend(y_test_fold)
    all_y_pred_knn.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - KNN")
    pname = method + "_fold_"+ str(fold_number) + "_KNN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_knn = time.time()
result(all_y_pred_knn, all_y_test_knn, "KNN", time_to_train, time_to_predict)

## **5.Random Forest Classifier**
# Initialize lists to store metrics
all_y_test_rf = []
all_y_pred_rf = []
start_cv_rf = time.time()
time_to_predict=0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Random Forest model for this fold
    rf = RandomForestClassifier(random_state=24)
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fit the model
    start_train = time.time()
    rf.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = rf.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict += end_predict -start_predict
    # Append metrics to lists
    all_y_test_rf.extend(y_test_fold)
    all_y_pred_rf.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Random Forest")
    pname = method + "_fold_"+ str(fold_number) + "_Random_Forest_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_rf = time.time()
result(all_y_pred_rf, all_y_test_rf, "Random Forest", time_to_train, time_to_predict)

## **6.Multi Layer Perceptron**
# Initialize lists to store metrics
all_y_test_mlp = []
all_y_pred_mlp = []
start_cv_mlp = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize MLP model for this fold
    mlp = MLPClassifier(random_state=24)
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fit the model
    start_train = time.time()
    mlp.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = mlp.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_mlp.extend(y_test_fold)
    all_y_pred_mlp.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - MLP")
    pname = method + "_fold_"+ str(fold_number) + "_MLP_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_mlp = time.time()
result(all_y_pred_mlp, all_y_test_mlp, "MLP", time_to_train, time_to_predict)

## **7.Bagging**
# Initialize lists to store metrics
all_y_test_bagging = []
all_y_pred_bagging = []
start_cv_bagging = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Create a base classifier (Decision Tree)
    base_classifier = DecisionTreeClassifier(random_state=42)
    # Create a bagging classifier
    bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the bagging classifier
    start_train = time.time()
    bagging_classifier.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = bagging_classifier.predict(X_test_fold)
    end_predict = time.time()
    time_to_predict += end_predict-start_predict
    # Append metrics to lists
    all_y_test_bagging.extend(y_test_fold)
    all_y_pred_bagging.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize']=8,8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Bagging")
    pname = method + "_fold_"+ str(fold_number) + "_Bagging_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_bagging = time.time()
result(all_y_pred_bagging, all_y_test_bagging, "Bagging", time_to_train, time_to_predict)

## **8. J48 (C4.5)**
# Initialize lists to store metrics
all_y_test_j48 = []
all_y_pred_j48 = []
start_cv_j48 = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize J48 (C4.5) classifier
    classifier_j48 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the J48 (C4.5) classifier
    start_train = time.time()
    classifier_j48.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = classifier_j48.predict(X_test_fold)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_j48.extend(y_test_fold)
    all_y_pred_j48.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - J48 (C4.5)")
    pname = method + "_fold_" + str(fold_number) + "_J48_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_j48 = time.time()
result(all_y_pred_j48, all_y_test_j48, "J48", time_to_train, time_to_predict)

## **9. ANN**
# Initialize lists to store metrics
all_y_test_ann = []
all_y_pred_ann = []
start_cv_ann = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize ANN model
    multi_ann = Sequential()
    # Adding the input layer and the first hidden layer
    multi_ann.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    # Adding the second hidden layer
    multi_ann.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    multi_ann.add(Dense(units=len(np.unique(y_train)), kernel_initializer='uniform', activation='softmax'))
    # Compiling the ANN
    multi_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fitting the ANN to the Training set
    start_train = time.time()
    history = multi_ann.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(multi_ann.predict(X_test_fold), axis=1)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_ann.extend(y_test_fold)
    all_y_pred_ann.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - ANN")
    pname = method + "_fold_" + str(fold_number) + "_ANN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_ann = time.time()
result(all_y_pred_ann, all_y_test_ann, "ANN", time_to_train, time_to_predict)

## **10. DNN**
# Initialize lists to store metrics
all_y_test_dnn = []
all_y_pred_dnn = []
start_cv_dnn = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize DNN model
    multi_dnn = Sequential()
    # Adding the input layer and the first hidden layer
    multi_dnn.add(Dense(units=19, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))
    # Adding the second hidden layer
    multi_dnn.add(Dense(units=19, kernel_initializer='uniform', activation='relu'))
    # Adding the third hidden layer
    multi_dnn.add(Dense(units=19, kernel_initializer='uniform', activation='relu'))
    # Adding the output layer
    multi_dnn.add(Dense(units=len(np.unique(y_train)), kernel_initializer='uniform', activation='softmax'))
    # Compiling the DNN
    multi_dnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fitting the DNN to the Training set
    start_train = time.time()
    history = multi_dnn.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(multi_dnn.predict(X_test_fold), axis=1)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_dnn.extend(y_test_fold)
    all_y_pred_dnn.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - DNN")
    pname = method + "_fold_" + str(fold_number) + "_DNN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_dnn = time.time()
result(all_y_pred_dnn, all_y_test_dnn, "DNN", time_to_train, time_to_predict)

## **11. CNN**
# Initialize lists to store metrics
all_y_test_cnn = []
all_y_pred_cnn = []
start_cv_cnn = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize CNN model
    multi_cnn = Sequential()
    multi_cnn.add(Reshape((-1,1), input_shape=(X_train.shape[1],)))
    multi_cnn.add(Conv1D(32, 3, activation='relu', padding='causal'))
    multi_cnn.add(Conv1D(64, 3, activation='relu', padding='causal'))
    multi_cnn.add(MaxPooling1D(pool_size=2))
    multi_cnn.add(Flatten())
    multi_cnn.add(Dense(len(np.unique(y_train)),activation='softmax'))
    multi_cnn.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Fitting the CNN to the Training set
    start_train = time.time()
    history = multi_cnn.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(multi_cnn.predict(X_test_fold), axis=1)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_cnn.extend(y_test_fold)
    all_y_pred_cnn.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - CNN")
    pname = method + "_fold_" + str(fold_number) + "_CNN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_cnn = time.time()
result(all_y_pred_cnn, all_y_test_cnn, "CNN", time_to_train, time_to_predict)

## **12. Gradient Boosting Classifier with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_gb = []
all_y_pred_gb = []
start_cv_gb = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Gradient Boosting Classifier
    multi_gb = GradientBoostingClassifier()
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model
    start_train = time.time()
    multi_gb.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = multi_gb.predict(X_test_fold)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_gb.extend(y_test_fold)
    all_y_pred_gb.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Gradient Boosting")
    pname = method + "_fold_" + str(fold_number) + "_GradientBoostingClassifier_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_gb = time.time()
result(all_y_pred_gb, all_y_test_gb, "Gradient Boosting", time_to_train, time_to_predict)

## ** 13 XGBoost Classifier**
# Initialize lists to store metrics
all_y_test_xgb = []
all_y_pred_xgb = []
start_cv_xgb = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize XGBoost Classifier
    xgb_model = XGBClassifier()
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model
    start_train = time.time()
    xgb_model.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = xgb_model.predict(X_test_fold)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_xgb.extend(y_test_fold)
    all_y_pred_xgb.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - XGBoost")
    pname = method + "_fold_" + str(fold_number) + "_XGBClassifier_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_xgb = time.time()
result(all_y_pred_xgb, all_y_test_xgb, "XGBoost", time_to_train, time_to_predict)

## **14. Gaussian Naive Bayes**
# Initialize lists to store metrics
all_y_test_nb = []
all_y_pred_nb = []
start_cv_nb = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Gaussian Naive Bayes model
    NB_model = GaussianNB()
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model
    start_train = time.time()
    NB_model.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = NB_model.predict(X_test_fold)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_nb.extend(y_test_fold)
    all_y_pred_nb.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Gaussian Naive Bayes")
    pname = method + "_fold_" + str(fold_number) + "_Gaussian_Naive_Bayes_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_nb = time.time()
result(all_y_pred_nb, all_y_test_nb, "Gaussian Naive Bayes", time_to_train, time_to_predict)

## **15. Adaptive Gradient Boosting**
# Initialize lists to store metrics
all_y_test_ab = []
all_y_pred_ab = []
start_cv_ab = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Adaptive Gradient Boosting model
    weak_learner = DecisionTreeClassifier(max_leaf_nodes=8)
    n_estimators = 300
    AB_model = AdaBoostClassifier(
        estimator=weak_learner,
        n_estimators=n_estimators,
        algorithm="SAMME",
        random_state=42,
    )
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model
    start_train = time.time()
    AB_model.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = AB_model.predict(X_test_fold)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_ab.extend(y_test_fold)
    all_y_pred_ab.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Adaptive Gradient Boosting")
    pname = method + "_fold_" + str(fold_number) + "_Adaptive_Gradient_Boosting_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_ab = time.time()
result(all_y_pred_ab, all_y_test_ab, "Adaptive Gradient Boosting", time_to_train, time_to_predict)

## **16. Quadratic Discriminant Analysis (QDA)**
# Initialize lists to store metrics
all_y_test_qda = []
all_y_pred_qda = []
start_cv_qda = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Initialize Quadratic Discriminant Analysis (QDA) model
    qda_multi = QuadraticDiscriminantAnalysis()
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model
    start_train = time.time()
    qda_multi.fit(X_train_fold, y_train_fold)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = qda_multi.predict(X_test_fold)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_qda.extend(y_test_fold)
    all_y_pred_qda.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Quadratic Discriminant Analysis (QDA)")
    pname = method + "_fold_" + str(fold_number) + "_QDA_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_qda = time.time()
result(all_y_pred_qda, all_y_test_qda, "QDA", time_to_train, time_to_predict)

## **17. Shallow Neural Network (SNN)**
# Initialize lists to store metrics
all_y_test_snn = []
all_y_pred_snn = []
start_cv_snn = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    num_classes =  len(np.unique(y_train))
    # Initialize Shallow Neural Network (SNN) model
    snn_multi = Sequential()
    snn_multi.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    snn_multi.add(Dense(32, activation='relu'))
    snn_multi.add(Dense(20, activation='relu'))
    snn_multi.add(Dense(num_classes, activation='softmax'))
    snn_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Train the model
    start_train = time.time()
    history = snn_multi.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(snn_multi.predict(X_test_fold), axis=1)
    end_predict = time.time()
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict
    # Append metrics to lists
    all_y_test_snn.extend(y_test_fold)
    all_y_pred_snn.extend(y_pred_fold)
    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Shallow Neural Network (SNN)")
    pname = method + "_fold_" + str(fold_number) + "_SNN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_snn = time.time()
result(all_y_pred_snn, all_y_test_snn, "SNN", time_to_train, time_to_predict)

## **18. Restricted Boltzmann Machine (RBM)**
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

# Initialize lists to store metrics
all_y_test_rbm = []
all_y_pred_rbm = []
start_cv_rbm = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    num_classes = len(np.unique(y_train))
    input_data = Input(shape=(X_train.shape[1],))
    rbm1_visible, rbm1_hidden = RBM(hidden_dim=128, name=f"rbm1_fold_{fold_number}")(input_data)
    rbm2_visible, rbm2_hidden = RBM(hidden_dim=64, name=f"rbm2_fold_{fold_number}")(rbm1_hidden)
    rbm3_visible, rbm3_hidden = RBM(hidden_dim=32, name=f"rbm3_fold_{fold_number}")(rbm2_hidden)
    rbm6_visible, rbm6_hidden = RBM(hidden_dim=64, name=f"rbm6_fold_{fold_number}")(rbm3_hidden)
    classifier_output = Dense(num_classes, activation='softmax', name=f'classifier_fold_{fold_number}')(rbm6_hidden)
    model = tf.keras.Model(inputs=input_data, outputs=classifier_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Train the model
    start_train = time.time()
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(model.predict(X_test_fold), axis=1)
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_rbm.extend(y_test_fold)
    all_y_pred_rbm.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Restricted Boltzmann Machine (RBM)")
    pname = method + "_fold_" + str(fold_number) + "_RBM_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_rbm = time.time()
result(all_y_pred_rbm, all_y_test_rbm, "RBM", time_to_train, time_to_predict)

## **19. LSTM**

# reloading as many transformations on X,Y causing errors for lstm code
X_train = train_data.drop(columns=['label'],axis=1)
X_test = test_data.drop(columns=['label'],axis=1)
y_train = train_data['label']
y_test = test_data['label']
X_train = pd.concat([X_train, X_test], axis=0)
y_train = pd.concat([y_train, y_test], axis=0)
# Reset indices of X_train and y_train
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# Initialize lists to store metrics
all_y_test_lstm = []
all_y_pred_lstm = []
start_cv_lstm = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    num_classes = len(np.unique(y_train))

    # Convert DataFrame to NumPy array and reshape input data for LSTM
    X_train_array_multi = X_train.iloc[train_index].to_numpy()
    X_test_array_multi = X_train.iloc[test_index].to_numpy()
    X_train_reshaped_multi = X_train_array_multi.reshape((X_train_array_multi.shape[0], X_train_array_multi.shape[1], 1))
    X_test_reshaped_multi = X_test_array_multi.reshape((X_test_array_multi.shape[0], X_test_array_multi.shape[1], 1))

    # Define the LSTM model
    rnn_multi = Sequential()
    rnn_multi.add(LSTM(128, input_shape=(X_train_reshaped_multi.shape[1], X_train_reshaped_multi.shape[2])))
    rnn_multi.add(Dense(32, activation='relu'))
    rnn_multi.add(Dense(num_classes, activation='softmax'))
    rnn_multi.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the LSTM model
    start_train = time.time()
    rnn_multi.fit(X_train_reshaped_multi, y_train.iloc[train_index], validation_data=(X_test_reshaped_multi, y_train.iloc[test_index]), epochs=10, batch_size=50, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(rnn_multi.predict(X_test_reshaped_multi), axis=1)
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_lstm.extend(y_train.iloc[test_index])
    all_y_pred_lstm.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_train.iloc[test_index], y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - LSTM")
    pname = method + "_fold_" + str(fold_number) + "_LSTM_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_lstm = time.time()
result(all_y_pred_lstm, all_y_test_lstm, "LSTM",time_to_train , time_to_predict)

## **20. Reconstruction Neural Networks**
# Initialize lists to store metrics
all_y_test_recon = []
all_y_pred_recon = []
start_cv_recon = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    num_classes = len(np.unique(y_train))

    # Assuming y_train_multi is one-hot encoded
    y_train_multi_onehot = tf.keras.utils.to_categorical(y_train.iloc[train_index], num_classes=num_classes)
    y_test_multi_onehot = tf.keras.utils.to_categorical(y_train.iloc[test_index], num_classes=num_classes)

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

    # Train the VAE model
    start_train = time.time()
    vae.fit(X_train.iloc[train_index], y_train_multi_onehot, epochs=10, batch_size=50, shuffle=True, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict
    start_predict = time.time()
    y_pred_fold = np.argmax(vae.predict(X_train.iloc[test_index]), axis=1)
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_recon.extend(y_train.iloc[test_index])
    all_y_pred_recon.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_train.iloc[test_index], y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Reconstruction Neural Network")
    pname = method + "_fold_" + str(fold_number) + "_reconstruction_NN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_recon = time.time()
result(all_y_pred_recon, all_y_test_recon, "reconstruction_NN", time_to_train, time_to_predict)

## **21. ELM with k-fold Cross-Validation**
class ELMClassifier:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.input_weights = None
        self.bias = None
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

# Initialize lists to store metrics
all_y_test_elm = []
all_y_pred_elm = []

start_cv_elm = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Specify the number of hidden layer neurons and output classes
    input_size = X_train.shape[1]
    hidden_layer_size = 500  # You can adjust this parameter
    output_size = len(np.unique(y_train))

    # Create and initialize the ELM classifier
    elm_classifier = ELMClassifier(input_size, hidden_layer_size, output_size)

    # Training phase
    start_train = time.time()
    elm_classifier.input_weights = np.random.randn(hidden_layer_size, input_size)
    elm_classifier.bias = np.random.randn(hidden_layer_size, 1)
    elm_classifier.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    end_train = time.time()
    time_to_train += end_train -start_train
    # Prediction phase
    start_predict = time.time()
    y_pred_fold = elm_classifier.predict(X_train.iloc[test_index])
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_elm.extend(y_train.iloc[test_index])
    all_y_pred_elm.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_train.iloc[test_index], y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - ELM")
    pname = method + "_fold_" + str(fold_number) + "_ELM_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_elm = time.time()
result(all_y_pred_elm, all_y_test_elm, "ELM", time_to_train, time_to_predict)

## **22. DANN with k-fold Cross-Validation**

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

# Initialize lists to store metrics
all_y_test_dann = []
all_y_pred_dann = []
start_cv_dann = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Convert class vectors to binary class matrices
    num_classes = len(np.unique(y_train))
    y_train_categorical = tf.keras.utils.to_categorical(y_train.iloc[train_index], num_classes)
    y_test_categorical = tf.keras.utils.to_categorical(y_train.iloc[test_index], num_classes)

    # Build and train DANN model for each fold
    input_shape = (X_train.shape[1],)
    lambda_val = 1e-3  # Trade-off parameter for domain adversarial loss
    dann_model = build_dann_model(input_shape, num_classes, lambda_val)

    # Training phase
    start_train = time.time()
    dann_model.fit(X_train.iloc[train_index],
                   {'source_classifier': y_train_categorical, 'domain_classifier': np.zeros((len(train_index), 1))},
                   epochs=10, batch_size=64,
                   validation_data=(X_train.iloc[test_index],
                                    {'source_classifier': y_test_categorical,
                                     'domain_classifier': np.ones((len(test_index), 1))}))
    end_train = time.time()
    time_to_train += end_train -start_train
    # Prediction phase
    X_test_fold = X_train.iloc[test_index]  # Use iloc to access test fold
    start_predict = time.time()
    predictions = dann_model.predict(X_test_fold)
    source_classifier_predictions = predictions[0]
    y_pred_fold = np.argmax(source_classifier_predictions, axis=1)
    end_predict = time.time()

    y_test = y_train.iloc[test_index]
    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_dann.extend(y_test)
    all_y_pred_dann.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_test, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - DANN")
    pname = method + "_fold_" + str(fold_number) + "_DANN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_dann = time.time()
result(all_y_pred_dann, all_y_test_dann, "DANN", time_to_train, time_to_predict)

## **23.Deep brief networks (DBNs)**
# Initialize lists to store metrics
all_y_test_dbn = []
all_y_pred_dbn = []
start_cv_dbn = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Number of classes
    num_classes = len(np.unique(y_train))

    # Create a pipeline with BernoulliRBM and MLPClassifier
    rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=20, random_state=42, verbose=True)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10, random_state=52)
    dbn_model = Pipeline(steps=[('rbm', rbm), ('mlp', mlp)])

    # Training phase
    start_train = time.time()
    dbn_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    end_train = time.time()
    time_to_train += end_train -start_train
    # Prediction phase
    X_test_fold = X_train.iloc[test_index]
    y_test_fold = y_train.iloc[test_index]
    start_predict = time.time()
    y_pred_fold = dbn_model.predict(X_test_fold)
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_dbn.extend(y_test_fold)
    all_y_pred_dbn.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - DBN")
    pname = method + "_fold_" + str(fold_number) + "_DBN_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_dbn = time.time()
result(all_y_pred_dbn, all_y_test_dbn, "DBN", time_to_train, time_to_predict)

## **24. Deep Boltzmann Machines (DBMs)** with k-fold Cross-Validation
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

# Initialize lists to store metrics
all_y_test_dbm = []
all_y_pred_dbm = []
start_cv_dbm = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold, X_test_fold = X_train.iloc[train_index].values, X_train.iloc[test_index].values
    y_train_fold, y_test_fold = y_train.iloc[train_index].values, y_train.iloc[test_index].values

    # Number of visible and hidden units
    visible_dim = X_train_fold.shape[1]
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
        for i in range(0, len(X_train_fold), batch_size):
            batch_data = X_train_fold[i:i+batch_size]
            rbm1.contrastive_divergence(tf.cast(batch_data, dtype=tf.float32))

    # Getting hidden layer representation from the first RBM
    hidden1_representation = tf.nn.relu(tf.sign(rbm1.sample_hidden(tf.cast(X_train_fold, dtype=tf.float32))))

    # Training second RBM using the hidden layer representation from the first RBM
    for epoch in range(num_epochs):
        for i in range(0, len(hidden1_representation), batch_size):
            batch_data = hidden1_representation[i:i+batch_size]
            rbm2.contrastive_divergence(batch_data)

    # Getting hidden layer representation from the second RBM
    hidden2_representation = tf.nn.relu(tf.sign(rbm2.sample_hidden(hidden1_representation)))

    # Fine-tuning for classification
    num_classes = len(np.unique(y_train_fold))
    dbm_model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim1, activation='relu'),
        tf.keras.layers.Dense(hidden_dim2, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    dbm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    start_train = time.time()
    dbm_model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, shuffle=True, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train

    # Predict on the test set
    start_predict = time.time()
    y_pred_probabilities = dbm_model.predict(X_test_fold)
    y_pred = np.argmax(y_pred_probabilities, axis=1)
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_dbm.extend(y_test_fold)
    all_y_pred_dbm.extend(y_pred)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - DBM")
    pname = method + "_fold_" + str(fold_number) + "_DBM_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_dbm = time.time()
result(all_y_pred_dbm, all_y_test_dbm, "DBM", time_to_train, time_to_predict)

## **25.DEEP AUTO ENCODERS(DA)**
# Initialize lists to store metrics
all_y_test_da = []
all_y_pred_da = []
start_cv_da = time.time()
time_to_predict = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Define the autoencoder model
    autoencoder = Sequential()

    # Encoder
    autoencoder.add(Dense(128, activation='relu', input_shape=(X_train_fold.shape[1],)))
    autoencoder.add(Dense(64, activation='relu'))
    autoencoder.add(Dense(32, activation='relu'))

    # Decoder
    autoencoder.add(Dense(64, activation='relu'))
    autoencoder.add(Dense(128, activation='relu'))
    autoencoder.add(Dense(X_train_fold.shape[1], activation='linear'))

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(X_train_fold, X_train_fold, epochs=10, batch_size=50, verbose=0)

    # Add a classification head on top of the trained autoencoder
    da_model = Sequential()
    da_model.add(autoencoder.layers[0])  # Add encoder layers
    da_model.add(autoencoder.layers[1])
    da_model.add(autoencoder.layers[2])
    da_model.add(Dense(num_classes, activation='softmax'))  # Adjust output layer for multiple classes

    # Compile the classification model
    da_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Convert labels to one-hot encoding
    y_train_fold_onehot = to_categorical(y_train_fold, num_classes=num_classes)
    y_test_fold_onehot = to_categorical(y_test_fold, num_classes=num_classes)

    # Train the classification model using the encoded representations
    start_train = time.time()
    history = da_model.fit(X_train_fold, y_train_fold_onehot, epochs=10, batch_size=32, shuffle=True, verbose=0)
    end_train = time.time()
    time_to_train += end_train -start_train
    # Predict on the test set
    start_predict = time.time()
    y_pred_probabilities = da_model.predict(X_test_fold)
    y_pred_fold = np.argmax(y_pred_probabilities, axis=1)
    end_predict = time.time()

    # Calculate time taken for prediction
    time_to_predict += end_predict - start_predict

    # Append metrics to lists
    all_y_test_da.extend(y_test_fold)
    all_y_pred_da.extend(y_pred_fold)

    # Generate confusion matrix and display
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - DA")
    pname = method + "_fold_" + str(fold_number) + "_DA_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_da = time.time()
result(all_y_pred_da, all_y_test_da, "DA", time_to_train, time_to_predict)

## **26. PassiveAggressiveClassifier with k-fold Cross-Validation**

# Initialize lists to store metrics
all_y_test_passive = []
all_y_pred_passive = []

start_cv_passive = time.time()
time_to_predict_passive = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold_passive, X_test_fold_passive = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_passive, y_test_fold_passive = y_train.iloc[train_index], y_train.iloc[test_index]

    # Initialize PassiveAggressiveClassifier for each fold
    model_passive = PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3)

    # Train the model
    start_train_passive = time.time()
    model_passive.fit(X_train_fold_passive, y_train_fold_passive)
    end_train_passive = time.time()
    time_to_train += end_train_passive -start_train_passive
    # Predict on the test set
    start_predict_passive = time.time()
    y_pred_passive = model_passive.predict(X_test_fold_passive)
    end_predict_passive = time.time()
    time_to_predict_passive += end_predict_passive - start_predict_passive

    # Append metrics to lists
    all_y_test_passive.extend(y_test_fold_passive)
    all_y_pred_passive.extend(y_pred_passive)

    # Generate confusion matrix and display
    cm_passive = confusion_matrix(y_test_fold_passive, y_pred_passive)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_passive = ConfusionMatrixDisplay(confusion_matrix=cm_passive)
    disp_passive.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - PassiveAggressiveClassifier")
    pname_passive = method + "_fold_" + str(fold_number) + "_PassiveAggressiveClassifier_confusion_matrix.png"
    plt.savefig(pname_passive)
    #plt.show()

# Calculate overall performance metrics
end_cv_passive = time.time()
result(all_y_pred_passive, all_y_test_passive, "PassiveAggressiveClassifier", time_to_train, time_to_predict_passive)

## **27. RidgeClassifier with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_ridge = []
all_y_pred_ridge = []
start_cv_ridge = time.time()
time_to_predict_ridge = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train.values, y_train.values), 1):
    # Split data into train and test sets
    X_train_fold_ridge, X_test_fold_ridge = X_train.values[train_index], X_train.values[test_index]
    y_train_fold_ridge, y_test_fold_ridge = y_train.values[train_index], y_train.values[test_index]

    # Initialize RidgeClassifier for each fold
    model_ridge = RidgeClassifier()

    # Train the model
    start_train_ridge = time.time()
    model_ridge.fit(X_train_fold_ridge, y_train_fold_ridge)
    end_train_ridge = time.time()
    time_to_train += end_train_ridge -start_train_ridge
    # Predict    on the test set
    start_predict_ridge = time.time()
    y_pred_ridge = model_ridge.predict(X_test_fold_ridge)
    end_predict_ridge = time.time()

    time_to_predict_ridge += end_predict_ridge - start_predict_ridge
    # Append metrics to lists
    all_y_test_ridge.extend(y_test_fold_ridge)
    all_y_pred_ridge.extend(y_pred_ridge)

    # Generate confusion matrix and display
    cm_ridge = confusion_matrix(y_test_fold_ridge, y_pred_ridge)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_ridge = ConfusionMatrixDisplay(confusion_matrix=cm_ridge)
    disp_ridge.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - RidgeClassifier")
    pname_ridge = method +"_fold_"+ str(fold_number) + "_RidgeClassifier_confusion_matrix.png"
    plt.savefig(pname_ridge)
    #plt.show()

# Calculate overall performance metrics
end_cv_ridge = time.time()
result(all_y_pred_ridge, all_y_test_ridge, "RidgeClassifier", time_to_train, time_to_predict_ridge)

## **28. NearestCentroid with k-fold Cross-Validation, Time to Predict, and Confusion Matrix**

# Initialize lists to store metrics
all_y_test_nc = []
all_y_pred_nc = []
start_cv_nc = time.time()
total_time_to_predict_nc = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold_nc, X_test_fold_nc = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_nc, y_test_fold_nc = y_train.iloc[train_index], y_train.iloc[test_index]

    # Initialize NearestCentroid model for each fold
    model_nc = NearestCentroid()

    # Start time for training
    start_train_nc = time.time()

    # Train the model
    model_nc.fit(X_train_fold_nc, y_train_fold_nc)

    # End time for training
    end_train_nc = time.time()
    time_to_train += end_train_nc - start_train_nc
    # Start time for prediction
    start_predict_nc = time.time()

    # Make predictions on the test set
    y_pred_fold_nc = model_nc.predict(X_test_fold_nc)

    # End time for prediction
    end_predict_nc = time.time()

    # Calculate time taken for prediction
    time_to_predict_nc = end_predict_nc - start_predict_nc
    total_time_to_predict_nc += time_to_predict_nc

    # Append metrics to lists
    all_y_test_nc.extend(y_test_fold_nc)
    all_y_pred_nc.extend(y_pred_fold_nc)

    # Generate confusion matrix and display
    cm_nc = confusion_matrix(y_test_fold_nc, y_pred_fold_nc)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_nc = ConfusionMatrixDisplay(confusion_matrix=cm_nc)
    disp_nc.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - NearestCentroid")
    pname_nc = method+"_fold_" + str(fold_number) + "_NearestCentroid_confusion_matrix.png"
    plt.savefig(pname_nc)
    #plt.show()

# End time for k-fold cross-validation
end_cv_nc = time.time()
# Calculate overall performance metrics
result(all_y_pred_nc, all_y_test_nc, "NearestCentroid", time_to_train, total_time_to_predict_nc)

## **29. Cost Sensitive Logistic Regression (CSLR) with k-fold Cross-Validation and Time to Predict**

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

# Initialize lists to store metrics
all_y_test_cslr = []
all_y_pred_cslr = []
# Define cost matrix for CSLR
cost_matrix = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],    # Costs for misclassifying class 0
                        [1, 0, 1, 2, 3, 4, 5, 6, 7],    # Costs for misclassifying class 1
                        [2, 1, 0, 1, 2, 3, 4, 5, 6],    # Costs for misclassifying class 2
                        [3, 2, 1, 0, 1, 2, 3, 4, 5],    # Costs for misclassifying class 3
                        [4, 3, 2, 1, 0, 1, 2, 3, 4],    # Costs for misclassifying class 4
                        [5, 4, 3, 2, 1, 0, 1, 2, 3],    # Costs for misclassifying class 5
                        [6, 5, 4, 3, 2, 1, 0, 1, 2],    # Costs for misclassifying class 6
                        [7, 6, 5, 4, 3, 2, 1, 0, 1],    # Costs for misclassifying class 7
                        [8, 7, 6, 5, 4, 3, 2, 1, 0]])   # Costs for misclassifying class 8

start_cv_cslr = time.time()
total_time_to_predict_fold_cslr = 0
time_to_train = 0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold_cslr, X_test_fold_cslr = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_cslr, y_test_fold_cslr = y_train.iloc[train_index], y_train.iloc[test_index]

    # Get sample weights for cost-sensitive learning
    sample_weights_fold_cslr = get_sample_weight(cost_matrix, y_train_fold_cslr)

    # Initialize Logistic Regression model for each fold
    model_cslr = LogisticRegression(solver='lbfgs')

    # Start time for training
    start_train_fold_cslr = time.time()

    # Train the model
    model_cslr.fit(X_train_fold_cslr, y_train_fold_cslr, sample_weight=sample_weights_fold_cslr)

    # End time for training
    end_train_fold_cslr = time.time()

    time_to_train +=  end_train_fold_cslr - start_train_fold_cslr
    # Start time for prediction
    start_predict_fold_cslr = time.time()

    # Make predictions on the test set
    y_pred_fold_cslr = model_cslr.predict(X_test_fold_cslr)

    # End time for prediction
    end_predict_fold_cslr = time.time()

    # Calculate time taken for prediction in this fold
    time_to_predict_fold_cslr = end_predict_fold_cslr - start_predict_fold_cslr
    total_time_to_predict_fold_cslr += time_to_predict_fold_cslr

    # Append metrics to lists
    all_y_test_cslr.extend(y_test_fold_cslr)
    all_y_pred_cslr.extend(y_pred_fold_cslr)

    # Generate confusion matrix and display
    cm_cslr = confusion_matrix(y_test_fold_cslr, y_pred_fold_cslr)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_cslr = ConfusionMatrixDisplay(confusion_matrix=cm_cslr)
    disp_cslr.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - CSLR")
    pname_cslr = method +"_fold_" + str(fold_number) + "_CSLR__confusion_matrix.png"
    plt.savefig(pname_cslr)
    #plt.show()

# End time for k-fold cross-validation
end_cv_cslr = time.time()
result(all_y_pred_cslr, all_y_test_cslr, "CSLR", time_to_train, total_time_to_predict_fold_cslr)

## **30. Cost-sensitive Bagging Classifier (CSBC) with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_csbc = []
all_y_pred_csbc = []
start_cv_csbc = time.time()
time_to_predict_fold_csbc = 0
time_to_train = 0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold_csbc, X_test_fold_csbc = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_csbc, y_test_fold_csbc = y_train.iloc[train_index], y_train.iloc[test_index]

    # Step 1: Compute class weights for the fold
    class_weights_fold = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_fold_csbc), y=y_train_fold_csbc)

    # Step 2: Initialize the base estimator and BaggingClassifier for the fold
    base_estimator = DecisionTreeClassifier(max_depth=5)
    bagging_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)

    # Step 3: Train the model on the fold
    start_train_fold_csbc = time.time()
    bagging_model.fit(X_train_fold_csbc, y_train_fold_csbc)
    end_train_fold_csbc = time.time()
    time_to_train += end_train_fold_csbc - start_train_fold_csbc
    # Step 4: Predict on the test set for the fold
    start_predict_fold_csbc = time.time()
    y_pred_fold_csbc = bagging_model.predict(X_test_fold_csbc)
    end_predict_fold_csbc = time.time()

    time_to_predict_fold_csbc += end_predict_fold_csbc - start_predict_fold_csbc

    # Append metrics to lists
    all_y_test_csbc.extend(y_test_fold_csbc)
    all_y_pred_csbc.extend(y_pred_fold_csbc)

    # Generate confusion matrix and display for the fold
    cm_csbc = confusion_matrix(y_test_fold_csbc, y_pred_fold_csbc)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_csbc = ConfusionMatrixDisplay(confusion_matrix=cm_csbc)
    disp_csbc.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - CSBC")
    pname_csbc = method + "_fold_" + str(fold_number) + "_CSBC_confusion_matrix.png"
    plt.savefig(pname_csbc)
    #plt.show()

# Calculate overall performance metrics
end_cv_csbc = time.time()
result(all_y_pred_csbc, all_y_test_csbc, "CSBC", time_to_train, time_to_predict_fold_csbc)

from lightgbm import LGBMClassifier

## **31. LightGBM with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_lgbm = []
all_y_pred_lgbm = []
start_cv_lgbm = time.time()
time_to_predict_fold_lgbm = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold_lgbm, X_test_fold_lgbm = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_lgbm, y_test_fold_lgbm = y_train.iloc[train_index], y_train.iloc[test_index]

    # Initialize LightGBM Classifier
    lgbm = LGBMClassifier()

    # Train the model on the fold
    start_train_fold_lgbm = time.time()
    lgbm.fit(X_train_fold_lgbm, y_train_fold_lgbm)
    end_train_fold_lgbm = time.time()
    time_to_train += end_train_fold_lgbm  - start_train_fold_lgbm
    # Predict on the test set for the fold
    start_predict_fold_lgbm = time.time()
    y_pred_fold_lgbm = lgbm.predict(X_test_fold_lgbm)
    end_predict_fold_lgbm = time.time()

    time_to_predict_fold_lgbm += end_predict_fold_lgbm - start_predict_fold_lgbm

    # Append metrics to lists
    all_y_test_lgbm.extend(y_test_fold_lgbm)
    all_y_pred_lgbm.extend(y_pred_fold_lgbm)

    # Generate confusion matrix and display for the fold
    cm_lgbm = confusion_matrix(y_test_fold_lgbm, y_pred_fold_lgbm)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_lgbm = ConfusionMatrixDisplay(confusion_matrix=cm_lgbm)
    disp_lgbm.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - LightGBM")
    pname_lgbm = method + "_fold_" + str(fold_number) + "_LightGBM_confusion_matrix.png"
    plt.savefig(pname_lgbm)
    #plt.show()

# Calculate overall performance metrics
end_cv_lgbm = time.time()
result(all_y_pred_lgbm, all_y_test_lgbm, "LightGBM", time_to_train, time_to_predict_fold_lgbm)

## **32. LinearDiscriminantAnalysis (LDA) with k-fold Cross-Validation
# Initialize lists to store metrics
all_y_test_lda = []
all_y_pred_lda = []
start_cv_lda = time.time()
time_to_predict_fold_lda = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets
    X_train_fold_lda, X_test_fold_lda = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_lda, y_test_fold_lda = y_train.iloc[train_index], y_train.iloc[test_index]

    # Apply Linear Discriminant Analysis (LDA) for dimensionality reduction
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_fold_lda = lda.fit_transform(X_train_fold_lda, y_train_fold_lda)
    X_test_fold_lda = lda.transform(X_test_fold_lda)

    # Train Random Forest Classifier on the transformed features
    classifier_lda = RandomForestClassifier(max_depth=2, random_state=0)

    # Train the model on the fold
    start_train_fold_lda = time.time()
    classifier_lda.fit(X_train_fold_lda, y_train_fold_lda)
    end_train_fold_lda = time.time()
    time_to_train += end_train_fold_lda -start_train_fold_lda
    # Predict on the test set for the fold
    start_predict_fold_lda = time.time()
    y_pred_fold_lda = classifier_lda.predict(X_test_fold_lda)
    end_predict_fold_lda = time.time()

    time_to_predict_fold_lda += end_predict_fold_lda - start_predict_fold_lda

    # Append metrics to lists
    all_y_test_lda.extend(y_test_fold_lda)
    all_y_pred_lda.extend(y_pred_fold_lda)

    # Generate confusion matrix and display for the fold
    cm_lda = confusion_matrix(y_test_fold_lda, y_pred_fold_lda)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_lda = ConfusionMatrixDisplay(confusion_matrix=cm_lda)
    disp_lda.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - LDA")
    pname_lda = method + "_fold_" + str(fold_number) + "_LDA_confusion_matrix.png"
    plt.savefig(pname_lda)
    #plt.show()

# Calculate overall performance metrics
end_cv_lda = time.time()
result(all_y_pred_lda, all_y_test_lda, "LDA", time_to_train, time_to_predict_fold_lda)

# **MULTI-CLASS CLASSIFICATION**
## **Data Splitting**
# reloading as many transformations on X,Y causing errors for gru code
X_train = train_data.drop(columns=['label'],axis=1)
X_test = test_data.drop(columns=['label'],axis=1)
y_train = train_data['label']
y_test = test_data['label']
X_train = pd.concat([X_train, X_test], axis=0)
y_train = pd.concat([y_train, y_test], axis=0)
# Reset indices of X_train and y_train
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

## **33. GRU with k-fold Cross-Validation**
num_classes =  len(np.unique(y_train))
X_train_array_multi = X_train.to_numpy()

# Initialize lists to store metrics
all_y_test_gru = []
all_y_pred_gru = []
start_cv_gru = time.time()
time_to_predict_fold = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Convert fold data to numpy arrays and reshape for GRU input
    X_train_fold_array = X_train_fold.to_numpy().reshape((X_train_fold.shape[0], X_train_fold.shape[1], 1))
    X_test_fold_array = X_test_fold.to_numpy().reshape((X_test_fold.shape[0], X_test_fold.shape[1], 1))

    # Define and compile the GRU model
    rnn_fold = Sequential([
        GRU(128, input_shape=(X_train_fold_array.shape[1], X_train_fold_array.shape[2])),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    rnn_fold.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the fold
    start_train_fold = time.time()
    rnn_fold.fit(X_train_fold_array, y_train_fold, epochs=10, batch_size=50, verbose=0)
    end_train_fold = time.time()
    time_to_train += end_train_fold -start_train_fold
    # Predict on the test set for the fold
    start_predict_fold = time.time()
    y_pred_fold = np.argmax(rnn_fold.predict(X_test_fold_array), axis=1)
    end_predict_fold = time.time()

    time_to_predict_fold += end_predict_fold - start_predict_fold

    # Append metrics to lists
    all_y_test_gru.extend(y_test_fold)
    all_y_pred_gru.extend(y_pred_fold)

    # Generate confusion matrix and display for the fold
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - GRU")
    pname = method + "_fold_" + str(fold_number) + "_GRU_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_gru = time.time()
val_accuracy_gru = accuracy_score(all_y_test_gru, all_y_pred_gru)
result(all_y_pred_gru, all_y_test_gru, "GRU", time_to_train, time_to_predict_fold)

## **34. Stochastic Gradient with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_sgd = []
all_y_pred_sgd = []
start_cv_sgd = time.time()
time_to_predict_fold = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    # Define the SGD classifier pipeline with standard scaler
    sgd_fold = make_pipeline(StandardScaler(), SGDClassifier(random_state=24))
    # Measure time to train on the fold
    start_train_fold = time.time()
    sgd_fold.fit(X_train_fold, y_train_fold)
    end_train_fold = time.time()
    time_to_train += end_train_fold -start_train_fold
    # Measure time to predict on the fold
    start_predict_fold = time.time()
    y_pred_fold = sgd_fold.predict(X_test_fold)
    end_predict_fold = time.time()
    time_to_predict_fold += end_predict_fold - start_predict_fold
    # Append metrics to lists
    all_y_test_sgd.extend(y_test_fold)
    all_y_pred_sgd.extend(y_pred_fold)
    # Generate confusion matrix and display for the fold
    cm = confusion_matrix(y_test_fold, y_pred_fold)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Stochastic Gradient")
    pname = method + "_fold_" + str(fold_number) + "_Stochastic_gradient_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()
# Calculate overall performance metrics
end_cv_sgd = time.time()
result(all_y_pred_sgd, all_y_test_sgd, "Stochastic_gradient", time_to_train, time_to_predict_fold)

## **35. Support Vector Machine (SVM) with k-fold Cross-Validation**

# Initialize lists to store metrics
all_y_test_svm = []
all_y_pred_svm = []
start_cv_svm = time.time()
time_to_predict_fold = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Define the SVM classifier pipeline with standard scaler
    svm_fold = make_pipeline(StandardScaler(), SVC(random_state=24))

    # Measure time to train on the fold
    start_train_fold = time.time()
    svm_fold.fit(X_train_fold, y_train_fold)
    end_train_fold = time.time()
    time_to_train += end_train_fold -start_train_fold
    # Measure time to predict on the fold
    start_predict_fold = time.time()
    y_pred_fold_svm = svm_fold.predict(X_test_fold)
    end_predict_fold = time.time()

    time_to_predict_fold += end_predict_fold - start_predict_fold

    # Append metrics to lists
    all_y_test_svm.extend(y_test_fold)
    all_y_pred_svm.extend(y_pred_fold_svm)

    # Generate confusion matrix and display for the fold
    cm = confusion_matrix(y_test_fold, y_pred_fold_svm)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Support Vector Machine (SVM)")
    pname = method + "_fold_" + str(fold_number) + "_SVM_confusion_matrix.png"
    plt.savefig(pname)
    #plt.show()

# Calculate overall performance metrics
end_cv_svm = time.time()
result(all_y_pred_svm, all_y_test_svm, "SVM", time_to_train, time_to_predict_fold)

## **36. Extra Trees Classifier with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_extra_trees = []
all_y_pred_extra_trees = []
start_cv_extra_trees = time.time()
time_to_predict_fold_extra_trees = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Define the Extra Trees classifier
    extra_trees_fold = ExtraTreesClassifier()

    # Measure time to train on the fold
    start_train_fold_extra_trees = time.time()
    extra_trees_fold.fit(X_train_fold, y_train_fold)
    end_train_fold_extra_trees = time.time()
    time_to_train += end_train_fold_extra_trees -start_train_fold_extra_trees
    # Measure time to predict on the fold
    start_predict_fold_extra_trees = time.time()
    y_pred_fold_extra_trees = extra_trees_fold.predict(X_test_fold)
    end_predict_fold_extra_trees = time.time()

    time_to_predict_fold_extra_trees += end_predict_fold_extra_trees - start_predict_fold_extra_trees

    # Append metrics to lists
    all_y_test_extra_trees.extend(y_test_fold)
    all_y_pred_extra_trees.extend(y_pred_fold_extra_trees)

    # Generate confusion matrix and display for the fold
    cm_extra_trees = confusion_matrix(y_test_fold, y_pred_fold_extra_trees)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_extra_trees = ConfusionMatrixDisplay(confusion_matrix=cm_extra_trees)
    disp_extra_trees.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Extra Trees Classifier")
    pname_extra_trees = method + "_fold_" + str(fold_number) + "_ExtraTreesClassifier_confusion_matrix.png"
    plt.savefig(pname_extra_trees)
    #plt.show()

# Calculate overall performance metrics
end_cv_extra_trees = time.time()
result(all_y_pred_extra_trees, all_y_test_extra_trees, "Extra Trees Classifier", time_to_train, time_to_predict_fold_extra_trees)

## **37. Feed Forward Neural Networks with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_ffnn = []
all_y_pred_ffnn = []
start_cv_ffnn = time.time()
time_to_predict_fold_ffnn = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Define the neural network architecture
    model_ffnn = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_fold.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_train_fold)), activation='softmax')
    ])

    # Compile the model
    model_ffnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Measure time to train on the fold
    start_train_fold_ffnn = time.time()
    history_ffnn = model_ffnn.fit(X_train_fold, y_train_fold, epochs=10, batch_size=50, verbose=1)
    end_train_fold_ffnn = time.time()
    time_to_train += end_train_fold_ffnn -start_train_fold_ffnn
    # Measure time to predict on the fold
    start_predict_fold_ffnn = time.time()
    y_pred_fold_ffnn_prob = model_ffnn.predict(X_test_fold)
    y_pred_fold_ffnn = np.argmax(y_pred_fold_ffnn_prob, axis=1)
    end_predict_fold_ffnn = time.time()

    time_to_predict_fold_ffnn += end_predict_fold_ffnn - start_predict_fold_ffnn

    # Append metrics to lists
    all_y_test_ffnn.extend(y_test_fold)
    all_y_pred_ffnn.extend(y_pred_fold_ffnn)

    # Generate confusion matrix and display for the fold
    cm_ffnn = confusion_matrix(y_test_fold, y_pred_fold_ffnn)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_ffnn = ConfusionMatrixDisplay(confusion_matrix=cm_ffnn)
    disp_ffnn.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Feed Forward Neural Networks")
    pname_ffnn = method + "_fold_" + str(fold_number) + "_FFNN_confusion_matrix.png"
    plt.savefig(pname_ffnn)
    #plt.show()

# Calculate overall performance metrics
end_cv_ffnn = time.time()
result(all_y_pred_ffnn, all_y_test_ffnn, "Feed Forward Neural Networks", time_to_train, time_to_predict_fold_ffnn)

## **38. Fuzzy with k-fold Cross-Validation**

all_y_test_fuzzy = []
all_y_pred_fuzzy = []
start_cv_fuzzy = time.time()
time_to_predict_fold_fuzzy = 0
time_to_train=0
# Generate fuzzy c-means clusters
n_clusters = 10  # Number of classes
# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    start_train_fold_fuzzy = time.time()
    # Generate fuzzy c-means clusters for training data of the fold
    centers, u_train_fold, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_train_fold.T, n_clusters, 2, error=0.005, maxiter=1000
    )

    # Measure time to train on the fold
    end_train_fold_fuzzy = time.time()
    time_to_train  += end_train_fold_fuzzy - start_train_fold_fuzzy
    # Predict cluster membership for test data of the fold
    start_predict_fold_fuzzy = time.time()
    u_test_fold, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X_test_fold.T, centers, 2, error=0.005, maxiter=1000
    )
    end_predict_fold_fuzzy = time.time()

    time_to_predict_fold_fuzzy += end_predict_fold_fuzzy - start_predict_fold_fuzzy

    # Assign class labels based on cluster membership
    y_pred_fold_fuzzy = np.argmax(u_test_fold, axis=0)

    # Append metrics to lists
    all_y_test_fuzzy.extend(y_test_fold)
    all_y_pred_fuzzy.extend(y_pred_fold_fuzzy)

    # Generate confusion matrix and display for the fold
    cm_fuzzy = confusion_matrix(y_test_fold, y_pred_fold_fuzzy)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_fuzzy = ConfusionMatrixDisplay(confusion_matrix=cm_fuzzy)
    disp_fuzzy.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - Fuzzy")
    pname_fuzzy = method + "_fold_" + str(fold_number) + "_Fuzzy_confusion_matrix.png"
    plt.savefig(pname_fuzzy)
    #plt.show()

# Calculate overall performance metrics
end_cv_fuzzy = time.time()
result(all_y_pred_fuzzy, all_y_test_fuzzy, "Fuzzy", time_to_train, time_to_predict_fold_fuzzy)

## **39. Ensemble of Deep Learning Networks (EDLNs) with k-fold Cross-Validation**
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



# Initialize lists to store metrics
all_y_test_EDLN = []
all_y_pred_EDLN = []
start_cv_EDLN = time.time()
time_to_predict_fold_EDLN = 0
time_to_train=0
# Define hyperparameters
num_networks = 5
epochs = 10

# Perform k-fold cross-validation
for fold_number, (train_index, test_index) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold_EDLN, X_test_fold_EDLN = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold_EDLN, y_test_fold_EDLN = y_train.iloc[train_index], y_train.iloc[test_index]

    # Train multiple neural networks
    start = time.time()
    models_fold_EDLN = []
    for i in range(num_networks):
        model_fold_EDLN = create_model(input_shape=X_train_fold_EDLN.shape[1:], num_classes=num_classes)
        model_fold_EDLN.fit(X_train_fold_EDLN, y_train_fold_EDLN, epochs=epochs, verbose=0)
        models_fold_EDLN.append(model_fold_EDLN)
    end = time.time()
    time_to_train += end - start
    # Measure time to predict on the fold
    start_predict_fold_EDLN = time.time()
    # Make predictions on test data using each model
    predictions_fold_EDLN = np.array([model_fold_EDLN.predict(X_test_fold_EDLN) for model_fold_EDLN in models_fold_EDLN])
    end_predict_fold_EDLN = time.time()

    time_to_predict_fold_EDLN += end_predict_fold_EDLN - start_predict_fold_EDLN

    # Aggregate predictions by averaging
    y_pred_fold_EDLN = np.argmax(np.mean(predictions_fold_EDLN, axis=0), axis=1)

    # Append metrics to lists
    all_y_test_EDLN.extend(y_test_fold_EDLN)
    all_y_pred_EDLN.extend(y_pred_fold_EDLN)

    # Generate confusion matrix and display for the fold
    cm_EDLN = confusion_matrix(y_test_fold_EDLN, y_pred_fold_EDLN)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_EDLN = ConfusionMatrixDisplay(confusion_matrix=cm_EDLN)
    disp_EDLN.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - EDLNs")
    pname_EDLN = method + "_fold_" + str(fold_number) + "_EDLNs_confusion_matrix.png"
    plt.savefig(pname_EDLN)
    #plt.show()

# Calculate overall performance metrics
end_cv_EDLN = time.time()
result(all_y_pred_EDLN, all_y_test_EDLN, "EDLNs", time_to_train, time_to_predict_fold_EDLN)

## **40. Gaussian Mixture Model (GMM) with k-fold Cross-Validation**
# Initialize lists to store metrics
all_y_test_gmm = []
all_y_pred_gmm = []
start_cv_gmm = time.time()
time_to_predict_fold_gmm = 0
time_to_train=0
# Perform k-fold cross-validation
for fold_number, (train_index_gmm, test_index_gmm) in enumerate(skf.split(X_train, y_train), 1):
    # Split data into train and test sets for the fold
    X_train_fold_gmm, X_test_fold_gmm = X_train.iloc[train_index_gmm], X_train.iloc[test_index_gmm]
    y_train_fold_gmm, y_test_fold_gmm = y_train.iloc[train_index_gmm], y_train.iloc[test_index_gmm]

    # Number of classes
    n_classes_gmm = len(set(y_train_fold_gmm))

    # Dictionary to store GMMs for each class
    gmm_models_fold_gmm = {}

    # Train GMMs for each class
    start_train_fold_gmm = time.time()
    for i in range(n_classes_gmm):
        # Filter data for the current class
        X_class_gmm = X_train_fold_gmm[y_train_fold_gmm == i]
        # Fit Gaussian Mixture Model
        gmm_fold_gmm = GaussianMixture(n_components=2)  # You can adjust n_components as needed
        gmm_fold_gmm.fit(X_class_gmm)
        # Store the trained GMM
        gmm_models_fold_gmm[i] = gmm_fold_gmm
    end_train_fold_gmm = time.time()
    time_to_train += end_train_fold_gmm - start_train_fold_gmm
    # Measure time to predict on the fold
    start_predict_fold_gmm = time.time()
    y_pred_fold_gmm = []
    for x_gmm in X_test_fold_gmm.values:  # Convert DataFrame to numpy array for iteration
        class_likelihoods_gmm = []
        # Reshape x to have the appropriate dimensions
        x_reshaped_gmm = x_gmm.reshape(1, -1)
        # Calculate likelihood for each class
        for i in range(n_classes_gmm):
            class_likelihood_gmm = gmm_models_fold_gmm[i].score_samples(x_reshaped_gmm)
            class_likelihoods_gmm.append(class_likelihood_gmm)
        # Assign the class with the highest likelihood
        predicted_class_gmm = max(zip(class_likelihoods_gmm, range(n_classes_gmm)))[1]
        y_pred_fold_gmm.append(predicted_class_gmm)
    end_predict_fold_gmm = time.time()

    time_to_predict_fold_gmm += end_predict_fold_gmm - start_predict_fold_gmm

    # Append metrics to lists
    all_y_test_gmm.extend(y_test_fold_gmm)
    all_y_pred_gmm.extend(y_pred_fold_gmm)

    # Generate confusion matrix and display for the fold
    cm_gmm = confusion_matrix(y_test_fold_gmm, y_pred_fold_gmm)
    plt.rcParams['figure.figsize'] = 8, 8
    sns.set_style("white")
    disp_gmm = ConfusionMatrixDisplay(confusion_matrix=cm_gmm)
    disp_gmm.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold_number} Confusion Matrix - GMM")
    pname_gmm = method + "_fold_" + str(fold_number) + "_GMM_confusion_matrix.png"
    plt.savefig(pname_gmm)
    #plt.show()

# Calculate overall performance metrics
end_cv_gmm = time.time()
result(all_y_pred_gmm, all_y_test_gmm, "GMM", time_to_train, time_to_predict_fold_gmm)

## **Closing the file**
file.close()