n_splits_for_cv = 5 #Dont Change 
## **Importing Modules and Libraries**

import datetime

# Get the current date and time
start_time_1 = datetime.datetime.now()


# Get the current time in the desired format
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
print(start_time_1)
print(timestamp)
# Generate the filename with the timestamp
log_filename = f"errors_{timestamp}.txt"
print(log_filename)


import logging

# Configure logging
logging.basicConfig(filename=log_filename, level=logging.ERROR, filemode='a')

import os
# Suppress TensorFlow GPU-related warnings
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

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, multilabel_confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, IsolationForest, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.utils import class_weight
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import RFE
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import sklearn.metrics as metrics

import skfuzzy as fuzz
from pgmpy.estimators import TreeSearch

from hmmlearn import hmm
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Activation, SimpleRNN, LSTM, GRU, Dropout, TimeDistributed, Reshape, Input, Lambda, Add
from keras.optimizers import Adam
from keras.utils import to_categorical
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")
from pgmpy.models import BayesianModel
from pomegranate import *
from pgmpy.models import BayesianModel
from pgmpy.models import JunctionTree
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from catboost import CatBoostClassifier
import tensorflow as tf
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from pgmpy.models import MarkovModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torchvision
from torch import optim
from skopt import BayesSearchCV
from aco import AntColony
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

"""## **Importing Datasets**"""

# dont change this
method = "UNSW_NB_15_FA_SMOTE_IPF"
method = method + "_k_fold_5_Metrics"

train_data = pd.read_csv('UNSW_NB_15_FA_SMOTE_IPF.csv')
test_data = pd.read_csv('UNSW_NB_15_test_FA.csv')

print(train_data.shape)
print(test_data.shape)

# **MULTI-CLASS CLASSIFICATION**
## **Data Splitting**
X_train = train_data.drop(columns=['label'],axis=1)
X_test = test_data.drop(columns=['label'],axis=1)
y_train = train_data['label']
y_test = test_data['label']
X_train = pd.concat([X_train, X_test], axis=0)
y_train = pd.concat([y_train, y_test], axis=0)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

print(X_train.shape)
print(y_train.shape)

fname = method + "_output.csv"
outfile = open(fname, 'w')
outfile.write("algo vs matrices,time to train(sec),time to predict(sec),accuracy_score,precision_score,recall_score,f1_score,fbeta_score,matthews_corrcoef,jaccard_score,cohen_kappa_score,hamming_loss,zero_one_loss,mean_absolute_error,mean_squared_error,mean_squared_error,balanced_accuracy_score,explained_variance_score\n")
def format_decimal(number):
    return f"{number:.{3}f}"
def result(y_pred,y_test,algo,time_to_predict,time_to_train):
    outfile.write(algo+",")
    outfile.write(str(format_decimal(time_to_train))+",")
    outfile.write(str(format_decimal(time_to_predict))+",")
    outfile.write(str(format_decimal(metrics.accuracy_score(y_test,y_pred)))+",")
    outfile.write(str(format_decimal(metrics.precision_score(y_test, y_pred, average='weighted')))+",")
    outfile.write(str(format_decimal(metrics.recall_score(y_test, y_pred, average='weighted')))+",")
    outfile.write(str(format_decimal(metrics.f1_score(y_test, y_pred, average='weighted')))+",")
    outfile.write(str(format_decimal(metrics.fbeta_score(y_test, y_pred,average='weighted', beta=0.5)))+",")
    outfile.write(str(format_decimal(metrics.matthews_corrcoef(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(metrics.jaccard_score(y_test, y_pred, average='weighted')))+",")
    outfile.write(str(format_decimal(metrics.cohen_kappa_score(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(metrics.hamming_loss(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(metrics.zero_one_loss(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(metrics.mean_absolute_error(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(metrics.mean_squared_error(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))+",")
    outfile.write(str(format_decimal(metrics.balanced_accuracy_score(y_test, y_pred)))+",")
    outfile.write(str(format_decimal(metrics.explained_variance_score(y_test, y_pred)*100))+"\n")

#X_train,temp1,y_train,temp2 = train_test_split(X_train,y_train,train_size=0.1,random_state=7)

print(X_train.shape)
print(y_train.shape)
# Initialize Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits= n_splits_for_cv, random_state=47, shuffle=True)

 
 
 
def separator(algo="temp"):
    with open("errors.txt", "a") as file:
        file.write(datetime.now().strftime("%d %b %Y %H:%M"))
        file.write("\n\n*********\n\n")
    outfile.write(algo.strip()+ " " + "erroralgo,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1\n")




print("*"*30)

try:
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
    outfile.close()
    outfile = open(fname, 'a')
    print("Decison Tree Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("Decison Tree")


print("*"*30)

try:
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
    outfile.close()
    outfile = open(fname, 'a')
    print("Linear Regression Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("Linear Regression")

print("*"*30)

try:
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
    outfile.close()
    outfile = open(fname, 'a')
    print("LogisticRegression Completed :) ")



except Exception as e:
    logging.exception(str(e))
    separator("LogisticRegression")


print("*"*30)

try:
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
    outfile.close()
    outfile = open(fname, 'a')
    print("KNN Completed :) ")


except Exception as e:
    logging.exception(str(e))
    separator("KNN")

print("*"*30)

try:
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
    outfile.close()
    outfile = open(fname, 'a')
    print("Random forest Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("Random forest")

print("*"*30)

try:
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


    outfile.close()
    outfile = open(fname, 'a')
    print("MLP Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("MLP")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("Bagging Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("Bagging")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("J48 Completed :) ")


except Exception as e:
    logging.exception(str(e))
    separator("J48")

print("*"*30)

try:
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
        multi_ann.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))
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

    outfile.close()
    outfile = open(fname, 'a')
    print("ANN Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("ANN")

print("*"*30)

try:
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
        multi_dnn.add(Dense(units=10, kernel_initializer='uniform', activation='softmax'))
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

    outfile.close()
    outfile = open(fname, 'a')
    print("DNN Completed :) ")


except Exception as e:
    logging.exception(str(e))
    separator("DNN")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("GradientBoostingClassifier Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("GradientBoostingClassifier")

print("*"*30)

try:
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

    #plt.show()
    outfile.close()
    outfile = open(fname, 'a')
    print("XGBClassifier Completed :) ")
except Exception as e:
    logging.exception(str(e))
    separator("XGBClassifier")

print("*"*30)

try:
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
    #plt.show()
    outfile.close()
    outfile = open(fname, 'a')
    print("Gaussian_Naive_Bayes Completed :) ")

except Exception as e:
    logging.exception(str(e))
    separator("Gaussian_Naive_Bayes")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("Adaptive Gradient Boosting Completed :) ")


except Exception as e:
    logging.exception(str(e))
    separator("Adaptive Gradient Boosting")

print("*"*30)

try:
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
    #plt.show()
    outfile.close()
    outfile = open(fname, 'a')
    print("QDA Completed :) ")


except Exception as e:
    logging.exception(str(e))
    separator("QDA")

print("*"*30)

try:
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
    outfile.close()
    outfile = open(fname, 'a')
    print("SNN Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("snn")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("RBM Completed :)  ")

except Exception as e:
    logging.exception(str(e))
    separator("RBM")

print("*"*30)

try:
    ## **19. LSTM**

    # reloading as many transformations on X,Y causing errors for lstm code


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

    outfile.close()
    outfile = open(fname, 'a')
    print("lstm Completed :)  ")

except Exception as e:
    logging.exception(str(e))
    separator("lstm")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("reconstruction neural networks, Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("reconstruction neural networks")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("DANN Completed :)  ")

except Exception as e:
    logging.exception(str(e))
    separator("DANN")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("DBNs")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("DBMs")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("DEEP AUTO ENCODERS")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("PassiveAggressiveClassifier")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("RidgeClassifier")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("NearestCentroid")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("CSLR")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("CSBC")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("LightGBM")

print("*"*30)

try:
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
    # **Data Splitting**
    # reloading as many transformations on X,Y causing errors for gru code
    X_train = train_data.drop(columns=['label'],axis=1)
    X_test = test_data.drop(columns=['label'],axis=1)
    y_train = train_data['label']
    y_test = test_data['label']
    X_train = pd.concat([X_train, X_test], axis=0)
    y_train = pd.concat([y_train, y_test], axis=0)

    # # 10 DATA COL EACH CLASS
    # # Get unique classes
    X_train,temp1,y_train,temp2 = train_test_split(X_train,y_train,train_size=0.1, random_state=7)

    # Reset indices of X_train and y_train
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

except Exception as e:
    logging.exception(str(e))
    separator("LDA")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("GRU")

print("*"*30)

try:
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


except Exception as e:
    logging.exception(str(e))
    separator("Stochastic_gradient")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("Extra Trees Classifier")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("Feed Forward Neural Networks")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("Fuzzy")

print("*"*30)

try:
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

except Exception as e:
    logging.exception(str(e))
    separator("EDLNs")

print("*"*30)

try:
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

    outfile.close()
    outfile = open(fname, 'a')
    print("GMM Completed :)  ")

except Exception as e:
    logging.exception(str(e))
    separator("GMM")




print("*"*30)

try:
    """## **41. Bernoulli Naive Bayes with k-fold Cross-Validation**"""
    # Initialize lists to store metrics
    all_y_test_bnb = []
    all_y_pred_bnb = []
    start_cv_bnb = time.time()
    time_to_predict_fold_bnb = 0
    time_to_train=0

    # Perform k-fold cross-validation
    for fold_number, (train_index_bnb, test_index_bnb) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_bnb, X_test_fold_bnb = X_train.iloc[train_index_bnb], X_train.iloc[test_index_bnb]
        y_train_fold_bnb, y_test_fold_bnb = y_train.iloc[train_index_bnb], y_train.iloc[test_index_bnb]

        # Create a Bernoulli Naive Bayes classifier
        bnb_fold_bnb = BernoulliNB()

        # Train the classifier
        start_train_fold_bnb = time.time()
        bnb_fold_bnb.fit(X_train_fold_bnb, y_train_fold_bnb)
        end_train_fold_bnb = time.time()
        time_to_train += end_train_fold_bnb - start_train_fold_bnb

        # Predict using the trained model
        start_predict_fold_bnb = time.time()
        y_pred_fold_bnb = bnb_fold_bnb.predict(X_test_fold_bnb)
        end_predict_fold_bnb = time.time()
        time_to_predict_fold_bnb += end_predict_fold_bnb - start_predict_fold_bnb

        # Append metrics to lists
        all_y_test_bnb.extend(y_test_fold_bnb)
        all_y_pred_bnb.extend(y_pred_fold_bnb)

        # Generate confusion matrix and display for the fold
        cm_bnb = confusion_matrix(y_test_fold_bnb, y_pred_fold_bnb)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_bnb = ConfusionMatrixDisplay(confusion_matrix=cm_bnb)
        disp_bnb.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Bernoulli Naive Bayes")
        pname_bnb = method + "_fold_" + str(fold_number) + "_Bernoulli_Naive_Bayes_confusion_matrix.png"
        plt.savefig(pname_bnb)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_bnb = time.time()
    result(all_y_pred_bnb, all_y_test_bnb, "Bernoulli Naive Bayes", time_to_train, time_to_predict_fold_bnb)

    outfile.close()
    outfile = open(fname, 'a')
    print("Bernoulli Naive Bayes with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("Bernoulli Naive Bayes")

print("*"*30)

try:
    """## **42. CatBoost with k-fold Cross-Validation**"""
    # Initialize lists to store metrics
    all_y_test_catboost = []
    all_y_pred_catboost = []
    start_cv_catboost = time.time()
    time_to_predict_fold_catboost = 0
    time_to_train=0

    # Perform k-fold cross-validation
    for fold_number, (train_index_catboost, test_index_catboost) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_catboost, X_test_fold_catboost = X_train.iloc[train_index_catboost], X_train.iloc[test_index_catboost]
        y_train_fold_catboost, y_test_fold_catboost = y_train.iloc[train_index_catboost], y_train.iloc[test_index_catboost]

        # Create and train the CatBoost model
        start_train_fold_catboost = time.time()
        catboost_model_fold_catboost = CatBoostClassifier(random_state=42)
        catboost_model_fold_catboost.fit(X_train_fold_catboost, y_train_fold_catboost)
        end_train_fold_catboost = time.time()
        time_to_train += end_train_fold_catboost - start_train_fold_catboost

        # Predict using the trained model
        start_predict_fold_catboost = time.time()
        y_pred_fold_catboost = catboost_model_fold_catboost.predict(X_test_fold_catboost)
        end_predict_fold_catboost = time.time()
        time_to_predict_fold_catboost += end_predict_fold_catboost - start_predict_fold_catboost

        # Append metrics to lists
        all_y_test_catboost.extend(y_test_fold_catboost)
        all_y_pred_catboost.extend(y_pred_fold_catboost)

        # Generate confusion matrix and display for the fold
        cm_catboost = confusion_matrix(y_test_fold_catboost, y_pred_fold_catboost)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_catboost = ConfusionMatrixDisplay(confusion_matrix=cm_catboost)
        disp_catboost.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - CatBoost")
        pname_catboost = method + "_fold_" + str(fold_number) + "_CatBoost_confusion_matrix.png"
        plt.savefig(pname_catboost)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_catboost = time.time()
    result(all_y_pred_catboost, all_y_test_catboost, "CatBoost", time_to_train, time_to_predict_fold_catboost)

    outfile.close()
    outfile = open(fname, 'a')
    print("CatBoost with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("CatBoost")

print("*"*30)

try:
    """## **43. Centralised blending with k-fold Cross-Validation**"""

    # Initialize lists to store metrics
    all_y_test_blend = []
    all_y_pred_blend = []
    start_cv_blend = time.time()
    time_to_predict_fold_blend = 0
    time_to_train = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_blend, test_index_blend) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_blend, X_test_fold_blend = X_train.iloc[train_index_blend], X_train.iloc[test_index_blend]
        y_train_fold_blend, y_test_fold_blend = y_train.iloc[train_index_blend], y_train.iloc[test_index_blend]

        # Define base models
        base_model1 = DecisionTreeClassifier(random_state=24)
        base_model2 = RandomForestClassifier(random_state=24)
        base_model3 = LogisticRegression(random_state=24)

        # Train base models
        start_train_fold_blend = time.time()
        base_model1.fit(X_train_fold_blend, y_train_fold_blend)
        base_model2.fit(X_train_fold_blend, y_train_fold_blend)
        base_model3.fit(X_train_fold_blend, y_train_fold_blend)
        end_train_fold_blend = time.time()
        time_to_train += end_train_fold_blend - start_train_fold_blend

        # Make predictions on validation data
        preds_val_base_model1 = base_model1.predict(X_test_fold_blend)
        preds_val_base_model2 = base_model2.predict(X_test_fold_blend)
        preds_val_base_model3 = base_model3.predict(X_test_fold_blend)

        # Combine predictions from base models into a feature matrix for meta-model
        X_val_meta_blend = np.column_stack((preds_val_base_model1, preds_val_base_model2, preds_val_base_model3))

        # Train meta-model (blender)
        blender_blend = LogisticRegression(random_state=24)
        blender_blend.fit(X_val_meta_blend, y_test_fold_blend)

        # Make predictions on test data using base models
        preds_test_base_model1 = base_model1.predict(X_test_fold_blend)
        preds_test_base_model2 = base_model2.predict(X_test_fold_blend)
        preds_test_base_model3 = base_model3.predict(X_test_fold_blend)

        # Combine predictions from base models into a feature matrix for meta-model
        X_test_meta_blend = np.column_stack((preds_test_base_model1, preds_test_base_model2, preds_test_base_model3))

        # Make predictions on test data using meta-model
        preds_test_meta_blend = blender_blend.predict(X_test_meta_blend)

        # Append metrics to lists
        all_y_test_blend.extend(y_test_fold_blend)
        all_y_pred_blend.extend(preds_test_meta_blend)

        # Generate confusion matrix and display for the fold
        cm_blend = confusion_matrix(y_test_fold_blend, preds_test_meta_blend)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_blend = ConfusionMatrixDisplay(confusion_matrix=cm_blend)
        disp_blend.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Centralised Blending")
        pname_blend = method + "_fold_" + str(fold_number) + "_Centralised_Blending_confusion_matrix.png"
        plt.savefig(pname_blend)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_blend = time.time()
    result(all_y_pred_blend, all_y_test_blend, "Centralised Blending", time_to_train, time_to_predict_fold_blend)

    """44.## Binary Logical Circular Neural Network (BLoCNet) with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_blocnet = []
    all_y_pred_blocnet = []
    start_cv_blocnet = time.time()
    time_to_predict_fold_blocnet = 0
    time_to_train = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_blocnet, test_index_blocnet) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_blocnet, X_test_fold_blocnet = X_train.iloc[train_index_blocnet], X_train.iloc[test_index_blocnet]
        y_train_fold_blocnet, y_test_fold_blocnet = y_train.iloc[train_index_blocnet], y_train.iloc[test_index_blocnet]

        # Define the architecture of the BLoCNet
        model_blocnet = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_fold_blocnet.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')  # Multi-class classification, so softmax activation
        ])

        # Compile the model
        model_blocnet.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',  # Multi-class classification, so sparse categorical crossentropy loss
                            metrics=['accuracy'])

        # Train the model
        start_train_fold_blocnet = time.time()
        history_blocnet = model_blocnet.fit(X_train_fold_blocnet, y_train_fold_blocnet, epochs=10, batch_size=32, validation_split=0.2)
        end_train_fold_blocnet = time.time()
        time_to_train += end_train_fold_blocnet - start_train_fold_blocnet

        # Evaluate the model
        start_predict_fold_blocnet = time.time()
        y_pred_fold_blocnet = model_blocnet.predict(X_test_fold_blocnet)
        y_pred_fold_blocnet = np.argmax(y_pred_fold_blocnet, axis=1)
        end_predict_fold_blocnet = time.time()
        time_to_predict_fold_blocnet += end_predict_fold_blocnet - start_predict_fold_blocnet

        # Append metrics to lists
        all_y_test_blocnet.extend(y_test_fold_blocnet)
        all_y_pred_blocnet.extend(y_pred_fold_blocnet)

        # Generate confusion matrix and display for the fold
        cm_blocnet = confusion_matrix(y_test_fold_blocnet, y_pred_fold_blocnet)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_blocnet = ConfusionMatrixDisplay(confusion_matrix=cm_blocnet)
        disp_blocnet.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - BLoCNet")
        pname_blocnet = method + "_fold_" + str(fold_number) + "_BLoCNet_confusion_matrix.png"
        plt.savefig(pname_blocnet)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_blocnet = time.time()
    result(all_y_pred_blocnet, all_y_test_blocnet, "BLoCNet", time_to_train, time_to_predict_fold_blocnet)

    outfile.close()
    outfile = open(fname, 'a')
    print("BLoCNet with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("BLoCNet")

print("*"*30)

try:
    """## 45.constructive_learning with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_constructive_learning = []
    all_y_pred_constructive_learning = []
    start_cv_constructive_learning = time.time()
    time_to_predict_fold_constructive_learning = 0
    time_to_train = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_constructive_learning, test_index_constructive_learning) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_constructive_learning, X_test_fold_constructive_learning = X_train.iloc[train_index_constructive_learning], X_train.iloc[test_index_constructive_learning]
        y_train_fold_constructive_learning, y_test_fold_constructive_learning = y_train.iloc[train_index_constructive_learning], y_train.iloc[test_index_constructive_learning]

        # Create a basic Decision Tree model for this fold
        base_model_fold_constructive_learning = DecisionTreeClassifier(random_state=42)

        # Train the base model for this fold
        start_train_fold_constructive_learning = time.time()
        base_model_fold_constructive_learning.fit(X_train_fold_constructive_learning, y_train_fold_constructive_learning)
        end_train_fold_constructive_learning = time.time()
        time_to_train += end_train_fold_constructive_learning - start_train_fold_constructive_learning

        # Evaluate the base model for this fold
        start_predict_fold_constructive_learning = time.time()
        y_pred_fold_constructive_learning = base_model_fold_constructive_learning.predict(X_test_fold_constructive_learning)
        end_predict_fold_constructive_learning = time.time()
        time_to_predict_fold_constructive_learning += end_predict_fold_constructive_learning - start_predict_fold_constructive_learning

        # Append metrics to lists
        all_y_test_constructive_learning.extend(y_test_fold_constructive_learning)
        all_y_pred_constructive_learning.extend(y_pred_fold_constructive_learning)

        # Generate confusion matrix and display for the fold
        cm_constructive_learning = confusion_matrix(y_test_fold_constructive_learning, y_pred_fold_constructive_learning)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_constructive_learning = ConfusionMatrixDisplay(confusion_matrix=cm_constructive_learning)
        disp_constructive_learning.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - constructive_learning")
        pname_constructive_learning = method + "_fold_" + str(fold_number) + "_constructive_learning_confusion_matrix.png"
        plt.savefig(pname_constructive_learning)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_constructive_learning = time.time()
    result(all_y_pred_constructive_learning, all_y_test_constructive_learning, "constructive_learning", time_to_train, time_to_predict_fold_constructive_learning)

    outfile.close()
    outfile = open(fname, 'a')
    print("constructive_learning with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("constructive_learning")

print("*"*30)

try:
    """## 46. Artificial Immune System (AIS) with k-fold Cross-Validation"""
    class AISModel:
        def __init__(self, base_model):
            self.base_model = base_model

        def fit(self, X_train, y_train):
            # AIS training algorithm
            self.base_model.fit(X_train, y_train)

        def predict(self, X_test):
            # AIS prediction algorithm
            y_pred = self.base_model.predict(X_test)
            return y_pred
    # Initialize lists to store metrics
    all_y_test_ais = []
    all_y_pred_ais = []
    start_cv_ais = time.time()
    time_to_predict_fold_ais = 0
    time_to_train = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_ais, test_index_ais) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_ais, X_test_fold_ais = X_train.iloc[train_index_ais], X_train.iloc[test_index_ais]
        y_train_fold_ais, y_test_fold_ais = y_train.iloc[train_index_ais], y_train.iloc[test_index_ais]

        # Create an instance of RandomForestClassifier as the base learner for this fold
        base_model_fold_ais = RandomForestClassifier(random_state=24)

        # Create an instance of AISModel with the base learner for this fold
        ais_model_fold_ais = AISModel(base_model_fold_ais)

        # Train the AIS model for this fold
        start_train_fold_ais = time.time()
        ais_model_fold_ais.fit(X_train_fold_ais, y_train_fold_ais)
        end_train_fold_ais = time.time()
        time_to_train += end_train_fold_ais - start_train_fold_ais

        # Predict using the trained model for this fold
        start_predict_fold_ais = time.time()
        y_pred_fold_ais = ais_model_fold_ais.predict(X_test_fold_ais)
        end_predict_fold_ais = time.time()
        time_to_predict_fold_ais += end_predict_fold_ais - start_predict_fold_ais

        # Append metrics to lists
        all_y_test_ais.extend(y_test_fold_ais)
        all_y_pred_ais.extend(y_pred_fold_ais)

        # Generate confusion matrix and display for the fold
        cm_ais = confusion_matrix(y_test_fold_ais, y_pred_fold_ais)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_ais = ConfusionMatrixDisplay(confusion_matrix=cm_ais)
        disp_ais.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - AIS")
        pname_ais = method + "_fold_" + str(fold_number) + "_AIS_confusion_matrix.png"
        plt.savefig(pname_ais)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_ais = time.time()
    result(all_y_pred_ais, all_y_test_ais, "AIS", time_to_train, time_to_predict_fold_ais)

    outfile.close()
    outfile = open(fname, 'a')
    print("AIS with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("AIS")

print("*"*30)

try:
    """## 48. GBBK Algorithm with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_gbkk = []
    all_y_pred_gbkk = []
    start_cv_gbkk = time.time()
    time_to_predict_fold_gbkk = 0
    time_to_train_gbkk = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_gbkk, test_index_gbkk) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_gbkk, X_test_fold_gbkk = X_train.iloc[train_index_gbkk], X_train.iloc[test_index_gbkk]
        y_train_fold_gbkk, y_test_fold_gbkk = y_train.iloc[train_index_gbkk], y_train.iloc[test_index_gbkk]

        # Initialize Gaussian Naive Bayes (GBBK) model for this fold
        gbbk_model_fold_gbkk = GaussianNB()

        # Train the GBBK model for this fold
        start_train_fold_gbkk = time.time()
        gbbk_model_fold_gbkk.fit(X_train_fold_gbkk, y_train_fold_gbkk)
        end_train_fold_gbkk = time.time()
        time_to_train_gbkk += end_train_fold_gbkk - start_train_fold_gbkk

        # Predict using the trained model for this fold
        start_predict_fold_gbkk = time.time()
        y_pred_fold_gbkk = gbbk_model_fold_gbkk.predict(X_test_fold_gbkk)
        end_predict_fold_gbkk = time.time()
        time_to_predict_fold_gbkk += end_predict_fold_gbkk - start_predict_fold_gbkk

        # Append metrics to lists
        all_y_test_gbkk.extend(y_test_fold_gbkk)
        all_y_pred_gbkk.extend(y_pred_fold_gbkk)

        # Generate confusion matrix and display for the fold
        cm_gbkk = confusion_matrix(y_test_fold_gbkk, y_pred_fold_gbkk)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_gbkk = ConfusionMatrixDisplay(confusion_matrix=cm_gbkk)
        disp_gbkk.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - GBBK")
        pname_gbkk = method + "_fold_" + str(fold_number) + "_GBBK_confusion_matrix.png"
        plt.savefig(pname_gbkk)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_gbkk = time.time()
    result(all_y_pred_gbkk, all_y_test_gbkk, "GBBK", time_to_train_gbkk, time_to_predict_fold_gbkk)

    outfile.close()
    outfile = open(fname, 'a')
    print("GBBK with k-fold Cross-Validation Completed :)  ")




except Exception as e:
    logging.exception(str(e))
    separator("GBBK")

print("*"*30)

try:
    """## 48. GE SVM Algorithm with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_gbkk = []
    all_y_pred_gbkk = []
    start_cv_gbkk = time.time()
    time_to_predict_fold_gbkk = 0
    time_to_train_gbkk = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_gbkk, test_index_gbkk) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_gbkk, X_test_fold_gbkk = X_train.iloc[train_index_gbkk], X_train.iloc[test_index_gbkk]
        y_train_fold_gbkk, y_test_fold_gbkk = y_train.iloc[train_index_gbkk], y_train.iloc[test_index_gbkk]

        # Initialize Gaussian Naive Bayes (GBBK) model for this fold
        gbbk_model_fold_gbkk = SVC(kernel='rbf', decision_function_shape='ovo')

        # Train the GBBK model for this fold
        start_train_fold_gbkk = time.time()
        gbbk_model_fold_gbkk.fit(X_train_fold_gbkk, y_train_fold_gbkk)
        end_train_fold_gbkk = time.time()
        time_to_train_gbkk += end_train_fold_gbkk - start_train_fold_gbkk

        # Predict using the trained model for this fold
        start_predict_fold_gbkk = time.time()
        y_pred_fold_gbkk = gbbk_model_fold_gbkk.predict(X_test_fold_gbkk)
        end_predict_fold_gbkk = time.time()
        time_to_predict_fold_gbkk += end_predict_fold_gbkk - start_predict_fold_gbkk

        # Append metrics to lists
        all_y_test_gbkk.extend(y_test_fold_gbkk)
        all_y_pred_gbkk.extend(y_pred_fold_gbkk)

        # Generate confusion matrix and display for the fold
        cm_gbkk = confusion_matrix(y_test_fold_gbkk, y_pred_fold_gbkk)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_gbkk = ConfusionMatrixDisplay(confusion_matrix=cm_gbkk)
        disp_gbkk.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - GE SVM")
        pname_gbkk = method + "_fold_" + str(fold_number) + "_GE_SVM_confusion_matrix.png"
        plt.savefig(pname_gbkk)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_gbkk = time.time()
    result(all_y_pred_gbkk, all_y_test_gbkk, "GE SVM", time_to_train_gbkk, time_to_predict_fold_gbkk)

    outfile.close()
    outfile = open(fname, 'a')
    print("GE SVM with k-fold Cross-Validation Completed :)  ")




except Exception as e:
    logging.exception(str(e))
    separator("GE SVM")

print("*"*30)

try:
    """## 51. Hidden Naive Bayes (HNB) with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_hnb = []
    all_y_pred_hnb = []
    start_cv_hnb = time.time()
    time_to_predict_fold_hnb = 0
    time_to_train_hnb = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_hnb, test_index_hnb) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_hnb, X_test_fold_hnb = X_train.iloc[train_index_hnb], X_train.iloc[test_index_hnb]
        y_train_fold_hnb, y_test_fold_hnb = y_train.iloc[train_index_hnb], y_train.iloc[test_index_hnb]

        # Train the HNB model for this fold
        hnb_model_fold = GaussianNB()
        start_train_fold_hnb = time.time()
        hnb_model_fold.fit(pd.DataFrame(X_train_fold_hnb), pd.DataFrame(y_train_fold_hnb))
        end_train_fold_hnb = time.time()
        time_to_train_hnb += end_train_fold_hnb - start_train_fold_hnb

        # Predict using the trained model for this fold
        start_predict_fold_hnb = time.time()
        y_pred_fold_hnb = hnb_model_fold.predict(X_test_fold_hnb.values)
        end_predict_fold_hnb = time.time()
        time_to_predict_fold_hnb += end_predict_fold_hnb - start_predict_fold_hnb

        # Append metrics to lists
        all_y_test_hnb.extend(y_test_fold_hnb)
        all_y_pred_hnb.extend(y_pred_fold_hnb)

        # Generate confusion matrix and display for the fold
        cm_hnb = confusion_matrix(y_test_fold_hnb, y_pred_fold_hnb)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_hnb = ConfusionMatrixDisplay(confusion_matrix=cm_hnb)
        disp_hnb.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - HNB")
        pname_hnb = method + "_fold_" + str(fold_number) + "_HNB_confusion_matrix.png"
        plt.savefig(pname_hnb)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_hnb = time.time()
    result(all_y_pred_hnb, all_y_test_hnb, "Hidden Naive Bayes (HNB)", time_to_train_hnb, time_to_predict_fold_hnb)

    outfile.close()
    outfile = open(fname, 'a')
    print("HNB with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("Hidden Naive Bayes")

print("*"*30)

try:
    """## 52. HistGradientBoostingClassifier with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_hgb = []
    all_y_pred_hgb = []
    start_cv_hgb = time.time()
    time_to_predict_fold_hgb = 0
    time_to_train_hgb = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_hgb, test_index_hgb) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_hgb, X_test_fold_hgb = X_train.iloc[train_index_hgb], X_train.iloc[test_index_hgb]
        y_train_fold_hgb, y_test_fold_hgb = y_train.iloc[train_index_hgb], y_train.iloc[test_index_hgb]

        # Train the HistGradientBoostingClassifier model for this fold
        hgb_model_fold = HistGradientBoostingClassifier(random_state=24)
        start_train_fold_hgb = time.time()
        hgb_model_fold.fit(X_train_fold_hgb, y_train_fold_hgb)
        end_train_fold_hgb = time.time()
        time_to_train_hgb += end_train_fold_hgb - start_train_fold_hgb

        # Predict using the trained model for this fold
        start_predict_fold_hgb = time.time()
        y_pred_fold_hgb = hgb_model_fold.predict(X_test_fold_hgb)
        end_predict_fold_hgb = time.time()
        time_to_predict_fold_hgb += end_predict_fold_hgb - start_predict_fold_hgb

        # Append metrics to lists
        all_y_test_hgb.extend(y_test_fold_hgb)
        all_y_pred_hgb.extend(y_pred_fold_hgb)

        # Generate confusion matrix and display for the fold
        cm_hgb = confusion_matrix(y_test_fold_hgb, y_pred_fold_hgb)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_hgb = ConfusionMatrixDisplay(confusion_matrix=cm_hgb)
        disp_hgb.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - HistGradientBoostingClassifier")
        pname_hgb = method + "_fold_" + str(fold_number) + "_HistGradientBoostingClassifier_confusion_matrix.png"
        plt.savefig(pname_hgb)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_hgb = time.time()
    result(all_y_pred_hgb, all_y_test_hgb, "HistGradientBoostingClassifier", time_to_train_hgb, time_to_predict_fold_hgb)

    outfile.close()
    outfile = open(fname, 'a')
    print("HistGradientBoostingClassifier with k-fold Cross-Validation Completed :)  ")





except Exception as e:
    logging.exception(str(e))
    separator("HistGradientBoostingClassifier")

print("*"*30)

try:
    """## IGRF-RFE with k-fold Cross-Validation"""

    # Initialize Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=24)

    # Initialize RFE (Recursive Feature Elimination) with Decision Tree Classifier as estimator
    rfe = RFE(estimator=dt_classifier)

    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits= n_splits_for_cv, shuffle=True, random_state=42)

    # Initialize lists to store metrics
    all_y_test_igrf_rfe = []
    all_y_pred_igrf_rfe = []
    start_cv_igrf_rfe = time.time()
    time_to_predict_fold_igrf_rfe = 0
    time_to_train_igrf_rfe = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_igrf_rfe, test_index_igrf_rfe) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_igrf_rfe, X_test_fold_igrf_rfe = X_train.iloc[train_index_igrf_rfe], X_train.iloc[test_index_igrf_rfe]
        y_train_fold_igrf_rfe, y_test_fold_igrf_rfe = y_train.iloc[train_index_igrf_rfe], y_train.iloc[test_index_igrf_rfe]

        # Fit RFE on training data for this fold
        start_feature_selection_fold_igrf_rfe = time.time()
        rfe.fit(X_train_fold_igrf_rfe, y_train_fold_igrf_rfe)
        end_feature_selection_fold_igrf_rfe = time.time()

        # Select features based on RFE ranking
        X_train_rfe_fold_igrf_rfe = rfe.transform(X_train_fold_igrf_rfe)
        X_test_rfe_fold_igrf_rfe = rfe.transform(X_test_fold_igrf_rfe)

        # Train Decision Tree Classifier using selected features for this fold
        start_train_fold_igrf_rfe = time.time()
        dt_classifier.fit(X_train_rfe_fold_igrf_rfe, y_train_fold_igrf_rfe)
        end_train_fold_igrf_rfe = time.time()
        time_to_train_igrf_rfe += end_train_fold_igrf_rfe - start_train_fold_igrf_rfe

        # Predict using the trained model for this fold
        start_predict_fold_igrf_rfe = time.time()
        y_pred_fold_igrf_rfe = dt_classifier.predict(X_test_rfe_fold_igrf_rfe)
        end_predict_fold_igrf_rfe = time.time()
        time_to_predict_fold_igrf_rfe += end_predict_fold_igrf_rfe - start_predict_fold_igrf_rfe

        # Append metrics to lists
        all_y_test_igrf_rfe.extend(y_test_fold_igrf_rfe)
        all_y_pred_igrf_rfe.extend(y_pred_fold_igrf_rfe)

        # Generate confusion matrix and display for the fold
        cm_igrf_rfe = confusion_matrix(y_test_fold_igrf_rfe, y_pred_fold_igrf_rfe)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_igrf_rfe = ConfusionMatrixDisplay(confusion_matrix=cm_igrf_rfe)
        disp_igrf_rfe.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - IGRF-RFE")
        pname_igrf_rfe = method + "_fold_" + str(fold_number) + "_IGRF_RFE_confusion_matrix.png"
        plt.savefig(pname_igrf_rfe)

    # Calculate overall performance metrics
    end_cv_igrf_rfe = time.time()
    result(all_y_pred_igrf_rfe, all_y_test_igrf_rfe, "IGRF-RFE with k-fold Cross-Validation", time_to_train_igrf_rfe, time_to_predict_fold_igrf_rfe)

    outfile.close()
    outfile = open(fname, 'a')
    print("IGRF-RFE with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("IGRF-RFE")

print("*"*30)

try:
    """## 55. Independent Component Analysis (ICA) with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_ica = []
    all_y_pred_ica = []
    start_cv_ica = time.time()
    time_to_predict_fold_ica = 0
    time_to_train_ica = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_ica, test_index_ica) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_ica, X_test_fold_ica = X_train.iloc[train_index_ica], X_train.iloc[test_index_ica]
        y_train_fold_ica, y_test_fold_ica = y_train.iloc[train_index_ica], y_train.iloc[test_index_ica]

        # Perform Independent Component Analysis (ICA) for dimensionality reduction
        ica_fold = FastICA(n_components=10, random_state=42)
        X_train_ica_fold = ica_fold.fit_transform(X_train_fold_ica)
        X_test_ica_fold = ica_fold.transform(X_test_fold_ica)

        # Normalize the data
        scaler_fold = StandardScaler()
        X_train_ica_fold = scaler_fold.fit_transform(X_train_ica_fold)
        X_test_ica_fold = scaler_fold.transform(X_test_ica_fold)

        # Train a RandomForestClassifier on the transformed data for this fold
        rf_model_fold_ica = RandomForestClassifier(random_state=24)
        start_train_fold_ica = time.time()
        rf_model_fold_ica.fit(X_train_ica_fold, y_train_fold_ica)
        end_train_fold_ica = time.time()
        time_to_train_ica += end_train_fold_ica - start_train_fold_ica

        # Predict using the trained model for this fold
        start_predict_fold_ica = time.time()
        y_pred_fold_ica = rf_model_fold_ica.predict(X_test_ica_fold)
        end_predict_fold_ica = time.time()
        time_to_predict_fold_ica += end_predict_fold_ica - start_predict_fold_ica

        # Append metrics to lists
        all_y_test_ica.extend(y_test_fold_ica)
        all_y_pred_ica.extend(y_pred_fold_ica)

        # Generate confusion matrix and display for the fold
        cm_ica = confusion_matrix(y_test_fold_ica, y_pred_fold_ica)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_ica = ConfusionMatrixDisplay(confusion_matrix=cm_ica)
        disp_ica.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - ICA")
        pname_ica = method + "_fold_" + str(fold_number) + "_ICA_confusion_matrix.png"
        plt.savefig(pname_ica)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_ica = time.time()
    result(all_y_pred_ica, all_y_test_ica, "ICA", time_to_train_ica, time_to_predict_fold_ica)

    outfile.close()
    outfile = open(fname, 'a')
    print("ICA with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("ICA")

print("*"*30)

try:
    """## 56. Lasso Regression with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_lasso = []
    all_y_pred_lasso = []
    start_cv_lasso = time.time()
    time_to_predict_fold_lasso = 0
    time_to_train_lasso = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_lasso, test_index_lasso) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_lasso, X_test_fold_lasso = X_train.iloc[train_index_lasso], X_train.iloc[test_index_lasso]
        y_train_fold_lasso, y_test_fold_lasso = y_train.iloc[train_index_lasso], y_train.iloc[test_index_lasso]

        # Initialize Lasso Regression model
        lasso_model_fold = LogisticRegression(penalty='l1', solver='saga', random_state=24, max_iter=1000)

        # Train the Lasso Regression model for this fold
        start_train_fold_lasso = time.time()
        lasso_model_fold.fit(X_train_fold_lasso, y_train_fold_lasso)
        end_train_fold_lasso = time.time()
        time_to_train_lasso += end_train_fold_lasso - start_train_fold_lasso

        # Predict using the trained model for this fold
        start_predict_fold_lasso = time.time()
        y_pred_fold_lasso = lasso_model_fold.predict(X_test_fold_lasso)
        end_predict_fold_lasso = time.time()
        time_to_predict_fold_lasso += end_predict_fold_lasso - start_predict_fold_lasso

        # Append metrics to lists
        all_y_test_lasso.extend(y_test_fold_lasso)
        all_y_pred_lasso.extend(y_pred_fold_lasso)

        # Generate confusion matrix and display for the fold
        cm_lasso = confusion_matrix(y_test_fold_lasso, y_pred_fold_lasso)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_lasso = ConfusionMatrixDisplay(confusion_matrix=cm_lasso)
        disp_lasso.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Lasso Regression")
        pname_lasso = method + "_fold_" + str(fold_number) + "_Lasso_Regression_confusion_matrix.png"
        plt.savefig(pname_lasso)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_lasso = time.time()
    result(all_y_pred_lasso, all_y_test_lasso, "Lasso Regression", time_to_train_lasso, time_to_predict_fold_lasso)

    outfile.close()
    outfile = open(fname, 'a')
    print("Lasso Regression with k-fold Cross-Validation Completed :)  ")

except Exception as e:
    logging.exception(str(e))
    separator("Lasso")

print("*"*30)

try:
    """## 57. Meta (KNN) with Neighbors (K) and k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_meta_knn = []
    all_y_pred_meta_knn = []
    start_cv_meta_knn = time.time()
    time_to_predict_fold_meta_knn = 0
    time_to_train_meta_knn = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_meta_knn, test_index_meta_knn) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_meta_knn, X_test_fold_meta_knn = X_train.iloc[train_index_meta_knn], X_train.iloc[test_index_meta_knn]
        y_train_fold_meta_knn, y_test_fold_meta_knn = y_train.iloc[train_index_meta_knn], y_train.iloc[test_index_meta_knn]

        # Initialize base classifier (KNN) for meta-learning
        k = 5
        base_classifier_meta_knn = KNeighborsClassifier(n_neighbors=k)

        # Initialize BaggingClassifier for meta-learning with KNN base classifier
        meta_knn_model_fold = BaggingClassifier(base_classifier_meta_knn, n_estimators=10, random_state=42)

        # Train the meta KNN model for this fold
        start_train_fold_meta_knn = time.time()
        meta_knn_model_fold.fit(X_train_fold_meta_knn, y_train_fold_meta_knn)
        end_train_fold_meta_knn = time.time()
        time_to_train_meta_knn += end_train_fold_meta_knn - start_train_fold_meta_knn

        # Predict using the trained model for this fold
        start_predict_fold_meta_knn = time.time()
        y_pred_fold_meta_knn = meta_knn_model_fold.predict(X_test_fold_meta_knn)
        end_predict_fold_meta_knn = time.time()
        time_to_predict_fold_meta_knn += end_predict_fold_meta_knn - start_predict_fold_meta_knn

        # Append metrics to lists
        all_y_test_meta_knn.extend(y_test_fold_meta_knn)
        all_y_pred_meta_knn.extend(y_pred_fold_meta_knn)

        # Generate confusion matrix and display for the fold
        cm_meta_knn = confusion_matrix(y_test_fold_meta_knn, y_pred_fold_meta_knn)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_meta_knn = ConfusionMatrixDisplay(confusion_matrix=cm_meta_knn)
        disp_meta_knn.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Meta (KNN)")
        pname_meta_knn = method + "_fold_" + str(fold_number) + "_Meta_KNN_confusion_matrix.png"
        plt.savefig(pname_meta_knn)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_meta_knn = time.time()
    result(all_y_pred_meta_knn, all_y_test_meta_knn, "Meta (KNN)", time_to_train_meta_knn, time_to_predict_fold_meta_knn)

    outfile.close()
    outfile = open(fname, 'a')
    print("Meta (KNN) with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("KNN")

print("*"*30)

try:
    """## T-SNE Random Forest (T-SNERF) with k-fold Cross-Validation"""

    # Initialize T-SNE
    tsne = TSNE(n_components=2, random_state=42)

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=24)

    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits= n_splits_for_cv, shuffle=True, random_state=42)

    # Initialize lists to store metrics
    all_y_test_tsnerf = []
    all_y_pred_tsnerf = []
    start_cv_tsnerf = time.time()
    time_to_predict_fold_tsnerf = 0
    time_to_train_tsnerf = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_tsnerf, test_index_tsnerf) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_tsnerf, X_test_fold_tsnerf = X_train.iloc[train_index_tsnerf], X_train.iloc[test_index_tsnerf]
        y_train_fold_tsnerf, y_test_fold_tsnerf = y_train.iloc[train_index_tsnerf], y_train.iloc[test_index_tsnerf]

        # Train T-SNE on the training data for this fold
        start_train_fold_tsnerf = time.time()
        X_train_tsne_fold_tsnerf = tsne.fit_transform(X_train_fold_tsnerf)
        end_train_fold_tsnerf = time.time()
        time_to_train_tsnerf += end_train_fold_tsnerf - start_train_fold_tsnerf

        # Train Random Forest on the T-SNE transformed data for this fold
        start_rf_fold_tsnerf = time.time()
        rf_model.fit(X_train_tsne_fold_tsnerf, y_train_fold_tsnerf)
        end_rf_fold_tsnerf = time.time()

        # Transform the test data using the trained T-SNE model for this fold
        start_transform_fold_tsnerf = time.time()
        X_test_tsne_fold_tsnerf = tsne.fit_transform(X_test_fold_tsnerf)
        end_transform_fold_tsnerf = time.time()

        # Predict using the trained Random Forest model for this fold
        start_predict_fold_tsnerf = time.time()
        y_pred_fold_tsnerf = rf_model.predict(X_test_tsne_fold_tsnerf)
        end_predict_fold_tsnerf = time.time()
        time_to_predict_fold_tsnerf += end_predict_fold_tsnerf - start_predict_fold_tsnerf

        # Append metrics to lists
        all_y_test_tsnerf.extend(y_test_fold_tsnerf)
        all_y_pred_tsnerf.extend(y_pred_fold_tsnerf)

        # Generate confusion matrix and display for the fold
        cm_tsnerf = confusion_matrix(y_test_fold_tsnerf, y_pred_fold_tsnerf)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_tsnerf = ConfusionMatrixDisplay(confusion_matrix=cm_tsnerf)
        disp_tsnerf.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - T-SNE Random Forest (T-SNERF)")
        pname_tsnerf = method + "_fold_" + str(fold_number) + "_TSNE_confusion_matrix.png"
        plt.savefig(pname_tsnerf)

    # Calculate overall performance metrics
    end_cv_tsnerf = time.time()
    result(all_y_pred_tsnerf, all_y_test_tsnerf, "T-SNE Random Forest (T-SNERF) with k-fold Cross-Validation", time_to_train_tsnerf, time_to_predict_fold_tsnerf)

    outfile.close()
    outfile = open(fname, 'a')
    print("T-SNE Random Forest (T-SNERF) with k-fold Cross-Validation Completed :)  ")




except Exception as e:
    logging.exception(str(e))
    separator("T-SNE Random Forest")

print("*"*30)

try:
    """## 60. Projected Gradient Descent (PGD) with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_pgd = []
    all_y_pred_pgd = []
    start_cv_pgd = time.time()
    time_to_predict_fold_pgd = 0
    time_to_train_pgd = 0

    # Define a simple neural network model
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 128)
            self.fc2 = nn.Linear(128, len(np.unique(y_train)))

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Perform k-fold cross-validation
    for fold_number, (train_index_pgd, test_index_pgd) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_pgd, X_test_fold_pgd = X_train.iloc[train_index_pgd], X_train.iloc[test_index_pgd]
        y_train_fold_pgd, y_test_fold_pgd = y_train.iloc[train_index_pgd], y_train.iloc[test_index_pgd]

        # Convert pandas DataFrame to PyTorch tensors for this fold
        X_train_tensor_pgd = torch.tensor(X_train_fold_pgd.values, dtype=torch.float32)
        y_train_tensor_pgd = torch.tensor(y_train_fold_pgd.values, dtype=torch.long)
        X_test_tensor_pgd = torch.tensor(X_test_fold_pgd.values, dtype=torch.float32)

        # Create a DataLoader for training for this fold
        train_dataset_pgd = TensorDataset(X_train_tensor_pgd, y_train_tensor_pgd)
        train_loader_pgd = DataLoader(train_dataset_pgd, batch_size=64, shuffle=True)

        # Define model, loss function, and optimizer for this fold
        model_pgd = SimpleNN()
        criterion_pgd = nn.CrossEntropyLoss()
        optimizer_pgd = optim.Adam(model_pgd.parameters(), lr=0.001)

        # Train the model for this fold
        start_train_fold_pgd = time.time()
        for epoch in range(10):
            for inputs, labels in train_loader_pgd:
                optimizer_pgd.zero_grad()
                outputs = model_pgd(inputs)
                loss = criterion_pgd(outputs, labels)
                loss.backward()
                optimizer_pgd.step()
        end_train_fold_pgd = time.time()
        time_to_train_pgd += end_train_fold_pgd - start_train_fold_pgd

        # Evaluate the model for this fold
        model_pgd.eval()
        start_predict_fold_pgd = time.time()
        with torch.no_grad():
            y_pred_fold_pgd = model_pgd(X_test_tensor_pgd).argmax(dim=1).detach().numpy()
        end_predict_fold_pgd = time.time()
        time_to_predict_fold_pgd += end_predict_fold_pgd - start_predict_fold_pgd

        # Append metrics to lists
        all_y_test_pgd.extend(y_test_fold_pgd)
        all_y_pred_pgd.extend(y_pred_fold_pgd)

        # Generate confusion matrix and display for the fold
        cm_pgd = confusion_matrix(y_test_fold_pgd, y_pred_fold_pgd)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_pgd = ConfusionMatrixDisplay(confusion_matrix=cm_pgd)
        disp_pgd.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Projected Gradient Descent (PGD)")
        pname_pgd = method + "_fold_" + str(fold_number) + "_PGD_confusion_matrix.png"
        plt.savefig(pname_pgd)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_pgd = time.time()
    result(all_y_pred_pgd, all_y_test_pgd, "Projected Gradient Descent (PGD)", time_to_train_pgd, time_to_predict_fold_pgd)

    outfile.close()
    outfile = open(fname, 'a')
    print("Projected Gradient Descent (PGD) with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("Projected Gradient Descent")

print("*"*30)

try:
    """## 61. Principal Component Analysis (PCA) with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_pca = []
    all_y_pred_pca = []
    start_cv_pca = time.time()
    time_to_predict_fold_pca = 0
    time_to_train_pca = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_pca, test_index_pca) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_pca, X_test_fold_pca = X_train.iloc[train_index_pca], X_train.iloc[test_index_pca]
        y_train_fold_pca, y_test_fold_pca = y_train.iloc[train_index_pca], y_train.iloc[test_index_pca]

        # Convert y_test_fold_pca to integer labels if it's not already
        label_encoder = LabelEncoder()
        y_test_encoded_fold_pca = label_encoder.fit_transform(y_test_fold_pca)

        # Apply PCA to reduce dimensionality for this fold
        pca_fold_pca = PCA(n_components=0.95, random_state=42)  # Retain 95% of variance
        X_train_fold_pca = pca_fold_pca.fit_transform(X_train_fold_pca)
        X_test_fold_pca = pca_fold_pca.transform(X_test_fold_pca)

        # Train the classifier on the reduced dimensionality data for this fold
        dt_multi_fold_pca = DecisionTreeClassifier(random_state=24)
        start_train_fold_pca = time.time()
        dt_multi_fold_pca.fit(X_train_fold_pca, y_train_fold_pca)
        end_train_fold_pca = time.time()
        time_to_train_pca += end_train_fold_pca - start_train_fold_pca

        # Predict using the trained model for this fold
        start_predict_fold_pca = time.time()
        y_pred_fold_pca = dt_multi_fold_pca.predict(X_test_fold_pca)
        end_predict_fold_pca = time.time()
        time_to_predict_fold_pca += end_predict_fold_pca - start_predict_fold_pca

        # Append metrics to lists
        all_y_test_pca.extend(y_test_encoded_fold_pca)
        all_y_pred_pca.extend(y_pred_fold_pca)

        # Generate confusion matrix and display for the fold
        cm_pca = confusion_matrix(y_test_encoded_fold_pca, y_pred_fold_pca)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca)
        disp_pca.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - PCA")
        pname_pca = method + "_fold_" + str(fold_number) + "_PCA_confusion_matrix.png"
        plt.savefig(pname_pca)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_pca = time.time()
    result(all_y_pred_pca, all_y_test_pca, "PCA", time_to_train_pca, time_to_predict_fold_pca)

    outfile.close()
    outfile = open(fname, 'a')
    print("PCA with k-fold Cross-Validation Completed :)  ")



except Exception as e:
    logging.exception(str(e))
    separator("PCA")

print("*"*30)

try:
    """## 48. RBF SVM Algorithm with k-fold Cross-Validation"""

    # Initialize lists to store metrics
    all_y_test_gbkk = []
    all_y_pred_gbkk = []
    start_cv_gbkk = time.time()
    time_to_predict_fold_gbkk = 0
    time_to_train_gbkk = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_gbkk, test_index_gbkk) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_gbkk, X_test_fold_gbkk = X_train.iloc[train_index_gbkk], X_train.iloc[test_index_gbkk]
        y_train_fold_gbkk, y_test_fold_gbkk = y_train.iloc[train_index_gbkk], y_train.iloc[test_index_gbkk]

        # Initialize Gaussian Naive Bayes (GBBK) model for this fold
        gbbk_model_fold_gbkk = SVC(kernel='rbf', random_state=24)

        # Train the GBBK model for this fold
        start_train_fold_gbkk = time.time()
        gbbk_model_fold_gbkk.fit(X_train_fold_gbkk, y_train_fold_gbkk)
        end_train_fold_gbkk = time.time()
        time_to_train_gbkk += end_train_fold_gbkk - start_train_fold_gbkk

        # Predict using the trained model for this fold
        start_predict_fold_gbkk = time.time()
        y_pred_fold_gbkk = gbbk_model_fold_gbkk.predict(X_test_fold_gbkk)
        end_predict_fold_gbkk = time.time()
        time_to_predict_fold_gbkk += end_predict_fold_gbkk - start_predict_fold_gbkk

        # Append metrics to lists
        all_y_test_gbkk.extend(y_test_fold_gbkk)
        all_y_pred_gbkk.extend(y_pred_fold_gbkk)

        # Generate confusion matrix and display for the fold
        cm_gbkk = confusion_matrix(y_test_fold_gbkk, y_pred_fold_gbkk)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_gbkk = ConfusionMatrixDisplay(confusion_matrix=cm_gbkk)
        disp_gbkk.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - RBF SVM")
        pname_gbkk = method + "_fold_" + str(fold_number) + "_RBF_confusion_matrix.png"
        plt.savefig(pname_gbkk)
        #plt.show()

    # Calculate overall performance metrics
    end_cv_gbkk = time.time()
    result(all_y_pred_gbkk, all_y_test_gbkk, "RBF", time_to_train_gbkk, time_to_predict_fold_gbkk)

    outfile.close()
    outfile = open(fname, 'a')
    print("RBF  with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("RBF")

print("*"*30)

try:
    """## Stacked Convolutional Neural Networks with k-fold Cross-Validation"""

    # Convert DataFrame to NumPy array
    X = X_train.to_numpy()
    y = y_train.to_numpy()

    # Reshape the input data for Conv1D layer
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Define the CNN model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Adjust the number of units based on the number of classes in your dataset
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits= n_splits_for_cv, shuffle=True, random_state=42)

    # Initialize lists to store metrics
    all_y_test_cnn = []
    all_y_pred_cnn = []
    start_cv_cnn = time.time()
    time_to_predict_fold_cnn = 0
    time_to_train_cnn = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_cnn, test_index_cnn) in enumerate(skf.split(X, y), 1):
        # Split data into train and test sets for the fold
        X_train_fold_cnn, X_test_fold_cnn = X[train_index_cnn], X[test_index_cnn]
        y_train_fold_cnn, y_test_fold_cnn = y[train_index_cnn], y[test_index_cnn]

        # Train the model for this fold
        start_train_fold_cnn = time.time()
        history = model.fit(X_train_fold_cnn, y_train_fold_cnn, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
        end_train_fold_cnn = time.time()
        time_to_train_cnn += end_train_fold_cnn - start_train_fold_cnn

        # Evaluate the model for this fold
        start_predict_fold_cnn = time.time()
        y_pred_fold_cnn = model.predict(X_test_fold_cnn)
        y_pred_fold_cnn = np.argmax(y_pred_fold_cnn, axis=1)
        end_predict_fold_cnn = time.time()
        time_to_predict_fold_cnn += end_predict_fold_cnn - start_predict_fold_cnn

        # Append metrics to lists
        all_y_test_cnn.extend(y_test_fold_cnn)
        all_y_pred_cnn.extend(y_pred_fold_cnn)

        # Generate confusion matrix and display for the fold
        cm_cnn = confusion_matrix(y_test_fold_cnn, y_pred_fold_cnn)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn)
        disp_cnn.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Stacked Convolutional Neural Networks")
        pname_cnn = method + "_fold_" + str(fold_number) + "_Stacked_Convolutional_Neural_Networks_confusion_matrix.png"
        plt.savefig(pname_cnn)

    # Calculate overall performance metrics
    end_cv_cnn = time.time()
    result(all_y_pred_cnn, all_y_test_cnn, "Stacked Convolutional Neural Networks with k-fold Cross-Validation", time_to_train_cnn, time_to_predict_fold_cnn)

    outfile.close()
    outfile = open(fname, 'a')
    print("Stacked Convolutional Neural Networks with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("Stacked Convolutional Neural Networks")

print("*"*30)

try:
    """## 64. Simulated Annealing (SA) with k-fold Cross-Validation"""

    # Define the objective function for optimization

    # Convert X_train and y_train to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Convert X_test to NumPy array if it's not already
    X_test = np.array(X_test)

    # Reshape the input data if needed
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
    if X_test.ndim > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    def objective_function(params):
        max_depth, min_samples_split, min_samples_leaf = params

        # Create and train the Decision Tree model with given hyperparameters
        dt_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=24)
        dt_model.fit(X_train, y_train)

        # Predict using the trained model
        y_pred = dt_model.predict(X_test)

        # Evaluate the model - Example: using accuracy
        accuracy = np.mean(y_pred == y_test)

        # Return the negative of the accuracy (since we want to minimize)
        return -accuracy
    # Simulated Annealing hyperparameter optimization
    def simulated_annealing(objective_function, space, n_calls=50, initial_temperature=100.0, cooling_rate=0.95):
        best_params = None
        best_score = float('-inf')
        temperature = initial_temperature

        current_params = [np.random.randint(low, high + 1) for low, high in space]

        for _ in range(n_calls):
            next_params = [np.random.randint(low, high + 1) for low, high in space]

            current_score = objective_function(current_params)
            next_score = objective_function(next_params)

            if next_score > current_score or np.random.rand() < np.exp((next_score - current_score) / temperature):
                current_params = next_params
                current_score = next_score

            if current_score > best_score:
                best_params = current_params
                best_score = current_score

            temperature *= cooling_rate

        return best_params
    # Define the search space for hyperparameters
    space = [(1, 20),  # max_depth
            (2, 20),  # min_samples_split
            (1, 20)]  # min_samples_leaf

    # Initialize lists to store metrics
    all_y_test_sa = []
    all_y_pred_sa = []
    start_cv_sa = time.time()
    time_to_predict_fold_sa = 0
    time_to_train_sa = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_sa, test_index_sa) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_sa, X_test_fold_sa = X_train[train_index_sa], X_train[test_index_sa]
        y_train_fold_sa, y_test_fold_sa = y_train[train_index_sa], y_train[test_index_sa]

        # Run the optimization using Simulated Annealing for this fold
        start_train_fold_sa = time.time()
        best_params = simulated_annealing(objective_function, space)
        end_train_fold_sa = time.time()

        # Train the final model with the best hyperparameters for this fold
        dt_model_final_fold_sa = DecisionTreeClassifier(max_depth=best_params[0], min_samples_split=best_params[1], min_samples_leaf=best_params[2], random_state=24)
        dt_model_final_fold_sa.fit(X_train_fold_sa, y_train_fold_sa)

        # Predict using the final model for this fold
        start_predict_fold_sa = time.time()
        y_pred_fold_sa = dt_model_final_fold_sa.predict(X_test_fold_sa)
        end_predict_fold_sa = time.time()

        # Append metrics to lists
        all_y_test_sa.extend(y_test_fold_sa)
        all_y_pred_sa.extend(y_pred_fold_sa)

        # Generate confusion matrix and display for the fold
        cm_sa = confusion_matrix(y_test_fold_sa, y_pred_fold_sa)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_sa = ConfusionMatrixDisplay(confusion_matrix=cm_sa)
        disp_sa.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - SA")
        pname_sa = method + "_fold_" + str(fold_number) + "_SA_confusion_matrix.png"
        plt.savefig(pname_sa)
        #plt.show()

        # Calculate time metrics
        time_to_train_fold_sa = end_train_fold_sa - start_train_fold_sa
        time_to_predict_fold_sa = end_predict_fold_sa - start_predict_fold_sa

        # Update overall time metrics
        time_to_train_sa += time_to_train_fold_sa
        time_to_predict_fold_sa += time_to_predict_fold_sa

    # Calculate overall performance metrics
    end_cv_sa = time.time()
    result(all_y_pred_sa, all_y_test_sa, "Simulated Annealing (SA)", time_to_train_sa, time_to_predict_fold_sa)

    outfile.close()
    outfile = open(fname, 'a')
    print("SA with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("Simulated Annealing")

print("*"*30)

try:
    """## SVM with Optimization and k-fold Cross-Validation"""

    # Reshape input data to have two dimensions
    X_flattened = X.reshape(X.shape[0], -1)

    # Initialize k-fold cross-validation
    skf = StratifiedKFold(n_splits= n_splits_for_cv, shuffle=True, random_state=42)

    # Initialize lists to store metrics
    all_y_test_svm = []
    all_y_pred_svm = []
    start_cv_svm = time.time()
    time_to_predict_fold_svm = 0
    time_to_train_svm = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_svm, test_index_svm) in enumerate(skf.split(X_flattened, y), 1):
        # Split data into train and test sets for the fold
        X_train_fold_svm, X_test_fold_svm = X_flattened[train_index_svm], X_flattened[test_index_svm]
        y_train_fold_svm, y_test_fold_svm = y[train_index_svm], y[test_index_svm]

        # Create and train the SVM model with optimization for this fold
        start_train_fold_svm = time.time()
        svm_model_fold = SVC(kernel='linear', C=1.0)  # Example hyperparameters, adjust as needed
        svm_model_fold.fit(X_train_fold_svm, y_train_fold_svm)
        end_train_fold_svm = time.time()
        time_to_train_svm += end_train_fold_svm - start_train_fold_svm

        # Predict using the trained model for this fold
        start_predict_fold_svm = time.time()
        y_pred_fold_svm = svm_model_fold.predict(X_test_fold_svm)
        end_predict_fold_svm = time.time()
        time_to_predict_fold_svm += end_predict_fold_svm - start_predict_fold_svm

        # Append metrics to lists
        all_y_test_svm.extend(y_test_fold_svm)
        all_y_pred_svm.extend(y_pred_fold_svm)

        # Generate confusion matrix and display for the fold
        cm_svm = confusion_matrix(y_test_fold_svm, y_pred_fold_svm)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
        disp_svm.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - SVM with Optimization")
        pname_svm = method + "_fold_" + str(fold_number) + "_SVM_with_Optimization_confusion_matrix.png"
        plt.savefig(pname_svm)

    # Calculate overall performance metrics
    end_cv_svm = time.time()
    result(all_y_pred_svm, all_y_test_svm, "SVM with Optimization and k-fold Cross-Validation", time_to_train_svm, time_to_predict_fold_svm)

    outfile.close()
    outfile = open(fname, 'a')
    print("SVM with Optimization and k-fold Cross-Validation Completed :)  ")




except Exception as e:
    logging.exception(str(e))
    separator("SVM")

print("*"*30)

try:
    """## 66 Stacking Classifier with k-fold Cross-Validation"""

    # Define base classifiers
    base_classifiers = [
        ('dt', DecisionTreeClassifier(random_state=24)),
        ('rf', RandomForestClassifier(random_state=24)),
        ('knn', KNeighborsClassifier())
    ]

    # Initialize Stacking Classifier with base classifiers
    stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=DecisionTreeClassifier())

    # Initialize lists to store metrics
    all_y_test_stacking = []
    all_y_pred_stacking = []
    start_cv_stacking = time.time()
    time_to_predict_fold_stacking = 0
    time_to_train_stacking = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_stacking, test_index_stacking) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_stacking, X_test_fold_stacking = X_train[train_index_stacking], X_train[test_index_stacking]
        y_train_fold_stacking, y_test_fold_stacking = y_train[train_index_stacking], y_train[test_index_stacking]

        # Train the Stacking Classifier for this fold
        start_train_fold_stacking = time.time()
        stacking_classifier.fit(X_train_fold_stacking, y_train_fold_stacking)
        end_train_fold_stacking = time.time()
        time_to_train_stacking += end_train_fold_stacking - start_train_fold_stacking

        # Predict using the trained model for this fold
        start_predict_fold_stacking = time.time()
        y_pred_fold_stacking = stacking_classifier.predict(X_test_fold_stacking)
        end_predict_fold_stacking = time.time()
        time_to_predict_fold_stacking += end_predict_fold_stacking - start_predict_fold_stacking

        # Append metrics to lists
        all_y_test_stacking.extend(y_test_fold_stacking)
        all_y_pred_stacking.extend(y_pred_fold_stacking)

        # Generate confusion matrix and display for the fold
        cm_stacking = confusion_matrix(y_test_fold_stacking, y_pred_fold_stacking)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_stacking = ConfusionMatrixDisplay(confusion_matrix=cm_stacking)
        disp_stacking.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Stacking Classifier")
        pname_stacking = method + "_fold_" + str(fold_number) + "_Stacking_Classifier_confusion_matrix.png"
        plt.savefig(pname_stacking)

    # Calculate overall performance metrics
    end_cv_stacking = time.time()


    result(all_y_pred_stacking, all_y_test_stacking, "Stacking Classifier",  time_to_train_stacking, time_to_predict_fold_stacking)

    outfile.close()
    outfile = open(fname, 'a')
    print("Stacking Classifier with k-fold Cross-Validation Completed :)  ")




except Exception as e:
    logging.exception(str(e))
    separator("Stacking Classifier")

print("*"*30)

try:
    """## 67 Stacking Dilated Convolutional Autoencoders with k-fold Cross-Validation"""

    # You may replace this with your DCAE model
    base_model = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=24))

    # Define the stacking classifier
    stacking_model = StackingClassifier(estimators=[('dt', base_model)], final_estimator=DecisionTreeClassifier())

    # Initialize lists to store metrics
    all_y_test_stacking_dcae = []
    all_y_pred_stacking_dcae = []
    time_to_predict_fold_stacking_dcae = 0
    time_to_train_stacking_dcae = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_stacking_dcae, test_index_stacking_dcae) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_stacking_dcae, X_test_fold_stacking_dcae = X_train[train_index_stacking_dcae], X_train[test_index_stacking_dcae]
        y_train_fold_stacking_dcae, y_test_fold_stacking_dcae = y_train[train_index_stacking_dcae], y_train[test_index_stacking_dcae]

        # Training the stacking model for this fold
        start_train_fold_stacking_dcae = time.time()
        stacking_model.fit(X_train_fold_stacking_dcae, y_train_fold_stacking_dcae)
        end_train_fold_stacking_dcae = time.time()
        time_to_train_stacking_dcae += end_train_fold_stacking_dcae - start_train_fold_stacking_dcae

        # Predict using the trained stacking model for this fold
        start_predict_fold_stacking_dcae = time.time()
        y_pred_fold_stacking_dcae = stacking_model.predict(X_test_fold_stacking_dcae)
        end_predict_fold_stacking_dcae = time.time()
        time_to_predict_fold_stacking_dcae += end_predict_fold_stacking_dcae - start_predict_fold_stacking_dcae

        # Generate confusion matrix for this fold
        cm_stacking_dcae = confusion_matrix(y_test_fold_stacking_dcae, y_pred_fold_stacking_dcae)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_stacking_dcae = ConfusionMatrixDisplay(confusion_matrix=cm_stacking_dcae)
        disp_stacking_dcae.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Stacking DCAE")
        pname_stacking_dcae = method + "_fold_" + str(fold_number) + "_Stacking_DCAE_confusion_matrix.png"
        plt.savefig(pname_stacking_dcae)

        # Append metrics to lists
        all_y_test_stacking_dcae.extend(y_test_fold_stacking_dcae)
        all_y_pred_stacking_dcae.extend(y_pred_fold_stacking_dcae)

    # Calculate overall performance metrics
    result(all_y_pred_stacking_dcae, all_y_test_stacking_dcae, "Stacking Dilated Convolutional Autoencoders", time_to_train_stacking_dcae, time_to_predict_fold_stacking_dcae)

    outfile.close()
    outfile = open(fname, 'a')
    print("Stacking Dilated Convolutional Autoencoders with k-fold Cross-Validation Completed :)  ")


except Exception as e:
    logging.exception(str(e))
    separator("Stacking Dilated Convolutional")

print("*"*30)

try:
    """## 68 Temporal Deep Feedforward Neural Network with k-fold Cross-Validation"""

    # Define the structure of the Temporal DNN model
    y_train = pd.Series(y_train)
    num_classes = y_train.nunique()
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Assuming len(y_train.unique()) is the number of classes
    ])

    # Compile the model
    model.compile(optimizer=Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Initialize lists to store metrics
    all_y_test_temporal_dnn = []
    all_y_pred_temporal_dnn = []
    time_to_predict_fold_temporal_dnn = 0
    time_to_train_temporal_dnn = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_temporal_dnn, test_index_temporal_dnn) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_temporal_dnn, X_test_fold_temporal_dnn = X_train[train_index_temporal_dnn], X_train[test_index_temporal_dnn]
        y_train_fold_temporal_dnn, y_test_fold_temporal_dnn = y_train[train_index_temporal_dnn], y_train[test_index_temporal_dnn]

        # Define the Temporal DNN model for this fold
        model_fold_temporal_dnn = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        model_fold_temporal_dnn.compile(optimizer=Adam(),
                                        loss='sparse_categorical_crossentropy',
                                        metrics=['accuracy'])

        # Training the Temporal DNN model for this fold
        start_train_fold_temporal_dnn = time.time()
        history = model_fold_temporal_dnn.fit(X_train_fold_temporal_dnn, y_train_fold_temporal_dnn, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        end_train_fold_temporal_dnn = time.time()
        time_to_train_temporal_dnn += end_train_fold_temporal_dnn - start_train_fold_temporal_dnn

        # Predict using the trained model for this fold
        start_predict_fold_temporal_dnn = time.time()
        y_pred_probs_fold_temporal_dnn = model_fold_temporal_dnn.predict(X_test_fold_temporal_dnn)
        y_pred_fold_temporal_dnn = np.argmax(y_pred_probs_fold_temporal_dnn, axis=1)
        end_predict_fold_temporal_dnn = time.time()
        time_to_predict_fold_temporal_dnn += end_predict_fold_temporal_dnn - start_predict_fold_temporal_dnn

        # Append metrics to lists
        all_y_test_temporal_dnn.extend(y_test_fold_temporal_dnn)
        all_y_pred_temporal_dnn.extend(y_pred_fold_temporal_dnn)

        # Generate confusion matrix for this fold
        cm_temporal_dnn = confusion_matrix(y_test_fold_temporal_dnn, y_pred_fold_temporal_dnn)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_temporal_dnn = ConfusionMatrixDisplay(confusion_matrix=cm_temporal_dnn)
        disp_temporal_dnn.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Temporal DNN")
        pname_temporal_dnn = method + "_fold_" + str(fold_number) + "_Temporal_DNN_confusion_matrix.png"
        plt.savefig(pname_temporal_dnn)

    # Calculate overall performance metrics
    result(all_y_pred_temporal_dnn, all_y_test_temporal_dnn, "Temporal Deep Feedforward Neural Network", time_to_train_temporal_dnn, time_to_predict_fold_temporal_dnn)

    outfile.close()
    outfile = open(fname, 'a')
    print("Temporal Deep Feedforward Neural Network with k-fold Cross-Validation Completed :)  ")





except Exception as e:
    logging.exception(str(e))
    separator("Temporal Deep Feedforward Neural")

print("*"*30)

try:
    """## 69 Voting Classifier with k-fold Cross-Validation"""

    # Initialize individual classifiers
    dt1 = DecisionTreeClassifier(random_state=24)
    dt2 = DecisionTreeClassifier(random_state=42)
    # Add more classifiers if needed

    # Create the Voting Classifier
    voting_clf = VotingClassifier(estimators=[
        ('dt1', dt1),
        ('dt2', dt2),
        # Add more classifiers here if needed
    ], voting='hard')  # You can use 'soft' voting if your classifiers support probability prediction

    # Initialize lists to store metrics
    all_y_test_voting = []
    all_y_pred_voting = []
    time_to_predict_fold_voting = 0
    time_to_train_voting = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_voting, test_index_voting) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_voting, X_test_fold_voting = X_train[train_index_voting], X_train[test_index_voting]
        y_train_fold_voting, y_test_fold_voting = y_train[train_index_voting], y_train[test_index_voting]

        # Train the Voting Classifier for this fold
        start_train_fold_voting = time.time()
        voting_clf.fit(X_train_fold_voting, y_train_fold_voting)
        end_train_fold_voting = time.time()
        time_to_train_voting += end_train_fold_voting - start_train_fold_voting

        # Predict using the trained model for this fold
        start_predict_fold_voting = time.time()
        y_pred_fold_voting = voting_clf.predict(X_test_fold_voting)
        end_predict_fold_voting = time.time()
        time_to_predict_fold_voting += end_predict_fold_voting - start_predict_fold_voting

        # Append metrics to lists
        all_y_test_voting.extend(y_test_fold_voting)
        all_y_pred_voting.extend(y_pred_fold_voting)

        # Generate confusion matrix for this fold
        cm_voting = confusion_matrix(y_test_fold_voting, y_pred_fold_voting)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_voting = ConfusionMatrixDisplay(confusion_matrix=cm_voting)
        disp_voting.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - Voting Classifier")
        pname_voting = method + "_fold_" + str(fold_number) + "_Voting_Classifier_confusion_matrix.png"
        plt.savefig(pname_voting)

    # Calculate overall performance metrics
    result(all_y_pred_voting, all_y_test_voting, "Voting Classifier", time_to_train_voting, time_to_predict_fold_voting)

    outfile.close()
    outfile = open(fname, 'a')
    print("Voting Classifier with k-fold Cross-Validation Completed :)  ")







except Exception as e:
    logging.exception(str(e))
    separator("Voting Classifier")

print("*"*30)

try:
    """## 70 GBT with k-fold Cross-Validation"""

    # Initialize the GBT classifier
    gbt_model = GradientBoostingClassifier(random_state=24)

    # Initialize lists to store metrics
    all_y_test_gbt = []
    all_y_pred_gbt = []
    time_to_predict_fold_gbt = 0
    time_to_train_gbt = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_gbt, test_index_gbt) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_gbt, X_test_fold_gbt = X_train[train_index_gbt], X_train[test_index_gbt]
        y_train_fold_gbt, y_test_fold_gbt = y_train[train_index_gbt], y_train[test_index_gbt]

        # Train the GBT model for this fold
        start_train_fold_gbt = time.time()
        gbt_model.fit(X_train_fold_gbt, y_train_fold_gbt)
        end_train_fold_gbt = time.time()
        time_to_train_gbt += end_train_fold_gbt - start_train_fold_gbt

        # Predict using the trained model for this fold
        start_predict_fold_gbt = time.time()
        y_pred_fold_gbt = gbt_model.predict(X_test_fold_gbt)
        end_predict_fold_gbt = time.time()
        time_to_predict_fold_gbt += end_predict_fold_gbt - start_predict_fold_gbt

        # Append metrics to lists
        all_y_test_gbt.extend(y_test_fold_gbt)
        all_y_pred_gbt.extend(y_pred_fold_gbt)

        # Generate confusion matrix for this fold
        cm_gbt = confusion_matrix(y_test_fold_gbt, y_pred_fold_gbt)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_gbt = ConfusionMatrixDisplay(confusion_matrix=cm_gbt)
        disp_gbt.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - GBT")
        pname_gbt = method + "_fold_" + str(fold_number) + "_GBT_confusion_matrix.png"
        plt.savefig(pname_gbt)

    # Calculate overall performance metrics
    result(all_y_pred_gbt, all_y_test_gbt, "GBT", time_to_train_gbt, time_to_predict_fold_gbt)

    outfile.close()
    outfile = open(fname, 'a')
    print("GBT with k-fold Cross-Validation Completed :)  ")





except Exception as e:
    logging.exception(str(e))
    separator("GBT")

print("*"*30)

try:
    """## 71 CART with k-fold Cross-Validation"""

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index  # Index of feature to split on
            self.threshold = threshold  # Threshold value to split on
            self.left = left  # Left subtree
            self.right = right  # Right subtree
            self.value = value  # Class label (for leaf nodes)

    class CART:
        def __init__(self, max_depth=None):
            self.max_depth = max_depth

        def fit(self, X, y):
            self.n_classes = len(set(y))
            self.n_features = X.shape[1]
            self.tree = self._build_tree(X, y)

        def _build_tree(self, X, y, depth=0):
            n_samples, n_features = X.shape
            n_labels = len(set(y))

            # Stop conditions
            if depth == self.max_depth or n_labels == 1:
                value = max(set(y), key=list(y).count)
                return Node(value=value)

            # Find best split
            best_gini = float('inf')
            best_feature_index = None
            best_threshold = None

            for feature_index in range(n_features):
                thresholds = sorted(set(X[:, feature_index]))
                for threshold in thresholds:
                    left_indices = (X[:, feature_index] <= threshold)
                    right_indices = (X[:, feature_index] > threshold)

                    left_gini = self._gini(y[left_indices])
                    right_gini = self._gini(y[right_indices])

                    gini = (len(left_indices) * left_gini + len(right_indices) * right_gini) / n_samples

                    if gini < best_gini:
                        best_gini = gini
                        best_feature_index = feature_index
                        best_threshold = threshold

            # Split data
            left_indices = (X[:, best_feature_index] <= best_threshold)
            right_indices = (X[:, best_feature_index] > best_threshold)

            left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
            right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

            return Node(best_feature_index, best_threshold, left_subtree, right_subtree)

        def _gini(self, y):
            n_samples = len(y)
            gini = 1.0
            for label in set(y):
                proportion = (y == label).sum() / n_samples
                gini -= proportion ** 2
            return gini

        def predict(self, X):
            return [self._predict_one(sample, self.tree) for sample in X]

        def _predict_one(self, sample, node):
            if node.value is not None:
                return node.value

            if sample[node.feature_index] <= node.threshold:
                return self._predict_one(sample, node.left)
            else:
                return self._predict_one(sample, node.right)

    # Initialize lists to store metrics
    all_y_test_cart = []
    all_y_pred_cart = []
    time_to_predict_fold_cart = 0
    time_to_train_cart = 0

    # Perform k-fold cross-validation
    for fold_number, (train_index_cart, test_index_cart) in enumerate(skf.split(X_train, y_train), 1):
        # Split data into train and test sets for the fold
        X_train_fold_cart, X_test_fold_cart = X_train[train_index_cart], X_train[test_index_cart]
        y_train_fold_cart, y_test_fold_cart = y_train[train_index_cart], y_train[test_index_cart]

        # Create and train the CART model for this fold
        cart_model_fold = CART(max_depth=1)
        start_train_fold_cart = time.time()
        cart_model_fold.fit(X_train_fold_cart, y_train_fold_cart)
        end_train_fold_cart = time.time()
        time_to_train_cart += end_train_fold_cart - start_train_fold_cart

        # Predict using the trained model for this fold
        start_predict_fold_cart = time.time()
        y_pred_fold_cart = cart_model_fold.predict(X_test_fold_cart)
        end_predict_fold_cart = time.time()
        time_to_predict_fold_cart += end_predict_fold_cart - start_predict_fold_cart

        # Append metrics to lists
        all_y_test_cart.extend(y_test_fold_cart)
        all_y_pred_cart.extend(y_pred_fold_cart)

        # Generate confusion matrix and display for the fold
        cm_cart = confusion_matrix(y_test_fold_cart, y_pred_fold_cart)
        plt.rcParams['figure.figsize'] = 8, 8
        sns.set_style("white")
        disp_cart = ConfusionMatrixDisplay(confusion_matrix=cm_cart)
        disp_cart.plot(cmap=plt.cm.Blues)
        plt.title(f"Fold {fold_number} Confusion Matrix - CART")
        pname_cart = method + "_fold_" + str(fold_number) + "_CART_confusion_matrix.png"
        plt.savefig(pname_cart)

    # Calculate overall performance metrics
    result(all_y_pred_cart, all_y_test_cart, "CART", time_to_train_cart, time_to_predict_fold_cart)

    outfile.close()
    outfile = open(fname, 'a')
    print("CART with k-fold Cross-Validation Completed :)  ")

except Exception as e:
    logging.exception(str(e))
    separator("CART")

## **Closing the outfile**
outfile.close()




import os

# Set the path to your directory
directory_path = '.'

# Patterns to look for in filenames
patterns = ['Metrics_fold_1', 'Metrics_fold_2', 'Metrics_fold_3', 'Metrics_fold_4']

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a PNG and contains any of the patterns
    if filename.endswith('.png') and any(pattern in filename for pattern in patterns):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        try:
            # Delete the file
            os.remove(file_path)
            print(f'Deleted: {file_path}')
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')

print('Done.')

# Your code goes here...

# Get the ending time
end_time_1 = datetime.datetime.now()

# Calculate the total time required for execution
total_time = end_time_1 - start_time_1

# Format the start time, end time, and total time
start_time_str = start_time_1.strftime("%d-%m-%Y %H:%M")
end_time_str = end_time_1.strftime("%d-%m-%Y %H:%M")
total_time_str = str(total_time)

# Write the start time, end time, and total time to the file
with open("time.txt", "a") as file:
    file.write(f"Start time: {start_time_str}\n")
    file.write(f"End time: {end_time_str}\n")
    file.write(f"Total time: {total_time_str}")


