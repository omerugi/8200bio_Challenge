# 8200 Bio Data Challenge 2021 - DermaDetect Ltd.
# Dear participant,
# This file is provided for guidance and framework, for your convenience.
# Feel free to modify it at will, add (well-documented) input arguments and functionalities, etc.
# ... so long the file is able to be evaluated as described in the instructions.
# Official submission is defined as the last commit to your personal branch
# at the challenge's end time.

# Enjoy your journey to the world of teledermatology & AI diagnostics, and good luck to all!
# DermaDetect

# DermaDetect Copyright (C), 2021

from pandas.core.arrays.integer import Int64Dtype
from scipy.sparse import data
from scipy.sparse.construct import random
from sklearn import tree
import sklearn
import pandas as pd
import pickle
import argparse
import os
import numpy as np

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2


#from ctgan import CTGANSynthesizer
#from torch._C import dtype
# from torch._C import int64

rng =  np.random.RandomState(0)

class DecisionTreeTrainer:
    """ A class for training a decision tree on DermaDetect data """
    trained_model_filename = 'trained_model.pkl'
    data_relative_path = 'data/dd_data.csv'

    def __init__(self):
        self.main_dir = os.path.dirname(os.path.abspath(__file__))

        # TODO Fine-tune the decision tree classifier's parameters here
        self.model = tree.DecisionTreeClassifier(random_state=rng,splitter='best')

    def load_data(self, input_file):
        data = pd.read_csv(input_file)
        preprocessed_data = self.preprocess_data(data)
        return preprocessed_data

    def load_training_data(self):
        input_file = os.path.join(self.main_dir, self.data_relative_path)
        return self.load_data(input_file)

    def save_model(self):
        filename = os.path.join(self.main_dir, self.trained_model_filename)
        print(filename)
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        filename = os.path.join(self.main_dir, self.trained_model_filename)
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        self.model = model

    def preprocess_data(self, data):
        
        original = data.copy()
        
        y_mask = data['pain.pain_type'].isna()
        index_train_pain = np.asarray(y_mask[y_mask == False].index)
        index_test_pain = np.asarray(y_mask[y_mask == True].index)

        y_mask_pain = data['location.coverage'].isna()
        index_train = np.asarray(y_mask_pain[y_mask_pain == False].index)
        index_test = np.asarray(y_mask_pain[y_mask_pain == True].index)

        mask = data.isnull()
        data = data.fillna('unknown')

        for str in ['diagnosis', 'topography', 'location.coverage', 'size', 'shape', 'pain.pain_type', 'gender', 'lossOfHair.type', 'quantity']:
            data[str] = LabelEncoder().fit_transform(data[str])

        data = data.where(~mask, original)

        # imputer = KNNImputer(n_neighbors=18,weights='distance')
        # data = pd.DataFrame(imputer.fit_transform(data))
        # data.columns = original.columns

        # scaling the numric data
        scaler = MinMaxScaler()
        data[['duration.days', 'temperature', 'age']] = scaler.fit_transform(data[['duration.days', 'temperature', 'age']])
        data.replace({False: 0, True: 1}, inplace=True)

        train = pd.DataFrame(data.to_numpy()[index_train, :], columns=data.columns)
        x_train = train.drop('diagnosis', axis=1).drop('location.coverage', axis=1).drop('pain.pain_type',axis=1)
        y_train = train['location.coverage']
        test = pd.DataFrame(data.to_numpy()[index_test, :], columns=data.columns)
        x_test = test.drop('diagnosis', axis=1).drop('location.coverage', axis=1).drop('pain.pain_type',axis=1)



        from sklearn.ensemble import RandomForestClassifier
        fill_model = RandomForestClassifier(random_state=rng)
        fill_model.fit(x_train, y_train)
        y_test = fill_model.predict(x_test)
        data.loc[index_test,'location.coverage'] = y_test

        # pain completion 
        train = pd.DataFrame(data.to_numpy()[index_train_pain, :], columns=data.columns)
        x_train = train.drop('diagnosis', axis=1).drop('pain.pain_type',axis=1)
        y_train = train['pain.pain_type']
        test = pd.DataFrame(data.to_numpy()[index_test_pain, :], columns=data.columns)
        x_test = test.drop('diagnosis', axis=1).drop('pain.pain_type',axis=1)

        fill_model = RandomForestClassifier(random_state=rng)
        fill_model.fit(x_train, y_train)
        y_test = fill_model.predict(x_test)
        data.loc[index_test_pain,'pain.pain_type'] = y_test

        # place holders
        X = data.drop('diagnosis',axis=1)
        y = data['diagnosis']

        # split data into train & test groups
        inputs_train, inputs_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=rng, stratify=y)

        train = pd.concat([inputs_train, outputs_train],axis=1)
        train.columns = data.columns
        test = pd.concat([inputs_test, outputs_test],axis=1)
        test.columns = data.columns

        # train model to assess feature importance
        model = tree.DecisionTreeClassifier(random_state=rng)
        model.fit(inputs_train,outputs_train)
        dfscores = pd.DataFrame(model.feature_importances_)
        dfcolumns = pd.DataFrame(inputs_train.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        
        feture_n = 88
        #featureScores.nlargest(feture_n, 'Score')['Specs'].to_csv("88feacture.csv")
        inputs_train = inputs_train[featureScores.nlargest(feture_n, 'Score')['Specs']]
        inputs_test = inputs_test[featureScores.nlargest(feture_n, 'Score')['Specs']]
        
        train = pd.concat([inputs_train,outputs_train],axis=1)
        test = pd.concat([inputs_test,outputs_test],axis=1)
        return train,test

    def train(self, inputs, outputs):
        self.model.fit(inputs, outputs)
        self.save_model()

    def evaluate(self, inputs, outputs):
        y_pred = self.model.predict(inputs)
        #print(classification_report(outputs, y_pred))

        # TODO: accurately compute the amount of correct predictions over the inputs size in %
        return sklearn.metrics.accuracy_score(outputs, y_pred)*100


################################################################################
def get_cmd_args():
    """ Input arguments """
    args = argparse.ArgumentParser(
        'DermaDetect - 8200 Data challenge 2021 - Decision tree diagnostics')
    args.add_argument('--evaluate_csv', '-e', type=str,
                      default='', help='Path to the test dataset')
    return args.parse_args()

#from imblearn.over_sampling import SMOTENC
################################################################################
def main():
    args = get_cmd_args()
    do_train = len(args.evaluate_csv) == 0

    the_tree = DecisionTreeTrainer()

    if do_train:
        train, test = the_tree.load_training_data()
        
        # training data agmuantion
        np.random.seed(32)
        for d in train['diagnosis'].value_counts().index:
            new_data = {}
            df = train[train["diagnosis"] == d]
            for c in train.columns:
                VC = df[c].value_counts()
                new_data[c] = (np.random.choice(VC.index, size=10, p=VC / VC.sum()))
            new_data["diagnosis"] = np.full(10, d)
            train = pd.concat([train, pd.DataFrame(new_data)])

        #train and save model
        the_tree.train(train.drop('diagnosis',axis=1), train['diagnosis'])

        print(train.shape)
        #evaluate on test group
        print('accuracy', the_tree.evaluate(test.drop('diagnosis',axis=1), test['diagnosis']))

    else:  # evaluate
        the_tree.load_model()
        test_data = the_tree.load_data(args.evaluate_csv)

        # TODO optionally save, display, etc.

        # Lastly, print test accuracy
        test_accuracy_percent = the_tree.evaluate(
            test_data.drop('diagnosis', axis=1), test_data['diagnosis'])
        print("Test accuracy: {:.2f}%".format(
            test_accuracy_percent))  # in [0-100], eg 99.99%


################################################################################
if __name__ == '__main__':
    main()
