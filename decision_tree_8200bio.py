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

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, MultiLabelBinarizer, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2

# from ctgan import CTGANSynthesizer
# from torch._C import dtype
# from torch._C import int64

rng = np.random.RandomState(0)
num_col = ["topography"
            , "location.coverage"
            , "size"
            , "shape"
            , "pain.pain_type"
            , "gender"
            , "lossOfHair.type"
            , "quantity"
            , "duration.days"
            , "temperature"
            , "age"
            , "swelling"
            , "bleeding"
            , "is_secondary_locations_lips"
            , "is_secondary_locations_kneepit"
            , "is_secondary_locations_knee"
            , "is_secondary_locations_hand_internal"
            , "is_secondary_locations_hand_external"
            , "is_secondary_locations_hand"
            , "is_secondary_locations_groins"
            , "is_secondary_locations_gingiva"
            , "is_secondary_locations_forehead"
            , "is_secondary_locations_forearm_internal"
            , "is_secondary_locations_forearm_external"
            , "is_secondary_locations_foot_internal"
            , "is_secondary_locations_foot_external"
            , "is_secondary_locations_foot"
            , "is_secondary_locations_fingers"
            , "is_secondary_locations_eyelid"
            , "is_secondary_locations_eyebrow"
            , "is_secondary_locations_eye"
            , "is_secondary_locations_elbow_pit"
            , "is_secondary_locations_elbow"
            , "is_secondary_locations_ear"
            , "is_secondary_locations_nails"
            , "is_secondary_locations_navel"
            , "is_secondary_locations_neck"
            , "is_secondary_locations_thigh_internal"
            , "is_secondary_locations_wrist_internal"
            , "is_secondary_locations_wrist_external"
            , "is_secondary_locations_vaginal_region"
            , "is_secondary_locations_upper_arm_internal"
            , "is_secondary_locations_upper_arm_external"
            , "is_secondary_locations_tongue"
            , "is_secondary_locations_toes_internal"
            , "is_secondary_locations_toes_external"
            , "is_secondary_locations_toes"
            , "is_secondary_locations_thigh_external"
            , "is_secondary_locations_nose"
            , "is_secondary_locations_shoulders"
            , "is_secondary_locations_shin_internal"
            , "is_secondary_locations_shin_external"
            , "is_secondary_locations_scrotum"
            , "is_secondary_locations_scalp"
            , "is_secondary_locations_rear_neck"
            , "is_secondary_locations_pubes"
            , "is_secondary_locations_perioral"
            , "is_secondary_locations_penis"
            , "is_secondary_locations_chin"
            , "is_secondary_locations_chest"
            , "is_secondary_locations_cheeks"
            , "is_texture_rough"
            , "is_color_condition_normal"
            , "is_color_condition_grey"
            , "is_color_condition_green"
            , "is_color_condition_brown"
            , "is_color_condition_blue"
            , "is_color_condition_black"
            , "is_texture_wet"
            , "is_texture_smooth"
            , "is_texture_scales"
            , "is_texture_not_sure"
            , "is_color_condition_red"
            , "is_texture_exfoliation"
            , "is_texture_dry"
            , "is_texture_cracks"
            , "itch"
            , "crater"
            , "lossOfHair.exist"
            , "vesicle"
            , "pain.is_pain"
            , "duration.from_birth"
            , "is_color_condition_purple"
            , "is_color_condition_white"
            , "is_secondary_locations_buttock"
            , "is_primary_locations_arm"
            , "is_secondary_locations_back"
            , "is_secondary_locations_armpit"
            , "is_secondary_locations_abdomen"
            , "pus"
            , "is_primary_locations_head"
            , "is_primary_locations_groin_area"
            , "is_primary_locations_chest"
            , "is_primary_locations_buttock"
            , "is_primary_locations_back"
            , "is_nailsdiseases_yellow_nail_syndrome"
            , "is_color_condition_yellow"
            , "is_nailsdiseases_terry_s_nails"
            , "is_nailsdiseases_pitting"
            , "is_nailsdiseases_onychomycosis"
            , "is_nailsdiseases_onycholysis"
            , "is_nailsdiseases_mees__lines"
            , "is_nailsdiseases_leukonychia__white_spots_"
            , "is_nailsdiseases_koilonychia__spooning_"
            , "is_nailsdiseases_clubbing"
            , "is_nailsdiseases_beau_s_lines"
            , "is_primary_locations_leg"
                    ]
ord_col = ["location.coverage"
    , "size"
    , "shape"
    , "pain.pain_type"
    , "gender"
    , "lossOfHair.type"
    , "quantity"
           ]

class DecisionTreeTrainer:
    """ A class for training a decision tree on DermaDetect data """
    trained_model_filename = 'trained_model.pkl'
    data_relative_path = 'data/dd_data.csv'

    def __init__(self):
        self.main_dir = os.path.dirname(os.path.abspath(__file__))

        # TODO Fine-tune the decision tree classifier's parameters here
        self.model = tree.DecisionTreeClassifier(random_state=rng, splitter='best')

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

        mask = data.isnull()
        data = data.fillna('unknown')

        ## Working ver - 43%
        # for str in ['diagnosis', 'topography', 'location.coverage', 'size', 'shape', 'pain.pain_type', 'gender', 'lossOfHair.type', 'quantity']:
        #     data[str] = LabelEncoder().fit_transform(data[str])

        for str in num_col:
            data[str] = LabelEncoder().fit_transform(data[str])

        for str in ord_col:
            data[str] = OrdinalEncoder().fit_transform(data[str].to_numpy().reshape(-1,1))

        data["diagnosis"] = LabelEncoder().fit_transform(data["diagnosis"])

        # Return original Nan values
        data = data.where(~mask, original)

        imputer = KNNImputer(n_neighbors=18, weights='distance')
        data = pd.DataFrame(imputer.fit_transform(data))
        data.columns = original.columns

        # scaling the numric data
        scaler = MinMaxScaler()
        data[['duration.days', 'temperature', 'age']] = scaler.fit_transform( data[['duration.days', 'temperature', 'age']])

        ## Working ver - 43%
        #data.replace({False: 0, True: 1}, inplace=True)

        # place holders
        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']

        # train model to assess feature importance
        model = tree.DecisionTreeClassifier(random_state=rng)
        model.fit(X, self.OneHot(y))
        dfscores = pd.DataFrame(model.feature_importances_)
        dfcolumns = pd.DataFrame(X.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns

        feture_n = 88
        X = X[featureScores.nlargest(feture_n, 'Score')['Specs']]
        t = pd.concat([X,y],axis=1)
        return t

    def train(self, inputs, outputs):
        #inputs = pd.get_dummies(inputs, columns= list(set(num_col) & set(inputs.columns)))
        self.model.fit(inputs, self.OneHot(outputs))
        self.save_model()

    def evaluate(self, inputs, outputs):
        #inputs = pd.get_dummies(inputs, columns=list(set(num_col) & set(inputs.columns)))
        y_pred = self.model.predict(inputs)
        # print(classification_report(outputs, y_pred))
        # TODO: accurately compute the amount of correct predictions over the inputs size in %
        return sklearn.metrics.accuracy_score(self.OneHot(outputs), y_pred) * 100

    def OneHot(self, data):
        ohe = OneHotEncoder()
        y = data.to_numpy().reshape(-1, 1)
        y_hot = ohe.fit_transform(y).toarray()
        return y_hot


################################################################################
def get_cmd_args():
    """ Input arguments """
    args = argparse.ArgumentParser(
        'DermaDetect - 8200 Data challenge 2021 - Decision tree diagnostics')
    args.add_argument('--evaluate_csv', '-e', type=str,
                      default='', help='Path to the test dataset')
    return args.parse_args()


# from imblearn.over_sampling import SMOTENC
################################################################################
def main():
    args = get_cmd_args()
    do_train = len(args.evaluate_csv) == 0

    the_tree = DecisionTreeTrainer()

    if do_train:
        data = the_tree.load_training_data()
        #data = pd.get_dummies(data, columns=list(set(num_col) & set(data.columns)))
        # split data into train & test groups
        inputs_train, inputs_test, outputs_train, outputs_test = sklearn.model_selection.train_test_split(data.drop('diagnosis',axis=1), data['diagnosis'],
                                                                                                          test_size=0.1,
                                                                                                          random_state=rng,
                                                                                                          stratify=data['diagnosis'])
        train = pd.concat([inputs_train,outputs_train],axis=1)
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

        # train and save model
        the_tree.train(train.drop('diagnosis', axis=1), train['diagnosis'])
        # evaluate on test group
        print('accuracy', the_tree.evaluate(inputs_test, outputs_test))

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
