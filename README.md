# public_8200bio_challenge

Name: Roi Peleg, Lilach Mor, Omer Rugi

Email:  lilach199612@gmail.com ,omerihay@gmail.com

## In our preprocessing we did:

Label encoding - For all the discrete data.

OneHot encoding – on the labels (after label encoding them).

MinMaxScalar – on the numeric data.

Zeros & Ones – Replace the values of Booleans.

Data Completion – In the features ‘location_covrage’ and ‘pain.pain_type’ there was missing data, we tried RandomForest and KNN imputer to fill in the missing data. The samples that contained the data = train, the samples with missing data = test. The RandomForest was selected, performed better.

Feature/Demention reduction – using a Decision Tree model we found the 50 most important features, after running it we had a list of the most important features (rf_selected_features). After finding those, every call to the preprocessing cleaned the unnecessary features based on the rf_selected_features (both in train and in test).

Data Generator - creating new data based on the given sample's statistics and distribution.

## When evaluating the model : 
Split test train – did it in an equal manner between the classes, so the train will “see” all the possible labels.   

Generating data – used statistic sampling of the train data to generate more samples to train the model better.

OneHot Encoding – used it on the labels when training the model.

** Note: we tried – 
1.	Change all the discrete data as OneHot vectors but it didn’t work.
2.	Use GAN’s to generate data

Didn’t give good results.
