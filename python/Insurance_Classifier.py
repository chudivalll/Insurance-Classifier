import os
import sys

import numpy as np
import pandas as pd
import pandasql
import sklearn.metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/mac/Downloads/exercise_03_train.csv")
# %%
# Cleaning data and making aesthetic
df["x41"] = df["x41"].str.replace("$", "").astype(float)

df["x45"] = df["x45"].str.replace("%", "").astype(float)


categorical = df.loc[:, ["x34", "x35", "x68", "x93"]]

categorical["x34"] = categorical.x34.str.upper()
categorical["x35"] = categorical.x35.str.upper()
categorical["x68"] = categorical.x68.str.upper().str.replace(".", "")
categorical["x93"] = categorical.x93.str.upper()

categorical["x35"] = categorical["x35"].str.replace("WEDNESDAY", "WED")
categorical["x35"] = categorical["x35"].str.replace("THURDAY", "THUR")
categorical["x35"] = categorical["x35"].str.replace("TUESDAY", "TUES")
categorical["x35"] = categorical["x35"].str.replace("FRIDAY", "FRI")
categorical["x35"] = categorical["x35"].str.replace("MONDAY", "MON")

categorical["x68"] = categorical["x68"].str.replace("JANUARY", "JAN")
categorical["x68"] = categorical["x68"].str.replace("DEV", "DEC")

categorical.x34.fillna(value="VOLKSWAGON", inplace=True)
categorical.x35.fillna(value="WED", inplace=True)
categorical.x68.fillna(value="JULY", inplace=True)
categorical.x93.fillna(value="ASIA", inplace=True)

for x in categorical:
    print(categorical[x].value_counts(dropna=False))

categorical1 = pd.DataFrame(
    pandasql.sqldf(
        '''SELECT x35,
                                          CASE
                                              WHEN x35 = "WED" THEN "WEDNESDAY"
                                              WHEN x35 = "WEDNESDAY" THEN "WEDNESDAY"
                                          END AS xx35
                                          FROM categorical 
                                          WHERE x35 LIKE "W%"'''
    )
)

categorical1 = categorical1.drop("x35", axis=1, inplace=False).rename(
    columns={"xx35": "x35"}, inplace=False
)
# %%Isolated the numerical data for exploration
numerical = df.loc[:, df.dtypes == float]

# Utilized a univariate imputation algorithm for the missing values and scaled the results
imp = SimpleImputer()
imp_numerical = pd.DataFrame(imp.fit_transform(numerical))
imp_numerical.columns = numerical.columns

stan = preprocessing.StandardScaler().fit(numerical)
scaled_imp_numerical = pd.DataFrame(stan.transform(imp_numerical))

imp_numerical.info()

# %% Creating dummies and converting to arrays for model placement

dummies = pd.get_dummies(categorical)

df_features = np.array(pd.concat([scaled_imp_numerical, dummies], axis=1))

df_labels = np.array(df["y"])

# %%

# Split the data into training and testing sets for feature selection
train_df_features, test_df_features, train_df_labels, test_df_labels = train_test_split(
    df_features, df_labels, test_size=0.25, random_state=42
)

print("Training Features Shape:", train_df_features.shape)
print("Training Labels Shape:", train_df_labels.shape)
print("Testing Features Shape:", test_df_features.shape)
print("Testing Labels Shape:", test_df_labels.shape)
# %%
# checking classification accuracy with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_df_features, train_df_labels)
prediction1 = knn.predict_proba(test_df_features)


# %%

# Creating 10 folds for cross validation
scores = cross_val_score(knn, df_features, df_labels, cv=10, scoring="accuracy")
print(scores)
scores.mean()  # Returned a mean accracy score of 93.6%. Implemented hyperparameter tuning (for n_neighbors) in order to optimize performance.

# %%
# Finding optimal value of k for KNN (computationally expensive)
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, df_features, df_labels, cv=10, scoring="accuracy")
    k_scores.append(scores.mean())
print(
    k_scores
)  # Of the 30 values tested for k, k=7 produced the best results with an accuracy score of 0.937
# %%
# This grid Search performed the same loop above but took even longer
k_range = range(1, 11)
knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=k_range)
GS = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")

GS.fit(df_features, df_labels)
GS_probability = GS.predict_proba(test_df_features)
print("AUC: ", roc_auc_score(test_df_labels, GS_probability[:, 1]))
print("Accuracy: ", GS.score(test_df_features, test_df_labels))


# %%
log = LogisticRegression(solver="lbfgs")
log.fit(train_df_features, train_df_labels)
log.score(test_df_features, test_df_labels)
prediction2 = log.predict(test_df_features)
print("LR Trial Prediction: ", prediction2)

print(sklearn.metrics.confusion_matrix(test_df_labels, prediction2))
print(sklearn.metrics.accuracy_score(test_df_labels, prediction2))
print("AUC: ", roc_auc_score(test_df_labels, prediction2))
scores2 = cross_val_score(log, df_features, df_labels, cv=10, scoring="accuracy").mean()


# %%
