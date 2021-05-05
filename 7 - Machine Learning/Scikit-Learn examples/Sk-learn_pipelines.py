#!/usr/bin/python

'''
 Example of how to perform advanced data analytics with scikit-learn library
 Example from https://towardsdatascience.com/
'''

import numpy as np
import pandas as pd
from seaborn import load_dataset

# Machine learning pipeline
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier

# For the good luck :)
seed = 13

'''
Step 1 : Analyze data
'''
exclude = ['pclass', 'embarked', 'who', 'adult_male', 'alive', 'alone']
df = load_dataset('titanic').drop(columns=exclude)

# Inspect shape of data and top rows
print(f"{df.shape[0]} rows, {df.shape[1]} columns")
print(df.columns)
print(df.head())

'''
 Step 2 : Choose a suitable algorithm -> how ? -> try multiple and combine with intuition
 ML model training requires partition of the data into train and test set
'''
# Set target
target = 'survived'
features = df.drop(columns=target).columns

# Split data into train & test
X_train, X_test, y_train, y_test = train_test_split(
                                                    df[features], df[target], 
                                                    test_size=.2, random_state=seed, 
                                                    stratify=df[target]
                                                   )

# Inspect data
print(f"Training data ({X_train.shape[0]} rows): Target distribution")
print(y_train.value_counts(normalize=True))
print(f"\nTest data ({X_test.shape[0]} rows): Target distribution")
print(y_train.value_counts(normalize=True))

# Define feature groups
numerical = X_train.select_dtypes(['number']).columns
print(f'\nNumerical: {numerical}')

categorical = X_train.columns.difference(numerical)
X_train[categorical] = X_train[categorical].astype('object')
print(f'Categorical: {categorical}')


'''
 Step 3 : A big problem when training ML models resides in how to load train samples
          In this example we prepare some data loaders that perform some preprocessing
            
          Imputer: Imputes with a constant value and returns the imputed data in a pandas DataFrame

          CardinalityReducer: Aggregates infrequent categories into ‘other’ category and returns the transformed data in a pandas DataFrame
'''

class Imputer(BaseEstimator, TransformerMixin):
    """A custom transformer that imputes with a constant value in place.
    
    Parameters
    ----------
    value: (optional) A value to impute with
    """
    def __init__(self, value='missing'):
        self.value = value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.fillna(self.value, inplace=True)
        return X



class CardinalityReducer(BaseEstimator, TransformerMixin):
    """A custom transformer that encodes infrequent labels into 'other' in place.
    
    Parameters
    ----------
    threshold: (optional) An integer for minimum threshold frequency count or a float for threshold of frequency proportion to keep the category.
    If category frequency doesn't surpass the threshold, its value will be overwritten with 'other'.
    """
    def __init__(self, threshold=.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.top_categories = {}
        for feature in X.columns:
            frequencies = pd.Series(X[feature].value_counts(normalize=True))
            if isinstance(self.threshold, int):
                top_categories = frequencies.head(self.threshold).index
            elif isinstance(self.threshold, float):   
                top_categories = frequencies[frequencies>self.threshold].index
            self.top_categories[feature] = list(top_categories)
        return self

    def transform(self, X):
        for feature in X.columns:
            X[feature] = np.where(
                X[feature].isin(self.top_categories[feature]),
                X[feature],
                'other')
        return X


'''
Actually, preprocess the data
'''
# Build preprocessing pipeline
categorical_pipe = Pipeline(
    [
        ('imputer', Imputer()),
        ('cardinality_reducer', CardinalityReducer()),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]
)

numerical_pipe = Pipeline(
    [
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ]
)

preprocessor = ColumnTransformer(transformers = [
    ('cat', categorical_pipe, categorical),
    ('num', numerical_pipe, numerical)]
)

# Fit and transform training data
preprocessor.fit(X_train)

cat = preprocessor.named_transformers_['cat']['encoder'].get_feature_names(categorical)
columns = np.append(cat, numerical)

X_train_transformed = pd.DataFrame(preprocessor.transform(X_train), columns=columns)
X_train_transformed.head()

'''
 Step 4 : Do some serious ML
'''

def create_baseline_classifiers(seed=seed):
    """Create a list of baseline classifiers.
    
    Parameters
    ----------
    seed: (optional) An integer to set seed for reproducibility

    Returns
    -------
    A list containing tuple of name, model object for each of these algortihms:
    DummyClassifier, LogisticRegression, SGDClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, RandomForestClassifier, MultinomialNB, SVC, XGBClassifier.
    """
    models = []
    models.append(('dum', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('log', LogisticRegression(random_state=seed)))
    models.append(('sgd', SGDClassifier(random_state=seed)))
    models.append(('etc', ExtraTreesClassifier(random_state=seed)))
    models.append(('gbm', GradientBoostingClassifier(random_state=seed)))
    models.append(('rfc', RandomForestClassifier(random_state=seed)))
    models.append(('mnb', MultinomialNB()))
    models.append(('svc', SVC(random_state=seed, probability=True)))
    models.append(('xgb', XGBClassifier(seed=seed)))

    return models

# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
def assess_models(X, y, models, cv=5, metrics=['roc_auc', 'f1', 'accuracy']):
    """Provide summary of cross validation results for models.
    
    Parameters
    ----------
    X: A pandas DataFrame containing feature matrix
    y: A pandas Series containing target vector
    models: A list of models to train
    cv: (optional) An integer to set number of folds in cross-validation
    metrics: (optional) A list of scoring metrics or a string for a metric
    Returns
    -------
    A pandas DataFrame containing summary of baseline models' performance.
    """

    summary = pd.DataFrame()
    for name, model in models:
        result = pd.DataFrame(cross_validate(model, X, y, cv=cv, scoring=metrics))
        mean = result.mean().rename('{}_mean'.format)
        std = result.std().rename('{}_std'.format)
        summary[name] = pd.concat([mean, std], axis=0)
    return summary.sort_index()


def extract_metric(summary, metric):
    """Provide summary of baseline models' performance for a metric.
    
    Parameters
    ----------
    summary: A pandas DataFrame containing the summary of baseline models
    metric: A string specifying the name of the metric to extract info
    
    Returns
    -------
    A pandas DataFrame containing mean, standard deviation, lower and upper
    bound of the baseline models' performance in cross validation according to
    the metric specified.
    """

    output = summary[summary.index.str.contains(metric)].T
    output.columns = output.columns.str.replace(f'test_{metric}_', '')
    output.sort_values(by='mean', ascending=False, inplace=True)
    output['lower'] = output['mean'] - 2*output['std']
    output['upper'] = output['mean'] + 2*output['std']
    return output


models = create_baseline_classifiers()
summary = assess_models(X_train_transformed, y_train, models)
print(summary)
extract_metric(summary, "accuracy")
extract_metric(summary, "f1")