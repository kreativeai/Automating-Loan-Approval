# import Flask and jsonify
from flask import Flask, jsonify

# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import collections
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle
import joblib

df = pd.read_csv("data.csv")

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

y_train = y_train.map({'N':0, 'Y':1}).astype(int)
y_test = y_test.map({'N':0, 'Y':1}).astype(int)

cat_feats = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
num_feats = ['LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'EMI', 'Balance_Income']


# Using own function in Pipeline
def removeLoanID(data):
    if 'Loan_ID' in data:
        return data.drop('Loan_ID', axis=1)
    else:
        return data
def prepareTotalIncome(data):
    data['Total_Income']=data['ApplicantIncome']+data['CoapplicantIncome']
    data = data.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)
    return data

def prepareEMI(data):
    data['EMI']=data['LoanAmount']/data['Loan_Amount_Term']
    return data

def prepareBalanceIncome(data):
    data['Balance_Income']=data['Total_Income']-(data['EMI']*1000)
    return data

def logLoanAmount(data):
    data['LoanAmount']=np.log(data['LoanAmount'])
    return data

def logTotalIncome(data):
    data['Total_Income']=np.log(data['Total_Income'])
    return data


def numFeat(data):
    return data[num_feats]

def catFeat(data):
    return data[cat_feats]

class ToDenseTransformer():

    # here you define the operation it should perform
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    # just return self
    def fit(self, X, y=None, **fit_params):
        return self



app = Flask(__name__)

api = Api(app)

class LoadPrediction(Resource):
    def get(self):
        # create request parser
        parser = reqparse.RequestParser()

        # create argument 'Loan_ID'
        parser.add_argument('Loan_ID', type=str, required=True)
        # parse 'Loan_ID'
        Loan_ID = parser.parse_args().get('Loan_ID')
     
        # create argument 'Gender'
        parser.add_argument('Gender', type=str, required=True)
        # parse 'Gender'
        Gender = parser.parse_args().get('Gender')

        # create argument 'Married'
        parser.add_argument('Married', type=str, required=True)
        # parse 'Married'
        Married = parser.parse_args().get('Married')

        # create argument 'Dependents'
        parser.add_argument('Dependents', type=str, required=True)
        # parse 'Dependents'
        Dependents = parser.parse_args().get('Dependents')

        # create argument 'Education'
        parser.add_argument('Education', type=str, required=True)
        # parse 'Education'
        Education = parser.parse_args().get('Education')

        # create argument 'Self_Employed'
        parser.add_argument('Self_Employed', type=str, required=True)
        # parse 'Self_Employed'
        Self_Employed = parser.parse_args().get('Self_Employed')

        # create argument 'ApplicantIncome'
        parser.add_argument('ApplicantIncome', type=int, required=True)
        # parse 'ApplicantIncome'
        ApplicantIncome = parser.parse_args().get('ApplicantIncome')

        # create argument 'CoapplicantIncome'
        parser.add_argument('CoapplicantIncome', type=int, required=True)
        # parse 'CoapplicantIncome'
        CoapplicantIncome = parser.parse_args().get('CoapplicantIncome')

        # create argument 'LoanAmount'
        parser.add_argument('LoanAmount', type=int, required=True)
        # parse 'LoanAmount'
        LoanAmount = parser.parse_args().get('LoanAmount')

        # create argument 'Loan_Amount_Term'
        parser.add_argument('Loan_Amount_Term', type=int, required=True)
        # parse 'Loan_Amount_Term'
        Loan_Amount_Term = parser.parse_args().get('Loan_Amount_Term')

        # create argument 'Credit_History'
        parser.add_argument('Credit_History', type=int, required=True)
        # parse 'Credit_History'
        Credit_History = parser.parse_args().get('Credit_History')

        # create argument 'Property_Area'
        parser.add_argument('Property_Area', type=str, required=True)
        # parse 'Property_Area'
        Property_Area = parser.parse_args().get('Property_Area')


        # load the model from disk
        filename = 'finalized_model.sav'
        loaded_pipeline = joblib.load(filename)

        input_data = {
            'Loan_ID': Loan_ID,
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area}

        df_new = pd.DataFrame([input_data])

        df_new['Loan_Status'] = loaded_pipeline.predict(df_new)

        df_new['Loan_Status'] = df_new['Loan_Status'].map({0:'N', 1:'Y'})

        Loan_Result = df_new['Loan_Status'][0]


        # make json from result 
        return jsonify(result=Loan_Result)


# assign endpoint
api.add_resource(LoadPrediction, '/loanpred',)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
    
