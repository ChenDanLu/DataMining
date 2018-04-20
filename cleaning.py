import csv
import pandas as pd
import numpy as np


data = pd.read_csv('./data/Building_Permits.csv', encoding='gbk', header=0, engine='python')

# drop_col = ['Unit',
#             'Unit Suffix',
#             'Street Number Suffix',
#             'Fire Only Permit',
#             'Structural Notification',
#             'TIDF Compliance',
#             'Voluntary Soft-Story Retrofit',
#             'Site Permit']
drop_col = [k for k,v in dict(data.isnull().sum(axis=0)).items() if v > 120000]
data = data.drop(drop_col, axis=1)

cor_col = {'Number of Existing Stories':'Number of Proposed Stories',
           'Existing Use':'Proposed Use',
           'Existing Units':'Proposed Units',
           'Estimated Cost':'Revised Cost'}
for c1,c2 in cor_col.items():
    condition = (data[c1].isnull()) & (data[c2].notnull())
    data.loc[condition, c1] = data.loc[condition, c2]
    condition = (data[c2].isnull()) & (data[c1].notnull())
    data.loc[condition, c2] = data.loc[condition, c1]

mode_col = ['Existing Construction Type',
            'Existing Construction Type Description',
            'Proposed Construction Type',
            'Proposed Construction Type Description',
            'Number of Proposed Stories',
            'Number of Existing Stories',
            'Existing Use',
            'Proposed Use',
            'Existing Units',
            'Proposed Units']
for col in mode_col:
    mode = data[col].dropna().mode()[0]
    data[col] = data[col].fillna(mode)

# mean_col = []
# for col in mean_col:
#     mean = data[col].dropna().mean()
#     data[col] = data[col].fillna(mean)

none_col = {'Permit Expiration Date':'NoDate',
            'First Construction Document Date':'NoDate',
            'Completed Date':'NoDate',
            'Issued Date':'NoDate',
            'Street Suffix':'NoSuffix',
            'Description':'NoDescrip',
            'Estimated Cost':0,
            'Revised Cost':0,
            'Plansets':0,}
data = data.fillna(none_col)

data = data.dropna()

data.to_csv('./data/new_Building_Permits.csv', index=False)