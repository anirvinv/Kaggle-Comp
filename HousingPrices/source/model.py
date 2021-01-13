import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

predict = 'SalePrice'

X = train_data.drop(predict, axis=1)
y = train_data[[predict]]

encoder = LabelEncoder()

for col in X.columns:
    X[col] = encoder.fit_transform(X[col])


print(y.head(), X.head())

# for col in train_data.columns:
#     corr = pd.Series.corr(train_data[predict], train_data[col])
#     print(f"Sales vs {col}: {corr}")
