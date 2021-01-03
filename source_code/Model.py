import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


train_data = pd.read_csv(r'..\csv_data\train.csv')
test_data = pd.read_csv(r'..\csv_data\test.csv')

# train_data.fillna(0)

# train_data[['Cabin']].replace('NAN', '0', inplace=True)

train_data['Embarked'] = train_data['Embarked'].replace(np.nan, "0")

print("Dropping unique traits: ==>\n")
train_data = train_data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)


encoder = preprocessing.LabelEncoder()

sex = (encoder.fit_transform(np.ravel(train_data[['Sex']])))
embarked = preprocessing.scale(encoder.fit_transform(np.ravel(train_data[['Embarked']])))
fare = preprocessing.scale(np.ravel(train_data[['Fare']]))
train_data['Sex'] = sex
train_data['Embarked'] = embarked
train_data['Fare'] = fare
train_data['Pclass'] = preprocessing.scale(np.ravel(train_data[['Pclass']]))

print(train_data.head())
print("\n\n")

print("Correlation")
for i in train_data.columns:
    print(f"Survived vs {i}: ", pd.Series.corr(train_data['Survived'], train_data[i]))


train_data = train_data[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked']]

print("New ==> \n")

print(train_data.head())

predict = "Survived"
X = train_data.drop([predict], axis=1)
y = train_data[[predict]]

print(X.head())
print(y.head())


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

model = KNeighborsClassifier(n_neighbors=4)

model.fit(train_X, np.ravel(train_y))

predictions = model.predict(test_X)

# for index, prediction in enumerate(predictions):
#     print(prediction, np.ravel(train_y)[index])


print(model.score(train_X, train_y))
