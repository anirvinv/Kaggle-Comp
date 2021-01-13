import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle


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

# print(train_data.head())
print("\n\n")

print("Correlation")
for i in train_data.columns:
    print(f"Survived vs {i}: ", pd.Series.corr(train_data['Survived'], train_data[i]))


train_data = train_data[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked']]

# print("New ==> \n")

# print(train_data.head())

predict = "Survived"
X = train_data.drop([predict], axis=1)
y = train_data[[predict]]

# print(X.head())
# print(y.head())


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.15)

accuracy = 0

for n in range(1, 20):

    model = KNeighborsClassifier(n_neighbors=n)

    model.fit(train_X, np.ravel(train_y))

    predictions = model.predict(test_X)

    # for index, prediction in enumerate(predictions):
    #     print(prediction, np.ravel(train_y)[index])
    score = model.score(test_X, test_y)
    print(f"Neighbors: {n} Accuracy score:", score)

    if score > accuracy:
        pickle.dump(model, open("model.sav", 'wb'))
        accuracy = score

        # test_data['Embarked'] = test_data['Embarked'].replace(np.nan, "0")
        # test_data['Pclass'] = test_data['Pclass'].replace(np.nan, "0")
        # test_data['Sex'] = test_data['Sex'].replace(np.nan, "0")
        # test_data['Fare'] = test_data['Fare'].replace(np.nan, "0")

        # # print("Dropping unique traits: ==>\n")

        # test_data = test_data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)

        # encoder = preprocessing.LabelEncoder()

        # sex = (encoder.fit_transform(np.ravel(test_data[['Sex']])))
        # embarked = preprocessing.scale(encoder.fit_transform(np.ravel(test_data[['Embarked']])))
        # fare = preprocessing.scale(np.ravel(test_data[['Fare']]))
        # test_data['Sex'] = sex
        # test_data['Embarked'] = embarked
        # test_data['Fare'] = fare
        # test_data['Pclass'] = preprocessing.scale(np.ravel(test_data[['Pclass']]))
        # test_data = test_data[['Pclass', 'Sex', 'Fare', 'Embarked']]

        # # print(test_data)

        # submission_prediction = model.predict(test_data)

        # submission = pd.DataFrame({
        #     "PassengerId": list(range(892, 1310)),
        #     "Survived": submission_prediction
        # })

        # print(len(submission))

        # submission.to_csv(r'..\submission.csv', index=False)
