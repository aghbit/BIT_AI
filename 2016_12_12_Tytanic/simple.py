import pandas as pd
from pandas import Series,DataFrame

titanic_df = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_df    = test_df.drop(['Name','Ticket','Cabin'], axis=1)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)


person_dummies_titanic  = pd.get_dummies(titanic_df['Sex'])
person_dummies_titanic.columns = ['Female','Male']

person_dummies_test  = pd.get_dummies(test_df['Sex'])
person_dummies_test.columns = ['Female','Male']

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)


X_train = titanic_df.drop("Survived",axis=1)[:600]
Y_train = titanic_df["Survived"][:600]

X_test = titanic_df.drop("Survived",axis=1)[600:]
Y_test = titanic_df["Survived"][600:]


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
print("{0:.2f}%".format(logreg.score(X_test, Y_test)*100))
