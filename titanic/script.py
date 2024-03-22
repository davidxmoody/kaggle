# %% Imports

import keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


# %% Load data

test_df = pd.read_csv("titanic/data/test.csv")
train_df = pd.read_csv("titanic/data/train.csv")

val_mask = np.random.rand(len(train_df)) > 0.8
val_df = train_df[val_mask]
train_df = train_df[~val_mask]

combined_df = pd.concat([train_df, test_df, val_df]).drop(["Survived"], axis=1)


# %% Process data

cabin_sizes = combined_df.Cabin.value_counts()

for df in (train_df, test_df, val_df):
    df["CabinSize"] = df.Cabin.map(cabin_sizes)
    df["Surname"] = df.Name.str.split(",").str.get(0)
    df["SibSpSurvived"] = df.apply(
        lambda row: (
            train_df.query(
                "SibSp > 0 and Surname == @row.Surname and PassengerId != @row.PassengerId"
            ).Survived.sum()
            > 0
            if row.SibSp > 0
            else False
        ),
        axis=1,
    )
    df["ParchSurvived"] = df.apply(
        lambda row: (
            train_df.query(
                "Parch > 0 and Surname == @row.Surname and PassengerId != @row.PassengerId"
            ).Survived.sum()
            > 0
            if row.Parch > 0
            else False
        ),
        axis=1,
    )


# %% Analyse

sns.pairplot(
    train_df.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1), hue="Survived"
)


# %% Random forest

y = train_df["Survived"]

features = [
    "Pclass",
    "Sex",
    "SibSp",
    "Parch",
    "Age",
    "Fare",
    "Embarked",
    # "SibSpSurvived",
    # "ParchSurvived",
    # "CabinSize",
]

X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

model = RandomForestClassifier(n_estimators=20, max_depth=5)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_df.PassengerId, "Survived": predictions})
output.to_csv("titanic/data/submission.csv", index=False)
