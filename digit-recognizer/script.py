import keras
from keras.layers import Dense, Conv2D, Flatten
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np


# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# %%
train.isnull().values.any() or test.isnull().values.any()


# %%
label_counts = train.label.value_counts().sort_index().reset_index()

fig = px.bar(label_counts.astype({"label": str}), y="count", x="label")
fig.show()


# %%
num_rows = 2
num_cols = 5

fig = sp.make_subplots(
    rows=num_rows, cols=num_cols, subplot_titles=[f'"{d}"' for d in range(10)]
)

for d in range(10):
    col = d % num_cols + 1
    row = d // num_cols + 1
    sampled = train.loc[train.loc[train.label == d].sample(1).index]
    pixels = sampled.values[0][1:].reshape(28, 28)
    fig.add_trace(px.imshow(pixels).data[0], row=row, col=col)

fig.update_yaxes(autorange="reversed")
fig.update_layout(coloraxis={"colorscale": "gray_r"})

fig.show()


# %%
Y_train = keras.utils.to_categorical(train.label)
X_train = train.drop("label", axis=1)

X_train = X_train / 255
X_test = test / 255

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)


# %%
model = keras.Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        activation="relu",
        input_shape=(28, 28, 1),
    )
)

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
    )
)

model.add(Flatten())

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"])


# %%
model.fit(X_train, Y_train, epochs=5, batch_size=100, validation_data=(X_val, Y_val))


# %%
Y_val_pred = model.predict(X_val)

confusion_matrix(np.argmax(Y_val_pred, axis=1), np.argmax(Y_val, axis=1))


# %%
test_pred = np.argmax(model.predict(X_test), axis=1)

submission = pd.DataFrame(
    {
        "ImageId": range(1, len(test) + 1),
        "Label": test_pred,
    }
)

submission.to_csv("data/submission.csv", index=False)
