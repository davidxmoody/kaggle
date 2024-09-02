import pandas as pd
import plotly.express as px
import plotly.subplots as sp


# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

Y_train = train.label
X_train = train.drop("label", axis=1)
del train

X_train = X_train / 255
test = test / 255


# %%
Y_train.isnull().values.any() or X_train.isnull().values.any() or test.isnull().values.any()


# %%
label_counts = Y_train.value_counts().sort_index().reset_index().astype({"label": str})

fig = px.bar(label_counts, y="count", x="label")
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
    index = Y_train.loc[Y_train == d].sample(1).index
    pixels = X_train.loc[index].values.reshape((28, 28))
    fig.add_trace(px.imshow(pixels).data[0], row=row, col=col)

fig.update_yaxes(autorange="reversed")
fig.update_layout(coloraxis={"colorscale": "gray_r"})

fig.show()
