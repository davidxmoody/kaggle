import pandas as pd
import plotly.express as px
import plotly.subplots as sp


# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# %%
num_rows = 3
num_cols = 6

sample = train.sample(num_rows * num_cols).values

fig = sp.make_subplots(
    rows=num_rows, cols=num_cols, subplot_titles=[f'"{r[0]}"' for r in sample]
)

for index, item in enumerate(sample):
    col = index % num_cols + 1
    row = index // num_cols + 1
    pixels = sample[index][1:].reshape((28, 28))
    fig.add_trace(px.imshow(pixels).data[0], row=row, col=col)

fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
fig.update_layout(coloraxis={"colorscale": "gray_r"})

fig.show()


# %%
label_counts = (
    train.label.value_counts().sort_index().reset_index().astype({"label": str})
)
fig = px.bar(label_counts, y="count", x="label")
fig.show()


# %%
train.isnull().values.any() or test.isnull().values.any()
