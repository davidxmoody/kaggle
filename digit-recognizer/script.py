import pandas as pd
import plotly.express as px


# %%
train = pd.read_csv("data/train.csv")


# %%
# Display an image of one random digit from train dataset
random_digit = train.sample(1).values[0]
label = random_digit[0]
pixels = random_digit[1:].reshape((28, 28))

fig = px.imshow(
    pixels, color_continuous_scale="gray_r", title=f"Digit with label '{label}'"
)
fig.show()
