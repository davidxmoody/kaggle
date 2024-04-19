# %% [markdown]
# ---
# title: "Kaggle Store Sales Forecasting"
# format:
#   html:
#     toc: true
#     code-fold: true
# jupyter: python3
# highlight-style: github
# ---


# %% [markdown]
# ## Imports and data


# %%
from datetime import timedelta

# from keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px


# %%
test = pd.read_csv("data/test.csv", index_col="id", parse_dates=["date"])
train = pd.read_csv("data/train.csv", index_col="id", parse_dates=["date"])


# %%
train.tail(3)


# %% [markdown]
# ## Sales by family
#
# Most product families seem mostly stable. Here are some points of interest:
#
# - `PRODUCE`, `BEVERAGES` and a few other families have multiple significant drops to near zero before June 2015. This looks like it could be an error in the data or possibly some significant region-wide event.
# - `SCHOOL AND OFFICE SUPPLIES` shows various spikes, likely around school term times.
# - `BOOKS` is at zero until around Oct 2016 when it spikes up. It's likely that it wasn't counted as a separate product family until then. `MAGAZINES` also jumps up around Oct 2015 so maybe that was counted in a different way before that.
# - `FROZEN FOODS`, `GROCERY I`, `LIQUOR,WINE,BEER` and some other families show spikes around Christmas.

# %%
monthly_sales = (
    train.groupby("family").resample("MS", on="date").sales.sum().reset_index()
)

fig = px.line(monthly_sales, x="date", y="sales", color="family")
fig.update_layout(title="Monthly sales by family")
fig.update_yaxes(range=[0, None])
fig.show()


# %%
produce_sales = (
    train.loc[train.family == "PRODUCE"]
    .groupby("store_nbr")
    .resample("W", on="date")[["sales"]]
    .sum()
    .reset_index()
)

fig = px.line(produce_sales, x="date", y="sales", color="store_nbr")
fig.update_layout(title="Produce sales by store")
fig.update_yaxes(range=[0, None])
fig.show()


# %%
# grocery_by_store = train.query("family == 'GROCERY I'").groupby("store_nbr").resample("MS", on="date").sales.sum().reset_index()
# sns.lineplot(grocery_by_store, x="date", y="sales", hue="family")
# plt.show()


# %% [markdown]
# ## Oil data

# %%
oil = pd.read_csv("data/oil.csv", index_col="date", parse_dates=["date"])

oil = oil.rename(columns={"dcoilwtico": "oil"})

oil = oil.reindex(pd.date_range(train.date.min(), train.date.max()))
oil = oil.interpolate(method="linear", limit_direction="both")


# %%
fig = px.line(oil, y="oil")
fig.update_yaxes(range=[0, None])
fig.update_xaxes(title="date")
fig.update_layout(title="Oil prices")
fig.show()


# %%
monthly_sales = (
    train.groupby("family").resample("MS", on="date").sales.mean().reset_index()
)
monthly_sales = monthly_sales.merge(
    oil.resample("MS").mean(), left_on="date", right_index=True, how="left"
)

plt.figure(figsize=(9, 3))
sns.scatterplot(monthly_sales, x="oil", y="sales", hue="family")
plt.title("Sales vs oil")
plt.legend().set_visible(False)
plt.show()


## Holiday data

# %%
holidays = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])

holidays.tail(3)


# To simplify this, extract a list of dates for national holidays after transfers have occurred.

# %%
holiday_dates = holidays.loc[
    (holidays.transferred == False) & (holidays.locale == "National")
].date

holiday_dates.tail(3)


## Store data

# %%
stores = pd.read_csv("data/stores.csv", index_col="store_nbr")

stores.tail()


## Transaction data

# %%
transactions = pd.read_csv(
    "data/transactions.csv",
    index_col=["date", "store_nbr"],
    parse_dates=["date"],
)


## Features

# %%
train["days_since_start"] = (train.date - train.date[0]).dt.days
# train["year"] = train.date.dt.year
train["month"] = train.date.dt.month
# train["day"] = train.date.dt.day
train["dayofweek"] = train.date.dt.dayofweek
train = train.merge(oil, left_on="date", right_index=True, how="left")
train = train.merge(
    stores.add_prefix("store_"), left_on="store_nbr", right_index=True, how="left"
)
train["holiday"] = train.date.isin(holiday_dates.values)

train["mean_sales"] = train.groupby(["store_nbr", "family"]).sales.transform("mean")

train["recent_sales"] = train.groupby(["store_nbr", "family"]).sales.transform(
    lambda x: x.rolling(window=30).mean().shift(15)
)


# %%
train_subset = train.dropna()

train_dummies = pd.get_dummies(
    train_subset,
    columns=[
        "family",
        "dayofweek",
        # "month",
        # "store_cluster",
        # "store_type",
    ],
).drop(
    columns=[
        # "family",
        "month",
        "date",
        "store_nbr",
        "store_city",
        "store_state",
        "store_cluster",
        "store_type",
    ]
)


# %%
test_selector = train_subset.date >= (train_subset.date.iloc[-1] - timedelta(days=15))

tmp_train = train_dummies.loc[~test_selector]
tmp_test = train_dummies.loc[test_selector]

y_train = tmp_train.sales
y_test = tmp_test.sales

X_train = tmp_train.drop(columns=["sales"]).astype(float)
X_test = tmp_test.drop(columns=["sales"]).astype(float)


# %%
# num_features = len(X_train.columns)
#
# model = models.Sequential(
#     [
#         layers.Dense(50, activation="relu", input_shape=(num_features,)),
#         layers.Dense(20, activation="relu"),
#         layers.Dense(1),
#     ]
# )
#
# model.compile(optimizer="adam", loss="mse")
#
# model.fit(X_train, y_train, epochs=10, batch_size=1000)
#
# print()
# model.evaluate(X_test, y_test)


# %%
# y_pred = model.predict(X_test)
# results = pd.DataFrame({"y_pred": y_pred[:, 0], "y_test": y_test})
# results.iloc[-60:].plot.bar()
