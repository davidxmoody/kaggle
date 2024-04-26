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

import pandas as pd
import plotly.express as px
from plotly_calplot import calplot


# %%
train = pd.read_csv("data/train.csv", index_col="id", parse_dates=["date"])
test = pd.read_csv("data/test.csv", index_col="id", parse_dates=["date"])

min_date = train.date.min()
max_date = test.date.max()
date_range = pd.date_range(min_date, max_date)

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


# %% [markdown]
# ## Oil data

# %%
oil = pd.read_csv("data/oil.csv", index_col="date", parse_dates=["date"])

oil = oil.rename(columns={"dcoilwtico": "oil"})

oil = oil.reindex(date_range)
oil = oil.interpolate(method="linear", limit_direction="both")

oil.tail(3)


# %%
fig = px.line(oil, y="oil")
fig.update_yaxes(range=[0, None])
fig.update_xaxes(title="date")
fig.update_layout(title="Oil prices")
fig.show()


# %%
fig = px.scatter(
    train.groupby(["family", "date"])
    .sales.sum()
    .reset_index()
    .merge(oil, left_on="date", right_index=True, how="left"),
    x="oil",
    y="sales",
    color="family",
    trendline="ols",
)
fig.update_yaxes(range=[0, None])
fig.update_layout(title="Sales vs oil")
fig.show()


# %% [markdown]
# ## Holiday data


# %%
holidays = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])

holidays.tail(3)


# %% [markdown]
# To simplify this, just use a single boolean for if a national holiday was present (after accounting for transfers).
#
# Note that the month-long block starting April 2016 was due to the earthquake.


# %%
holiday_dates = holidays.loc[
    (holidays.transferred == False) & (holidays.locale == "National")
].date

fig = calplot(
    pd.DataFrame({"holiday": 1, "date": holiday_dates}).query("date >= '2013'"),
    x="date",
    y="holiday",
    cmap_min=0,
    cmap_max=1.5,
    years_title=True,
)
fig.show()


# %% [markdown]
# ## Store data

# %%
stores = pd.read_csv("data/stores.csv", index_col="store_nbr")

stores.tail(3)


# %% [markdown]
# ## Transaction data
#
# - There are very consistent peaks around Christmas
# - The total number of transactions slowly rises over time
# - Some stores have zero transactions for brief periods (e.g. store_nbr 24)
# - Some stores only start having data mid-way through the period (e.g. store_nbr 52)
# - No transaction data is given for the test data period

# %%
transactions = pd.read_csv("data/transactions.csv", parse_dates=["date"])

monthly_transactions = (
    transactions.groupby("store_nbr")
    .resample("MS", on="date")
    .transactions.sum()
    .reset_index()
)

fig = px.line(monthly_transactions, x="date", y="transactions", color="store_nbr")
fig.update_yaxes(range=[0, None])
fig.update_xaxes(range=[min_date, max_date])
fig.show()


# %% [markdown]
# ## Features

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
