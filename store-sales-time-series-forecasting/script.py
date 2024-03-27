from datetime import timedelta
import pandas as pd
from keras import layers, models
import seaborn as sns
import matplotlib.pyplot as plt

# %%

data_dir = "store-sales-time-series-forecasting/data"

holidays = pd.read_csv(data_dir + "/holidays_events.csv", parse_dates=["date"])

oil = pd.read_csv(data_dir + "/oil.csv", index_col="date", parse_dates=["date"])

stores = pd.read_csv(data_dir + "/stores.csv", index_col="store_nbr")

test = pd.read_csv(data_dir + "/test.csv", index_col="id", parse_dates=["date"])

train = pd.read_csv(data_dir + "/train.csv", index_col="id", parse_dates=["date"])

transactions = pd.read_csv(
    data_dir + "/transactions.csv",
    index_col=["date", "store_nbr"],
    parse_dates=["date"],
)

# %%

oil = oil.rename(columns={"dcoilwtico": "oil"})
oil = oil.reindex(pd.date_range(train.date.min(), train.date.max()))
oil = oil.interpolate(method="linear", limit_direction="both")

# %%

holiday_dates = holidays.loc[
    (holidays.transferred == False) & (holidays.locale == "National")
].date

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

# %%

train_subset = train.query("family == 'DAIRY'")

train_dummies = pd.get_dummies(
    train_subset,
    columns=[
        # "family",
        "dayofweek",
        # "month",
        # "store_cluster",
        # "store_type",
    ],
).drop(
    columns=[
        "family",
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

# Potential features:
# - date (days since start, year, month, day, day of week)
# - store_nbr
# - family
# - recent sales (in family and store)
# - onpromotion
# - oil price
# - holidays (type, transferred, matches locale)
# - transactions?

# Things to try:
# - scale onpromotion (based on historical maxes within category?)
# - make day of week categorical (same for months?)
# - make store_nbr categorical

# %%

# train_subset = train_dummies.loc[(train_dummies.store_nbr == 1) & (train_dummies.family == "DAIRY")]

# split_date = train.iloc[-1, 0] - timedelta(days=30)

test_selector = train_subset.date >= (train_subset.date.iloc[-1] - timedelta(days=15))

tmp_train = train_dummies.loc[~test_selector]
tmp_test = train_dummies.loc[test_selector]

y_train = tmp_train.sales
y_test = tmp_test.sales

X_train = tmp_train.drop(columns=["sales"]).astype(float)
X_test = tmp_test.drop(columns=["sales"]).astype(float)

# %%

num_features = len(X_train.columns)

model = models.Sequential(
    [
        layers.Dense(32, activation="relu", input_shape=(num_features,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=10, batch_size=32)

loss = model.evaluate(X_test, y_test)
print(loss)

# %%

y_pred = model.predict(X_test)
results = pd.DataFrame({"y_pred": y_pred[:,0], "y_test": y_test})
results.plot.bar()

# %%

sns.scatterplot(train_subset, y="sales", x="oil", hue="family")
ax = plt.gca()
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
