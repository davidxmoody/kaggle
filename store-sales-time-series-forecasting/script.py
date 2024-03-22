import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

oil = oil.interpolate(method="linear", limit_direction="both")

# %%

holiday_dates = holidays.loc[
    (holidays.transferred == False) & (holidays.locale == "National")
].date

# %%

train["days_since_start"] = (train.date - train.date[0]).dt.days
train["year"] = train.date.dt.year
train["month"] = train.date.dt.month
train["day"] = train.date.dt.day
train["dayofweek"] = train.date.dt.dayofweek
train = train.merge(oil, left_on="date", right_index=True, how="left")
train = train.merge(
    stores.add_prefix("store_"), left_on="store_nbr", right_index=True, how="left"
)
train["holiday"] = train.date.isin(holiday_dates.values)

train_dummies = pd.get_dummies(
    train,
    columns=[
        "family",
        "dayofweek",
        "store_cluster",
        "store_type",
    ],
).drop(
    columns=["date", "store_nbr", "year", "month", "day", "store_city", "store_state"]
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
# train_selector = train_subset.date <= split_date

tmp_train = train_dummies.iloc[:-1654]
tmp_test = train_dummies.iloc[-1654:]

y_train = tmp_train.sales
y_test = tmp_test.sales

X_train = tmp_train.drop(columns=["sales"])
X_test = tmp_test.drop(columns=["sales"])

# %%

model = RandomForestRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(mse)
