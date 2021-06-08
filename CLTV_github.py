# Customer Lifetime Value Prediction w/ BeteGeoFiiter

# pip install lifetimes
# pip install sqlalchemy
# pip install mysql-connector-python-rf
# pip install mysql
# pip install pymysql

import mysql
import pymysql
import mysql.connector
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Read .csv file:
# Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.shape

##### Data Preparation:
# First look at the data, drop null observations,
# drop negative invoice observations,
# Drop quantity observations that are < 0

df.describe().T
df.dropna(inplace=True)
df.isnull().sum()
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df.head()

##### Accessing and Suppressing Outliers:

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity") # aykırı değerleri çeyrekliklerde olan değerlere baskılama işlemi
replace_with_thresholds(df, "Price")
df.describe().T



###### BGNBD MODEL PREPARATION:

# recency represents the age of the customer when they made their most recent purchases. / weekly
# This is equal to the duration between a customer’s first purchase and their latest purchase.
# T: Customer's age. Weekly. (Calculated upon the date of first transaction of the customer.)
# frequency: Number of repatative transactions (frequency>1)
# monetary_value: Average profit per purchase

# Let's choose a region: UK!
df = df[df["Country"] == "United Kingdom"]
df.head()
df.shape
df["Country"].nunique()

# Data Preparation for BGNBD
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: ((date.max() - date.min()).days) / 7,
                                                         lambda date: ((today_date - date.min()).days) / 7],
                                         "Invoice": lambda freq: freq.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
cltv_df.head()
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary_value"]

# Expressing monetary value as average profit per purchase
cltv_df["monetary_value"] = cltv_df["monetary_value"] / cltv_df["frequency"]

# Choosing monetary values greater than zero
cltv_df = cltv_df[cltv_df["monetary_value"] > 0]

# frequency must be greater than 1.
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]
cltv_df.describe().T
(cltv_df["frequency"]<1).value_counts()

####### BGNBD MODEL:

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

plot_period_transactions(bgf)
plt.show()

# Ex: the 10 customers we expect the most to purchase in a week:

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

cltv_df.head()
cltv_df.sort_values(by="expected_purc_1_week", ascending=False).head(10)


###### GAMMA GAMMA MODEL:

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_value"])

cltv_df["exp_avg_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                    cltv_df["monetary_value"])

cltv_df.sort_values("exp_avg_profit", ascending=False)


###################
#  Finally! Calculation of CLTV with BG-NBD and GG model.
###################

# Ex: For 6 months of time prediction

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary_value"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# Let's see the final table:

cltv_final.sort_values(by="clv", ascending=False).head(10)

###### Segmentation of Customers:

# 1. Scaling for better insight of dataframe:
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

# 2. Segmentation:
cltv_final["Segment"] = pd.qcut(cltv_final["scaled_clv"], 4, ["D", "C", "B", "A"])
cltv_final.head()

cltv_final.groupby("Segment").agg(["count", "mean", "sum"])
