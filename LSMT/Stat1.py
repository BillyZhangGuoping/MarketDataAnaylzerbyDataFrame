# encoding: UTF-8
import pandas as pd

df = pd.read_csv("C:\\Project\\rb1905.csv")
df["datetime"] = pd.to_datetime(df['datetime'])
print(df)