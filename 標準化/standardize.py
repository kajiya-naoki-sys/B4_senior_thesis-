from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import csv

#カレントディレクトリ
path = os.getcwd()
path += "/標準化"

normal_path = path + "/normal.csv"
with open(normal_path) as f:
    csvreader = csv.reader(f)
    n_data = [row for row in csvreader]
    del n_data[0]
    for row in n_data:
        del row[0]

anormal_path = path + "/anormal.csv"
with open(normal_path) as f:
    csvreader = csv.reader(f)
    an_data = [row for row in csvreader]
    del an_data[0]
    for row in an_data:
        del row[0]


def standardize(data):
    scaler = StandardScaler()
    standardize_data = scaler.fit_transform(data)
    return standardize_data

df = pd.DataFrame(n_data)
standardized_normal_ds = pd.DataFrame(standardize(df))
standardized_normal_ds.to_csv(path+'/stadardized_normal_data_set.csv')

df = pd.DataFrame(an_data)
standardized_anormal_ds = pd.DataFrame(standardize(df))
standardized_anormal_ds.to_csv(path+'/standardized_anormal_data_set.csv')