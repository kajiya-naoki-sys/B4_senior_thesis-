from sklearn.preprocessing import Normalizer
import pandas as pd
import os
import csv

#カレントディレクトリ
path = os.getcwd()
path = path + '/正規化'

normal_path = path + "/normal.csv"
with open(normal_path) as f:
    csvreader = csv.reader(f)
    n_data = [row for row in csvreader]
    del n_data[0]
    for row in n_data:
        del row[0]
n_data = [[float(num) for num in row] for row in n_data]

anormal_path = path + "/anormal.csv"
with open(normal_path) as f:
    csvreader = csv.reader(f)
    an_data = [row for row in csvreader]
    del an_data[0]
    for row in an_data:
        del row[0]
an_data = [[float(num) for num in row] for row in an_data]

def normalizer(data):
    scaler = Normalizer(norm = 'l2')
    normalizer_data = scaler.fit_transform(data)
    return normalizer_data

df=pd.DataFrame(n_data)
normalizered_normal_ds = pd.DataFrame(normalizer(df))
normalizered_normal_ds.to_csv(path+'/normalizered_normal_data_set.csv')

df = pd.DataFrame(an_data)
normalizered_anormal_ds = pd.DataFrame(normalizer(df))
normalizered_anormal_ds.to_csv(path+'/normalizered_anormal_data_set.csv')