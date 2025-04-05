from sklearn.preprocessing import RobustScaler
import pandas as pd
import os
import csv

#カレントディレクトリ
path = os.getcwd()
path = path + '/ロバストスケーリング'

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

def robust_scaler(data):
    scaler = RobustScaler()
    robust_scaler_data = scaler.fit_transform(data)
    return robust_scaler_data

df=pd.DataFrame(n_data)
robust_scalerd_normal_ds = pd.DataFrame(robust_scaler(df))
robust_scalerd_normal_ds.to_csv(path+'/robust_scalered_normal_data_set.csv')

df = pd.DataFrame(an_data)
robust_scalerd_anormal_ds = pd.DataFrame(robust_scaler(df))
robust_scalerd_anormal_ds.to_csv(path+'/robust_scalered_anormal_data_set.csv')