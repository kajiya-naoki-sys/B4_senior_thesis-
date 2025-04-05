import random
import csv
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

#カレントディレクトリのパスを取得
file_path = os.getcwd()

#シード値固定
random.seed(42)

# 交差検証の結果を格納するリスト
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

#csvからデータを抽出
n_ds_f = file_path + '/DS/normal_data_set.csv'
with open(n_ds_f) as f:
    csvreader = csv.reader(f)
    n_data = [row for row in csvreader]
    del n_data[0]
    for row in n_data:
        del row[0]
#各要素をflow型でキャスト
n_data = [[float(i) for i in row] for row in n_data]
#ラベルを追加(1)
[row.append(1) for row in n_data]


an_ds_f = file_path + '/DS/anormal_data_set.csv'
with open(an_ds_f) as f:
    csvreader = csv.reader(f)
    an_data = [row for row in csvreader]
    del an_data[0]
    for row in an_data:
        del row[0]
#各要素をflow型でキャスト
an_data = [[float(i) for i in row] for row in an_data]
#ラベルを追加(-1)
[row.append(-1) for row in an_data]

#データ結合
data = np.vstack((n_data, an_data))
#list型にキャスト
data_list = data.tolist()
#シャッフルする
data = random.sample(data_list, len(data_list))
#ラベルを抽出
label = []
for row in data:
    label.append(int(row[-1]))
    del row[-1]

#NumPy.ndarrayにキャスト
data = np.array(data)
label = np.array(label)

#k分割交差検証の設定
kf = KFold(n_splits=10, random_state=42, shuffle=True)
#OCSVMの設定
#10,000のデータのうち，100が異常データなのでnu = 0.01
clf = OneClassSVM(nu=0.01, kernel='poly', gamma='scale', degree=2)

#結果を収納するdfを作成
df = pd.DataFrame(columns=["精度", "適合率", "再現率", "F1スコア"])
index = 0

for train, test in kf.split(data):
    index += 1
    #正常データを訓練用とテスト用に振り分ける
    train_data, test_data = data[train], data[test]
    label_test = label[test]
    #訓練
    clf.fit(train_data)
    #テスト
    predict = clf.predict(test_data)
    
    # 精度の計算
    accuracy = accuracy_score(label_test, predict)
    precision = precision_score(label_test, predict, pos_label=1)
    recall = recall_score(label_test, predict, pos_label=1)
    f1 = f1_score(label_test, predict, pos_label=1)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    df.loc[index] = [accuracy, precision, recall, f1]
    print(f'Test 精度: {accuracy:.6f}, 適合率: {precision:.6f}, 再現率: {recall:.6f}, F1スコア: {f1:.6f}')

# k分割交差検証の平均評価指標
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
df.loc["平均"] = [average_accuracy, average_precision, average_recall, average_f1]

#全データポイントに対して、正しく分類されたポイントの割合
print(f'平均精度: {average_accuracy:.6f}')
#正常と予測された中で本当に正常だった割合
print(f'平均適合率: {average_precision:.6f}')
#全ての正常データの中でどれだけ正しく正常と予測できたか
print(f'平均再現率: {average_recall:.6f}')
#適合率と再現率の調和平均を示す．適合率と再現率のバランスを取るための指標
print(f'平均F1スコア: {average_f1:.6f}')

df.to_csv(file_path + "/result.csv")