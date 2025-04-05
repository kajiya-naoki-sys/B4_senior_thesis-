import matplotlib.pyplot as plt
import os
import csv

#カレントディレクトリのパスを取得
file_path = os.path.abspath('')

#各データセットをリストに代入
normal_path = file_path + '/散布図化/normal_data_set.csv'
with open(normal_path) as f:
    csvreader = csv.reader(f)
    n_data = [row for row in csvreader]
    del n_data[0]
    for row in n_data:
        del row[0]
anormal_path = file_path +'/散布図化/anormal_data_set.csv'
with open(anormal_path) as f:
    csvreader = csv.reader(f)
    an_data = [row for row in csvreader]
    del an_data[0]
    for row in an_data:
        del row[0]

#各要素をfloat型にキャスト
nX = [float(row[0]) for row in n_data]
nY = [float(row[1]) for row in n_data]
aX = [float(row[0]) for row in an_data]
aY = [float(row[1]) for row in an_data]

#maplotlib
fig, ax = plt.subplots(1,1, figsize = (10, 10))
plt.xlim(-30, 30)
plt.ylim(-30, 30)
ax.set_xlabel("x", size=10)
ax.set_ylabel("y", size=10)

#散布図
plt.scatter(nX, nY, c = "b", alpha = 0.2)
plt.scatter(aX, aY, c = "r", alpha = 0.2)
plt.legend(("normal", "anormal"), loc="lower right", fontsize = 20)

plt.show()