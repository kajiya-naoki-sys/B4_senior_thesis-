import os
import numpy as np
from numpy.random import *
import pandas as pd
import matplotlib.pyplot as plt

#カレントディレクトリのパスを取得
file_path = os.path.abspath('')

#シード値設定
np.random.seed(42)

#パラメータ設定
mu1 = np.array([-7, -7])
sigma1 = np.array([[10, 4], [4, 10]])
mu2 = np.array([10, 10])
sigma2 = np.array([[11, 5], [5, 11]])

# (x1, x2)格子点を作成
x = y = np.arange(-30, 30, 0.5)
X, Y = np.meshgrid(x, y)
z = np.c_[X.ravel(), Y.ravel()]

#2次元正規分布の確立密度を返す(等高線用)
def gaussian(x, mu, sigma):
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    #print(det)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    #print(inv)
    Z = np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    return Z

#2次元正規分布を生成
#正常データ(9900)
values1 = multivariate_normal(mu1, sigma1, 9900)
X1 = [row[0] for row in values1]
Y1 = [row[1] for row in values1]
Z1 = gaussian(z, mu1, sigma1)
shape = X.shape
Z1 = Z1.reshape(shape)
#異常データ(100)
values2 = multivariate_normal(mu2, sigma2, 100)
X2 = [row[0] for row in values2]
Y2 = [row[1] for row in values2]
Z2 = gaussian(z, mu2, sigma2)
Z2 = Z2.reshape(shape)
#ファイル作成
df = pd.DataFrame(values1)
df.to_csv(file_path+"/normal_data_set.csv")
df = pd.DataFrame(values2)
df.to_csv(file_path+"/anormal_data_set.csv")

#matplotlib
fig, ax = plt.subplots(1,1, figsize = (10, 10))
plt.xlim(-30, 30)
plt.ylim(-30, 30)
ax.set_xlabel("x", size=10)
ax.set_ylabel("y", size=10)
#等高線
plt.contour(X, Y, Z1, cmap = "bone_r", linewidths = 3)
plt.contour(X, Y, Z2, cmap = "gist_heat_r", linewidths = 3)

#散布図
plt.scatter(X1, Y1, c = "b", alpha = 0.2)
plt.scatter(X2, Y2, c = "r", alpha = 0.2)
plt.legend(("normal", "anormal"), loc="lower right", fontsize = 20)

plt.show()