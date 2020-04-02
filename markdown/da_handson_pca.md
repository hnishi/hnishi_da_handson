
<a href="https://colab.research.google.com/github/hnishi/hnishi_da_handson/blob/dev/da_handson_pca.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# 主成分分析 （主成分解析、Principal component analysis : PCA）

## 概要

- 主成分分析は、教師なし線形変換法の１つ
  - データセットの座標軸を、データの分散が最大になる方向に変換し、元の次元と同じもしくは、元の次元数より低い新しい特徴部分空間を作成する手法
- 主なタスク
  - 次元削減
- 次元削減を行うことで以下の目的を達成できる
  - 特徴抽出
  - データの可視化
- 次元削減を行うメリット
  - 計算コスト（計算時間、メモリ使用量）を削減できる
  - 特徴量を削減したことによる情報の喪失をできるだけ小さくする
  - モデルを簡素化できる（パラメータが減る）ため、オーバーフィッティングを防げる
  - 人間が理解可能な空間にデータを投影することができる（非常に高次元な空間を、身近な3次元、2次元に落とし込むことができる）

## 応用例

- タンパク質分子の立体構造モデルの構造空間の次元削減と可視化
- タンパク質の全原子モデルの立体構造は、分子内に含まれる原子の座標情報で表すことができる （原子数 × 3 (x, y, z) 次元のベクトル）

以下は、タンパク質の分子シミュレーションで使われるモデルの1例。  
（紫色とオレンジ色で表されたリボンモデルがタンパク質で、周りに水とイオンが表示されている）  
（この場合、3547 個の原子 --> 10641 次元）

<img src="https://github.com/hnishi/hnishi_da_handson/blob/master/images/cdr-h3-pbc.png?raw=true" width="50%">

主成分分析により、この立体構造空間を、2次元空間に投影することができる。  
以下は、その投影に対して自由エネルギーを計算した図。

![pmf](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/peds/29/11/10.1093_protein_gzw028/4/m_gzw028f01.jpeg?Expires=1587240307&Signature=VSLAYMRRIS1aEDvRh~qXlKnqad3NhFo9Sg39TcSQwD3Rn0ikOvue4NsGUKc7V5QQLoEw9qFO256N8gDt9yxtOz8RLYZ5dizEkYat15R3Gr0Jjcs3aOq~kqlevS9Zx4uDqNDN6NfjjkXv2DCt1pdrecxhyIqOTSdRmFXWiAOybVk0QCHNNEhUscvPDbLVsWoM9839Oa9Bb~QaSgaNr~dWu3nNI-8IKy03m45ybWtMZxXjamZjMFR6cxiv5qwynkmpfaBEgjyboRPe8q0otTHvVGJ4yjqAQiD5OpLZwClI5ex3CGTd1CGEKiDFwHDwKYYCgpEF42JtGfkRn1gFt5sEYQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

2次元空間上の1点が、1つの立体構造を表している。  
つまり、この例では、もともと10641次元あった空間を2次元にまで削減している。

Ref) [Nishigami, H., Kamiya, N., & Nakamura, H. (2016). Revisiting antibody modeling assessment for CDR-H3 loop. Protein Engineering, Design and Selection, 29(11), 477-484.](https://academic.oup.com/peds/article/29/11/477/2462452)

## 主成分分析 (PCA) が行う座標変換のイメージ

以下は、PCAが行う座標変換の例

$x_1$ , $x_2$ は、データセットの元々の座標軸であり、  
PC1, PC2 は座標変換後に得られる新しい座標軸、主成分1、主成分2 である (Principal Components)。  



<img src="https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/code/ch05/images/05_01.png?raw=true" width="50%">


- PCA は、高次元データにおいて分散が最大となる方向を見つけ出し、座標を変換する （これはつまり、すべての主成分が、他の主成分と相関がない（直交する) ように座標変換している)
- 最初の主成分 (PC1) の分散が最大となる

## 主成分分析の主要な手順

d 次元のデータを k 次元に削減する場合

1. d 次元のデータの標準化（特徴量間のスケールが異なる場合のみ）
1. 分散共分散行列の作成
1. 分散共分散行列の固有値と固有ベクトルを求める
1. 固有値を降順にソートして、固有ベクトルをランク付けする
1. 最も大きい k 個の固有値に対応する k 個の固有ベクトルを選択 (k ≦ d)
1. k 個の固有ベクトルから射影(変換)行列 W を作成
1. 射影(変換)行列を使って d 次元の入力データセットを新しい k 次元の特徴部分空間を取得する

---

固有値問題を解くことで、線形独立な基底ベクトルを得ることができる。  
詳細は、線形代数の書籍等を参考にする（ここでは詳細な解説をしない）。

参考）  

https://dora.bk.tsukuba.ac.jp/~takeuchi/?%E7%B7%9A%E5%BD%A2%E4%BB%A3%E6%95%B0II%2F%E5%9B%BA%E6%9C%89%E5%80%A4%E5%95%8F%E9%A1%8C%E3%83%BB%E5%9B%BA%E6%9C%89%E7%A9%BA%E9%96%93%E3%83%BB%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AB%E5%88%86%E8%A7%A3

## python による PCA の実行

以下、Python を使った PCA の実行を順番に見ていく。  
その後、scikit-learn ライブラリを使った PCA の簡単で効率のよい実装を見る。  

### データセット

- データセットは、 [Wine](https://archive.ics.uci.edu/ml/datasets/Wine) というオープンソースのデータセットを使う。  
- 178 行のワインサンプルと、それらの化学的性質を表す 13 列の特徴量で構成されている。
- それぞれのサンプルに、クラス 1, 2, 3 のいずれかがラベルされており、  
イタリアの同じ地域で栽培されている異なる品種のブドウを表している   
（PCA は教師なし学習なので、学習時にラベルは使わない）。



```python
from IPython.display import Image
%matplotlib inline
```


```python
import pandas as pd

# df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
#                       'machine-learning-databases/wine/wine.data',
#                       header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

df_wine = pd.read_csv('https://github.com/rasbt/python-machine-learning-book-2nd-edition'
                      '/raw/master/code/ch05/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class label</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>



Wine データセットの先頭 5 行のデータは上記。


```python
for i_label in df_wine['Class label'].unique():
  print('label:', i_label)
  print('shape:', df_wine[df_wine['Class label'] == i_label].shape)
```

    label: 1
    shape: (59, 14)
    label: 2
    shape: (71, 14)
    label: 3
    shape: (48, 14)


ラベルの数はおおよそ揃っている。  
次に、ラベルごとにデータの分布を見てみる。


```python
import numpy as np
import matplotlib.pyplot as plt

for i_feature in df_wine.columns:
  if i_feature == 'Class label': continue
  print('feature: ' + str(i_feature))
  # ヒストグラムの描画
  plt.hist(df_wine[df_wine['Class label'] == 1][i_feature], alpha=0.5, bins=20, label="1") 
  plt.hist(df_wine[df_wine['Class label'] == 2][i_feature], alpha=0.3, bins=20, label="2", color='r')
  plt.hist(df_wine[df_wine['Class label'] == 3][i_feature], alpha=0.1, bins=20, label="3", color='g') 
  plt.legend(loc="upper left", fontsize=13) # 凡例表示
  plt.show()
```

    feature: Alcohol



![png](da_handson_pca_files/da_handson_pca_14_1.png)


    feature: Malic acid



![png](da_handson_pca_files/da_handson_pca_14_3.png)


    feature: Ash



![png](da_handson_pca_files/da_handson_pca_14_5.png)


    feature: Alcalinity of ash



![png](da_handson_pca_files/da_handson_pca_14_7.png)


    feature: Magnesium



![png](da_handson_pca_files/da_handson_pca_14_9.png)


    feature: Total phenols



![png](da_handson_pca_files/da_handson_pca_14_11.png)


    feature: Flavanoids



![png](da_handson_pca_files/da_handson_pca_14_13.png)


    feature: Nonflavanoid phenols



![png](da_handson_pca_files/da_handson_pca_14_15.png)


    feature: Proanthocyanins



![png](da_handson_pca_files/da_handson_pca_14_17.png)


    feature: Color intensity



![png](da_handson_pca_files/da_handson_pca_14_19.png)


    feature: Hue



![png](da_handson_pca_files/da_handson_pca_14_21.png)


    feature: OD280/OD315 of diluted wines



![png](da_handson_pca_files/da_handson_pca_14_23.png)


    feature: Proline



![png](da_handson_pca_files/da_handson_pca_14_25.png)


データを70％のトレーニングと30％のテストサブセットに分割する。


```python
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)
```

データの標準化を行う。


```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train) # トレーニングセットの標準偏差と平均値を使って、標準化を行う
X_test_std = sc.transform(X_test) # "トレーニングセット"の標準偏差と平均値を使って、標準化を行う

# いずれの特徴量も、値がおおよそ、-1 から +1 の範囲にあることを確認する。
print('standardize train', X_train_std[0:2])
print('standardize test', X_test_std[0:2])
```

    standardize train [[ 0.71225893  2.22048673 -0.13025864  0.05962872 -0.50432733 -0.52831584
      -1.24000033  0.84118003 -1.05215112 -0.29218864 -0.20017028 -0.82164144
      -0.62946362]
     [ 0.88229214 -0.70457155  1.17533605 -0.09065504  2.34147876  1.01675879
       0.66299475  1.0887425  -0.49293533  0.13152077  1.33982592  0.54931269
       1.47568796]]
    standardize test [[ 0.89443737 -0.38811788  1.10073064 -0.81201711  1.13201117  1.09807851
       0.71204102  0.18101342  0.06628046  0.51285923  0.79629785  0.44829502
       1.90593792]
     [-1.04879931 -0.77299397  0.54119006 -0.24093881  0.3494145  -0.70721922
      -0.30812129  0.67613838 -1.03520519 -0.90656727  2.24570604 -0.56188171
      -1.22874035]]


---

**注意**

テストデータの標準化の際に、テストデータの標準偏差と平均値を用いてはいけない（トレーニングデータの標準偏差と平均値を用いること）。  
また、ここで求めた標準偏差と平均値は、未知のデータを標準化する際にも再使用するので、記録しておくこと。  
（今回は、ノートブックだけで完結するので、外部ファイル等に記録しなくても問題ない）  

- 分散共分散行列を作成
- 固有値問題を解いて、固有値と固有ベクトルを求める

固有値問題とは、以下の条件を満たす、固有ベクトル $v$ と、スカラー値である固有値 $\lambda$ を求める問題のことである  
（詳細は線形代数の書籍等を参考）。

$$\Sigma v=\lambda v$$

$\Sigma$ は分散共分散行列である（総和記号ではないことに注意）。  
  
分散共分散行列に関しては、 [前回の資料](https://github.com/hnishi/hnishi_da_handson/blob/master/da_handson_basic_statistic_values.ipynb) を参照。



```python
import numpy as np
import seaborn as sns

cov_mat = np.cov(X_train_std.T)

# 共分散行列のヒートマップ
df = pd.DataFrame(cov_mat, index=df_wine.columns[1:], columns=df_wine.columns[1:])
ax = sns.heatmap(df, cmap="YlGnBu") 
```


![png](da_handson_pca_files/da_handson_pca_21_0.png)



```python
# 固有値問題を解く（固有値分解）
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)
print('\nShape of eigen vectors\n', eigen_vecs.shape)
```

    
    Eigenvalues 
    [0.10754642 0.15362835 0.1808613  0.21357215 0.3131368  0.34650377
     0.51828472 0.6620634  0.84166161 0.96120438 1.54845825 2.41602459
     4.84274532]
    
    Shape of eigen vectors
     (13, 13)


**注意**: 

固有値分解（固有分解とも呼ばれる）する numpy の関数は、

- [`numpy.linalg.eig`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html)
- [`numpy.linalg.eigh`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html) 

がある。  
`numpy.linalg.eig` は対称正方行列と非対称正方行列を固有値分解する関数。複素数の固有値を返すことがある。  
`numpy.linalg.eigh` はエルミート行列（各成分が複素数で、転置させた各成分の虚部の値の正負を反転させたものがもとの行列と等しくなる行列）を固有値分解する関数。常に実数の固有値を返す。  

分散共分散行列は、対称正方行列であり、虚数部が 0 のエルミート行列でもある。  
対称正方行列の操作では、`numpy.linalg.eigh` の方が数値的に安定しているらしい。

Ref) *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017.


## 全分散と説明分散（Total and explained variance）

- 固有値の大きさは、データに含まれる情報（分散）の大きさに対応している
- 主成分j (PCj: j-th principal component) に対応する固有値 $\lambda_j$ の分散説明率（寄与率、contribution ratio/propotion とも呼ばれる）は以下のように定義される。

$$\dfrac {\lambda _{j}}{\sum ^{d}_{j=1}\lambda j}$$ 

$\lambda_j$ は、j 番目の固有値、d は全固有値の数（元々の特徴量の数/次元数）。  

分散説明率を見ることで、その主成分が特徴量全体がもつ情報のうち、どれぐらいの情報を表すことができているかを確認できる。  
以下に、分散説明率と、その累積和をプロットする。


```python
# 固有値の合計
tot = sum(eigen_vals)
# 分散説明率の配列を作成
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を作成
cum_var_exp = np.cumsum(var_exp)
```


```python
import matplotlib.pyplot as plt


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()
```


![png](da_handson_pca_files/da_handson_pca_26_0.png)


グラフから以下のことがわかる。

- 最初の主成分だけで、全体の約 4 割の分散を占めている
- 2 つの主成分も用いるだけで、もともとあった特徴量全体の約 6 割を説明できている

## 特徴変換 (Feature transformation)

射影（変換）行列を取得し、適用して特徴変換を行う。

---

$X' = XW$  

$X'$ : 射影（変換）後の座標（行列）  
$X$ : もともとの座標（行列）   
$W$ : 射影（変換）行列  
  
$W$ は、次元削減後の次元数の固有ベクトルから構成される。  

$W = [v_1 v_2 ... v_k] \in \mathbb{R} ^{n\times k}$  


```python
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
```

### まずは、次元削減を行わずに、13 次元 --> 13 次元の座標変換を見てみる

$X' = XW$  

$W = [v_1 v_2 ... v_13] \in \mathbb{R} ^{13\times 13}$  
$x \in \mathbb{R} ^{13}$    
$x' \in \mathbb{R} ^{13}$    


```python
# 変換行列 w の作成
w = eigen_pairs[0][1][:, np.newaxis]
for i in range(1, len(eigen_pairs)):
  # print(i)
  w = np.hstack((w, eigen_pairs[i][1][:, np.newaxis]))
w.shape
```




    (13, 13)




```python
# 座標変換
X_train_pca = X_train_std.dot(w)
# print(X_train_pca.shape)

cov_mat = np.cov(X_train_pca.T)

# 共分散行列のヒートマップ
df = pd.DataFrame(cov_mat)
ax = sns.heatmap(df, cmap="YlGnBu") 
```


![png](da_handson_pca_files/da_handson_pca_32_0.png)


主成分空間に変換後の各特徴量は、互いに相関が全くないことがわかる（互いに線形独立）。  
対角成分は分散値であり、第1主成分から大きい順に並んでいることがわかる。

### 座標変換された空間から元の空間への復元

## 座標変換された空間から元の空間への復元

$X = X'W^T$  

$X'$ : 座標変換後の座標（行列）  
$X$ : もともとの空間に復元された座標（行列）   
$W^T \in \mathbb{R} ^{n\times n}$ : 転置された変）行列  
   
$x' \in \mathbb{R} ^{n}$   
$x_{approx} \in \mathbb{R} ^{n}$   



```python
# 1つ目のサンプルに射影行列を適用（内積を作用させる）
x0 = X_train_std[0]
print('もともとの特徴量:', x0)
z0 = x0.dot(w)
print('変換後の特徴量:', z0)
x0_reconstructed = z0.dot(w.T)
print('復元された特徴量:', x0_reconstructed)
```

    もともとの特徴量: [ 0.71225893  2.22048673 -0.13025864  0.05962872 -0.50432733 -0.52831584
     -1.24000033  0.84118003 -1.05215112 -0.29218864 -0.20017028 -0.82164144
     -0.62946362]
    変換後の特徴量: [ 2.38299011  0.45458499  0.22703207  0.57988399 -0.57994169 -1.73317476
     -0.70180475 -0.21617248  0.23666876  0.16548767 -0.29726982 -0.23489704
      0.40161994]
    復元された特徴量: [ 0.71225893  2.22048673 -0.13025864  0.05962872 -0.50432733 -0.52831584
     -1.24000033  0.84118003 -1.05215112 -0.29218864 -0.20017028 -0.82164144
     -0.62946362]


完全に復元できていることがわかる。

###  13 次元 --> 2 次元に次元削減する

$X' = XW$  

$W = [v_1 v_2] \in \mathbb{R} ^{13\times 2}$  
$x \in \mathbb{R} ^{13}$    
$x' \in \mathbb{R} ^{2}$    


```python
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
```

    Matrix W:
     [[-0.13724218  0.50303478]
     [ 0.24724326  0.16487119]
     [-0.02545159  0.24456476]
     [ 0.20694508 -0.11352904]
     [-0.15436582  0.28974518]
     [-0.39376952  0.05080104]
     [-0.41735106 -0.02287338]
     [ 0.30572896  0.09048885]
     [-0.30668347  0.00835233]
     [ 0.07554066  0.54977581]
     [-0.32613263 -0.20716433]
     [-0.36861022 -0.24902536]
     [-0.29669651  0.38022942]]


**注意**

NumPy と LAPACK のバージョンによっては、上記の例とは符号が反転した射影行列 w が作成されることがあるが、問題はない。  
以下の式が成り立つからである。  

行列 $\Sigma$ に対して、 $v$ が固有ベクトル、$\lambda$ が固有値のとき、  
$$\Sigma v = \lambda v,$$

ここで $-v$ もまた同じ固有値をもつ固有ベクトルとなる。  
$$\Sigma \cdot (-v) = -\Sigma v = -\lambda v = \lambda \cdot (-v).$$

(主成分軸のベクトルの向きの違い）


```python
# 各サンプルに射影行列を適用（内積を作用）させることで、変換後の座標（特徴量）を得ることができる。
X_train_std[0].dot(w)
```




    array([2.38299011, 0.45458499])



### 2次元に射影後の全データを、ラベルごとに色付けしてプロットする


```python
X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()
```


![png](da_handson_pca_files/da_handson_pca_43_0.png)


PC1 軸方向をみると、PC2 軸方向よりもよりもデータが広く分布しており、データをよりよく区別できていることがわかる。

## 次元削減された空間から元の空間への復元

$X_{approx} = X'W^T$  

$X'$ : 射影後の座標（行列）  
$X_{approx}$ : もともとの空間に、近似的に、復元された座標（行列）   
$W^T \in \mathbb{R} ^{n\times k}$ : 転置された射影（変換）行列  
   
$x' \in \mathbb{R} ^{k}$   
$x_{approx} \in \mathbb{R} ^{n}$   
  
$k = n$ のとき、$X = X_{approx}$ が成り立つ（上述）。


```python
# 1つ目のサンプルに射影行列を適用（内積を作用させる）
x0 = X_train_std[0]
print('もともとの特徴量:', x0)
z0 = x0.dot(w)
print('変換後の特徴量:', z0)
x0_reconstructed = z0.dot(w.T)
print('復元された特徴量:', x0_reconstructed)
```

    もともとの特徴量: [ 0.71225893  2.22048673 -0.13025864  0.05962872 -0.50432733 -0.52831584
     -1.24000033  0.84118003 -1.05215112 -0.29218864 -0.20017028 -0.82164144
     -0.62946362]
    変換後の特徴量: [2.38299011 0.45458499]
    復元された特徴量: [-0.09837469  0.66412622  0.05052458  0.44153949 -0.23613841 -0.91525549
     -1.00494135  0.76968396 -0.72702683  0.42993247 -0.87134462 -0.9915977
     -0.53417827]


完全には復元できていないことがわかる（近似値に復元される）。

## Principal component analysis in scikit-learn

上記で行った PCA の実装は、scikit-learn を使うことで簡単に実装できる。
以下にその実装を示す。


```python
from sklearn.decomposition import PCA

pca = PCA()
# 主成分分析の実行
X_train_pca = pca.fit_transform(X_train_std)
# 分散説明率の表示
pca.explained_variance_ratio_
```




    array([0.36951469, 0.18434927, 0.11815159, 0.07334252, 0.06422108,
           0.05051724, 0.03954654, 0.02643918, 0.02389319, 0.01629614,
           0.01380021, 0.01172226, 0.00820609])




```python
# 分散説明率とその累積和のプロット
plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()
```


![png](da_handson_pca_files/da_handson_pca_51_0.png)



```python
# 2次元に削減
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
```


```python
# 2次元空間にプロット
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
```


![png](da_handson_pca_files/da_handson_pca_53_0.png)


## 2次元に次元削減された特徴量を用いてロジスティック回帰を行ってみる



```python
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
```

Training logistic regression classifier using the first 2 principal components.


```python
from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression(penalty='l2', C=1.0)
# lr = LogisticRegression(penalty='none')
lr = lr.fit(X_train_pca, y_train)
```


```python
print(X_train_pca.shape)
```

    (124, 2)



```python
print('Cumulative explained variance ratio:', sum(pca.explained_variance_ratio_))
```

    Cumulative explained variance ratio: 0.5538639565949177


### 学習時間の計測


```python
%timeit lr.fit(X_train_pca, y_train)
```

    100 loops, best of 3: 5.11 ms per loop



```python
from sklearn.metrics import plot_confusion_matrix

# 精度
print('accuracy', lr.score(X_train_pca, y_train))
# confusion matrix
plot_confusion_matrix(lr, X_train_pca, y_train)
```

    accuracy 0.9838709677419355





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f54fca16da0>




![png](da_handson_pca_files/da_handson_pca_62_2.png)


### トレーニングデータセットの予測結果


```python
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.



![png](da_handson_pca_files/da_handson_pca_64_1.png)


### テストデータに対する予測結果


```python
from sklearn.metrics import plot_confusion_matrix

# 精度
print('accuracy', lr.score(X_test_pca, y_test))
# confusion matrix
plot_confusion_matrix(lr, X_test_pca, y_test)
```

    accuracy 0.9259259259259259





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f54f53ec160>




![png](da_handson_pca_files/da_handson_pca_66_2.png)



```python
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_05.png', dpi=300)
plt.show()
```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.



![png](da_handson_pca_files/da_handson_pca_67_1.png)


次元削減せずに全てのの主成分を取得したい場合は、 `n_components=None` にする。


```python
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
```




    array([0.36951469, 0.18434927, 0.11815159, 0.07334252, 0.06422108,
           0.05051724, 0.03954654, 0.02643918, 0.02389319, 0.01629614,
           0.01380021, 0.01172226, 0.00820609])



## 3 次元に次元削減された特徴量を用いてロジスティック回帰を行ってみる


```python
from sklearn.linear_model import LogisticRegression

k = 3

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression(penalty='l2', C=1.0)
# lr = LogisticRegression(penalty='none')
lr = lr.fit(X_train_pca, y_train)
```


```python
print(X_train_pca.shape)
```

    (124, 3)



```python
print('Cumulative explained variance ratio:', sum(pca.explained_variance_ratio_))
```

    Cumulative explained variance ratio: 0.6720155475408875



```python
%timeit lr.fit(X_train_pca, y_train)
```

    100 loops, best of 3: 5.76 ms per loop



```python
from sklearn.metrics import plot_confusion_matrix

# 精度
print('accuracy', lr.score(X_train_pca, y_train))
# confusion matrix
plot_confusion_matrix(lr, X_train_pca, y_train)
```

    accuracy 0.9838709677419355





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f54f606fcf8>




![png](da_handson_pca_files/da_handson_pca_75_2.png)



```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X_train_pca[:,0], X_train_pca[:,1], X_train_pca[:,2], c='r', marker='o')
ax.scatter(X_train_pca[:,0], X_train_pca[:,1], X_train_pca[:,2], c=y_train, marker='o')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()
```


![png](da_handson_pca_files/da_handson_pca_76_0.png)



```python
# plotly を使った interactive な 3D 散布図
import plotly.express as px

df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'])
df['label'] = y_train

fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
              color='label', opacity=0.7, )
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="50877fa6-75d2-4516-8ee8-799aa48be9ad" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("50877fa6-75d2-4516-8ee8-799aa48be9ad")) {
                    Plotly.newPlot(
                        '50877fa6-75d2-4516-8ee8-799aa48be9ad',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "PC1=%{x}<br>PC2=%{y}<br>PC3=%{z}<br>label=%{marker.color}", "legendgroup": "", "marker": {"color": [3, 1, 1, 1, 3, 2, 2, 3, 2, 2, 2, 1, 2, 3, 1, 3, 2, 1, 3, 3, 2, 1, 2, 2, 2, 2, 3, 1, 2, 2, 1, 1, 3, 1, 2, 1, 1, 2, 3, 3, 1, 3, 3, 3, 1, 2, 3, 3, 2, 3, 2, 2, 2, 1, 2, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 3, 1, 2, 3, 2, 2, 3, 1, 2, 1, 2, 2, 3, 2, 1, 1, 1, 3, 2, 1, 1, 2, 2, 3, 3, 2, 1, 1, 2, 2, 3, 1, 3, 1, 2, 2, 2, 2, 1, 3, 1, 1, 1, 1, 2, 2, 3, 3, 2, 2], "coloraxis": "coloraxis", "opacity": 0.7, "symbol": "circle"}, "mode": "markers", "name": "", "scene": "scene", "showlegend": false, "type": "scatter3d", "x": [2.38299010628165, -1.9657818342761368, -2.5390759826731437, -1.4301077634823764, 3.1414722743374535, 0.5025355174304724, 0.048677216444456356, 2.478889890837675, 2.0190025895476813, 0.7515658263299464, 0.7226891496983433, -3.0036621050580106, 2.575188784771916, 3.7315110360943025, -1.1227651833811303, 2.8599685344062253, -0.7471712452249434, -1.5842787774162708, 3.3888710052679927, 3.154054730027625, -1.2803650563636255, -1.7143891145760053, -1.5504029140397453, 1.1098448898248685, -0.6910841834221154, -2.0860359990382626, 2.9039345555832674, -2.0763578431192635, -1.7475618546062412, 2.594244559712473, -2.5037235506940494, -2.1944840194590016, 3.9163453413579057, -1.1173961847439984, -0.8999680418605585, -1.7146917780649822, -2.4858130250697856, -0.7608056164374378, 2.9265370962579427, 2.9442371639540426, -2.389932189360117, 2.638850494578335, 2.51009031453245, 3.6524808642815887, -2.6516960860534327, 0.5254455878346656, 2.7019757342128936, 3.184147081997617, 1.1251704080946257, 2.923665189936381, -1.9612231430392655, 0.5447367346888976, -0.7703030782492855, -1.1667045525555042, -1.3647530932690284, 0.4356373211295947, 2.961917452749134, 2.8360955735919395, 1.9040208928654767, -2.485839098390368, -2.165755675041433, 0.006697760954698862, 1.065601807991443, 2.1311791064704395, 1.5354348318688578, -2.6678311196413826, 0.5727999846207674, -0.7071091585977504, -0.9960657676095674, -2.6732415348111105, -2.3636737797286114, -0.39171874937152845, -2.989088448563118, -1.9182253898038528, 2.3114458032411545, -1.0605050296925769, -2.7485860866279204, 2.266500773662799, -1.1551746905921498, 0.16602502953654413, 1.3558938936752154, -3.311850567401112, -0.33245686428865434, -2.2320508545598754, 0.1858375793819492, 0.8456085631947511, 2.695004718284889, 0.4464567413734273, -1.8896100733724928, -3.081317605096027, -3.4571634848499873, 3.8766562871337222, 1.575515996805267, -3.4334437068095607, -4.206425966320472, -0.14042970769668142, 1.8273152077183488, 2.2056474409250013, 1.6499905365619116, -1.461103297067776, -0.60047516329304, -3.082762313096913, 0.450357491433118, 0.9080689663896474, 3.2497363711740253, -3.0788205461527727, 2.5427730604375243, -2.8483815700907678, -0.889972711099617, 0.32368249266219995, 0.32007526662162156, 0.4488918764416006, -2.4658255817715466, 2.8167811296155114, -2.1698302538907335, -2.667282287971753, -3.532239235540976, -1.9663768792211023, 1.6874121615416129, 0.4352107672298763, 2.590451154057883, 4.353083972897038, -1.8431537322857185, -0.4086095466800211], "y": [0.45458499212572817, 1.653769389946968, 1.0290906609492259, 0.6024010988851962, 0.6621497927952746, -2.089071314900294, -2.275360439959899, -0.08603318154093348, -1.3538718966099923, -2.5536794688487303, -1.1840439069993787, 0.9462693400710848, -1.0697548993837769, 1.0196887612755863, 0.13877000455001104, 2.2881955863337526, -3.217460605442277, 0.16048054563872385, 2.1155068896594753, 0.5423396606121389, -1.7292687100189417, 0.717452488216606, -1.758059099309678, -1.2048069313885958, -1.7138537379166594, -1.6845367082243832, 1.9525880482264442, 1.4718330358089013, -1.2584254568353685, -0.10560370337047677, 0.7041221157350614, 2.186575516704052, 0.16136475261615515, 0.5192108611268781, -2.047595749541602, 0.6139216924698934, 0.7683956144482913, -1.6761562740805074, 0.1885474069864281, 1.3481238835106775, 1.0848073988606313, 0.7527493656088564, 2.25237952563597, 1.7483992539309818, 1.0199747570259248, -2.1352824927289515, 0.5647630703165921, 2.580946953456133, -1.8505444858696203, 0.41699914819304196, -1.286136607215191, -1.0789722623216003, -1.9338681459723375, 0.004898153278469697, -2.135722694347315, -2.569296070852136, 1.9109100891890225, 0.6538603161616434, -0.3529654196842034, -0.21308835163899584, 1.1468486030611238, -0.9433762379360541, 3.312210247467154, 1.9055130448224509, -1.5085497903961367, 1.7593359879252402, -2.7511382987206985, -2.437985493112964, -1.4772410970859913, 1.3577960886192333, 1.6653792681790593, 0.13747499082853856, 2.169831648464864, 1.6014180917551382, 0.2071229959000705, 0.6004608026867289, -0.29016054005909436, 2.144917577966878, -0.5026290907909564, -2.2685005136961114, 0.33353006517394007, 1.392401154379454, -2.156398650686876, 0.528681428028275, -1.444469666607523, 0.17151684309953996, 2.7452249204468324, -0.6239394338901929, -0.04400723172587037, 1.5972442904902546, 1.2142844243212103, 0.4644600353261582, -1.8229983929101163, 1.6116813964798882, 2.2014536641495304, -2.3687163884609372, -1.3948510277267125, 1.2846206606350752, 2.33211133751735, -0.4648032418097786, 0.009200722248501145, 0.28287147804942203, -2.2026375497519135, -2.088168596057688, -0.18273485467899997, 0.6962262068790516, 1.8857165233678352, 0.6327432521212526, -0.6792722626770021, -2.070061748331748, -2.8870851865821625, -2.148725319680299, 1.074557704273905, 0.5634444415365472, 0.16644198505497165, 1.381377024885905, 2.579060288556329, 1.1831918462821642, -1.3507532086518426, -2.4035581702000854, 1.6385292107546472, 0.6653604055959265, -1.506884150845844, -1.2972060696048262], "z": [-0.2270320677349445, 1.3870926828172951, 1.3255184108244742, 0.5553081171464955, -1.0839324559915855, 1.0409565268777894, -1.043136940468716, -1.2381264084106791, -0.46023961239804284, -1.2442904559566033, -0.0006901709449677448, -0.2886848499539379, -2.114663579181573, -1.1621097950380643, -0.049870947628221834, -0.9409884465852398, -0.9025752942872188, -0.362234984390805, -0.11166595315479734, -1.00148347834704, -0.6782172297299867, -1.1666537965536592, 1.2934197980610458, 2.7685813865986235, 0.72456435628768, -2.6629996000922254, -0.2129967432803451, 0.39670547324227967, 1.715368236746086, 0.23779325588115957, -0.3220419555799858, 1.0184722695099773, 0.2781225838352601, 3.8609222853979954, 0.7872247411571505, -0.09271058996769377, 0.7411087442122195, 0.19616402901118468, 0.02784295032844946, -1.043155169786938, -1.8675884082660665, -0.7265531325606531, -0.6469831655115019, -0.05803122949672302, -0.6857228767678358, 0.838523433846456, -0.8059273250060421, 1.3034770964031852, 0.6335805269535196, 1.1848078786175544, -1.5555128822639182, 0.05617795710660953, -0.5646222046595545, -1.4347649943539271, -0.04635943946032398, 1.990415819259947, -0.6965329166378816, -1.194710330561304, -1.6583042331304751, -0.45242036270388886, 0.30165345090158413, -1.8573548786896341, 1.2702379911598245, 1.0369582400103496, 3.0381148745451902, -0.6606357543398687, 0.4104824511629599, -0.42896009421930675, -0.7178693695222867, -0.612563942487306, 0.4101514355306667, -0.35134368510197816, -0.5493324843743828, 0.8908670287841625, 1.3664242905206578, 2.0239610407823343, 3.3930654789408026, 1.851977902767043, -1.5774139741349285, 0.011883882067901573, -0.9631600022805533, -1.5300218751602022, -0.8651992490155089, 1.3674735481769849, 0.7897928315179916, -0.5613143331253624, -0.6217247339202082, 1.3958062070978248, 0.3199063641477553, -0.24126476283168863, -0.5104173332955815, 1.7330900016773214, 0.41727633077125453, -0.5342364738017759, -1.4373869228383276, -1.4161921536716089, 0.48771348003442977, 0.15647768648544366, 0.581246893460613, 5.164589987839916, -0.9120626727150363, -2.39782355864901, 0.6889949847866246, -1.9141895112102991, -0.7834381724555941, -0.45256558407285535, 0.0010826094164175504, -0.11503852354097888, 0.26738151323259524, -0.22885869746337908, -0.7770952487881233, -0.9367556757341098, 0.47059168411902397, 0.9086122280900272, -1.2563517390889176, -0.283764586021869, -0.3343118958009132, -0.17103925648107232, -0.0333375406320053, 0.9917386515706548, 0.39576068878727394, -1.3173735763612588, 0.2463139455120486, 1.403156024649178]}],
                        {"coloraxis": {"colorbar": {"title": {"text": "label"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "scene": {"domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "xaxis": {"title": {"text": "PC1"}}, "yaxis": {"title": {"text": "PC2"}}, "zaxis": {"title": {"text": "PC3"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('50877fa6-75d2-4516-8ee8-799aa48be9ad');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                
            </script>
        </div>
</body>
</html>


人間の目で確認できるのは 3 次元が限界。

## 次元削減せずにロジスティック回帰を行ってみる




```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', C=1.0)
# lr = LogisticRegression(penalty='none')
lr = lr.fit(X_train_std, y_train)
```


```python
# 学習時間
%timeit lr.fit(X_train_std, y_train)
```

    100 loops, best of 3: 7.23 ms per loop



```python
from sklearn.metrics import plot_confusion_matrix

print('Evaluation of training dataset')
# 精度
print('accuracy', lr.score(X_train_std, y_train))
# confusion matrix
plot_confusion_matrix(lr, X_train_std, y_train)

print('Evaluation of test dataset')
# 精度
print('accuracy', lr.score(X_test_std, y_test))
# confusion matrix
plot_confusion_matrix(lr, X_test_std, y_test)
```

    Evaluation of training dataset
    accuracy 1.0
    Evaluation of test dataset
    accuracy 1.0





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f54f56a2c50>




![png](da_handson_pca_files/da_handson_pca_82_2.png)



![png](da_handson_pca_files/da_handson_pca_82_3.png)


元々の全ての特徴量を使って学習させた方が精度が高くなった。  
学習時間は、次元削減したほうがわずかに早くなっている。
（主成分 2 つで学習した場合 4.9 ms に対し、元々の特徴量全て使った場合 5.64 ms）  
結論として、今回のタスクでは、PCA を適用するべきではなく、すべての特徴量を使用したほうが良い。  
  
もっとデータ数が大きい場合や、モデルのパラメータ数が多い場合には、次元削減が効果的となる。

### 2つの特徴量だけでロジスティック回帰を行ってみる




```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', C=1.0)
# lr = LogisticRegression(penalty='none')
lr = lr.fit(X_train_std[:,:2], y_train)
```


```python
%timeit lr.fit(X_train_std[:,:2], y_train)
```

    100 loops, best of 3: 3.53 ms per loop



```python
from sklearn.metrics import plot_confusion_matrix

print('Evaluation of training dataset')
# 精度
print('accuracy', lr.score(X_train_std[:,:2], y_train))
# confusion matrix
plot_confusion_matrix(lr, X_train_std[:,:2], y_train)

print('Evaluation of test dataset')
# 精度
print('accuracy', lr.score(X_test_std[:,:2], y_test))
# confusion matrix
plot_confusion_matrix(lr, X_test_std[:,:2], y_test)
```

    Evaluation of training dataset
    accuracy 0.7580645161290323
    Evaluation of test dataset
    accuracy 0.7777777777777778





    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f54f560de10>




![png](da_handson_pca_files/da_handson_pca_87_2.png)



![png](da_handson_pca_files/da_handson_pca_87_3.png)


もともとの特徴量を 2 つだけ使った場合、精度はかなり下がる。
これと比べると、PCA によって特徴抽出した 2 つの主成分を使った場合には、精度がかなり高いことがわかる。

## まとめ

主成分分析により以下のタスクを行うことができる。

- 次元削減
  - データを格納するためのメモリやディスク使用量を削減できる
  - 学習アルゴリズムを高速化できる
- 可視化
  - 多数の特徴量（次元）をもつデータを2次元などの理解しやすい空間に落とし込んで議論、解釈することができる。
  
しかし、機械学習の前処理として利用する場合には、以下のことに注意する必要がある。

- 次元削減を行うことによって、多少なりとも情報が失われている
- まずは、すべての特徴量を使ってトレーニングを試すことが大事  
- 次元削減によってオーバーフィッティングを防ぐことができるが、次元削減を使う前に正則化を使うべし
- 上記を試してから、それでも望む結果を得られない場合、次元削減を使う
- 機械学習のトレーニングでは、通常は、99% の累積寄与率が得られるように削減後の次元数を選ぶことが多い  

参考) [Andrew Ng先生の講義](https://www.coursera.org/learn/machine-learning)

## References

1. *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017. Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
1.  [Andrew Ng先生の講義](https://www.coursera.org/learn/machine-learning)
