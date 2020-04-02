# データ分析 勉強会 資料

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


## 目次

- [基本的な統計量とガウス分布](da_handson_basic_statistic_values.ipynb)
- [主成分分析](da_handson_pca.ipynb)

## 使い方

各ノートブックの上部の、"Open in Colab" のボタンをクリックすると、
Google Golaboratory 上で開くことができます。
上から順番に実行（Shift + enter）で、python のコードを実行できます。

## 推奨環境

Google Colabratory で動作確認を行った。

オフライン環境の場合、画像が正常に表示されない箇所があります。

## 作業メモ

### notebook に google colab へのリンク（Open in Colab）を貼る

- google colab で作業
- ファイル --> github にコピーを保存
- リンクを含めるにチェックする
- レポジトリとブランチを選択してコピー

### notebook を markdown に変換する

[nbconvert](https://nbconvert.readthedocs.io/en/latest/) と依存ライブラリをインストールする。

以下を実行。

```bash
jupyter nbconvert --to markdown da_handson_basic_statistic_values.ipynb
```

image の path を vim で置換する。

```
:%s;da_handson_pca_\(.*\).png;https://github.com/hnishi/hnishi_da_handson/blob/dev/markdown/da_handson_pca_
files/da_handson_pca_h(\1).png?raw=true;gc
```
