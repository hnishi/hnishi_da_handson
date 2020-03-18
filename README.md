# データ分析 勉強会 資料

## 目次

- [基本的な統計量とガウス分布](da_handson_basic_statistic_values.ipynb)
- [主成分分析](da_handson_pca.ipynb)

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


