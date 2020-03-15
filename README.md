# hnishi_da_handson

## development

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


