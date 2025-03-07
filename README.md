# Chat-Models

## 概要

このプロジェクトはチャットモデルを実装したものです。

## インストール方法

リポジトリをクローンした後、以下のコマンドで必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

## 使用方法

### 訓練

data/conversations.json を任意のサンプルデータに書き換えて以下のコマンドで訓練を開始します。

```bash
python3 train.py
```

### チャット

以下のコマンドで事前に訓練された対話モデルを使用してチャットボットを実行します。

```bash
python3 inference.py
```
