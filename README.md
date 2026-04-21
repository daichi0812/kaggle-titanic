# Kaggle Titanic - Machine Learning from Disaster

Kaggle の入門コンペ [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) に取り組んだリポジトリ。
乗客の属性（性別・年齢・チケットクラスなど）から生存を予測する二値分類タスク。

## ディレクトリ構成

```
.
├── README.md
├── baseline.ipynb      # EDA・前処理・モデル構築・提出までの一連の流れ
├── data/
│   ├── train.csv       # 学習データ (891件)
│   ├── test.csv        # テストデータ (418件)
│   └── gender_submission.csv  # 提出フォーマットのサンプル
└── submission.csv      # 最終提出ファイル
```

## データ

| カラム | 説明 |
|--------|------|
| PassengerId | 乗客ID |
| Survived | 生存フラグ (0=死亡, 1=生存) |
| Pclass | チケットクラス (1=上級, 2=中級, 3=下級) |
| Name | 名前 |
| Sex | 性別 |
| Age | 年齢 |
| SibSp | 同乗した兄弟・配偶者の数 |
| Parch | 同乗した親・子供の数 |
| Ticket | チケット番号 |
| Fare | 運賃 |
| Cabin | 客室番号 |
| Embarked | 乗船港 (C=Cherbourg, Q=Queenstown, S=Southampton) |

## 手法

### 前処理

- **欠損値補完**: Age は Sex × Pclass グループの中央値、Fare は Pclass グループの中央値で補完
- **Cabin**: 欠損が多いため、客室の有無 (`HasCabin`) に変換
- **Embarked**: 欠損2件は削除せずワンホットエンコーディングで処理

### 特徴量エンジニアリング

| 特徴量 | 説明 |
|--------|------|
| Title | 名前から敬称 (Mr, Mrs, Miss, Master, Rare) を抽出 |
| FamilySize | SibSp + Parch |
| IsAlone | 家族がいない場合 1 |
| FarePerPerson | Fare / (FamilySize + 1) |
| HasCabin | Cabin 情報がある場合 1 |

カテゴリ変数 (Title, Embarked) はワンホットエンコーディング、Sex は 0/1 にマッピング。

### モデル

- **Random Forest** (GridSearchCV でハイパーパラメータチューニング)
- **Logistic Regression**
- **MLP (多層パーセプトロン)** — hidden_layer_sizes=(100, 100, 10)
- **アンサンブル** — 上記3モデルの予測確率の平均

## 結果

| モデル | Train Accuracy | Valid Accuracy |
|--------|---------------|---------------|
| Random Forest | 0.865 | 0.828 |
| Logistic Regression | 0.839 | 0.832 |
| MLP | 0.835 | 0.810 |
| アンサンブル | — | 0.817 |

GridSearchCV による Random Forest の最適パラメータ: `max_depth=5, min_samples_leaf=2` (CV Score: 0.836)

最終提出にはアンサンブル（3モデルの確率平均）を使用。

## 使い方

### 環境構築

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 実行

1. `data/` に Kaggle からダウンロードした `train.csv`, `test.csv`, `gender_submission.csv` を配置
2. `baseline.ipynb` を Jupyter Notebook で開いて上から順に実行
3. `submission.csv` が生成される
