# Toxic comment classification - BERT 

> [Kaggle競賽連結](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)
>
> [資料下載區](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

Google對話AI團隊研究計劃，目前正在開發有助於改善在線對話的工具。重點研究領域之一是負面網絡行為的研究，例如有害評論（即粗魯，無禮或可能使某人離開討論的評論）。到目前為止，他們已經建立了通過Perspective API服務的一系列公開可用的模型。但是，當前的模型仍然會出錯。
在這場比賽中，您面臨的挑戰是建立一個多頭模型，該模型能夠比Perspective的當前模型更好地檢測各種類型的有害性，例如威脅，淫穢，侮辱和基於身份的仇恨。您將使用Wikipedia對話頁編輯中的評論數據集，其改進有望幫助在線討論變得更加富有成效。


## 執行方式
```python preprocess.py```
```python pytorch_toxic.py```

### 資料前處理

```python preprocess.py```

## 軟體需求

- Python 3
- 以下為 Python 須安裝之套件
  - numpy
  - pandas
  - torch
  - random
  - transformers
  - time
  - datetime
  - sklearn
---

## 資料架構
- data/
  - train.csv
  - test.csv
- preprocess.py
- pytorch_toxic.py
---

## 解說
此次專案使用hugging face/transformers 進行Bert model的fine-tune 以達到文本分類的功能，為了更快達到效果，採用`bert-base-uncased`做為pretrain-model，並將資料進行預處理後，轉換成tsv檔儲存，再丟入模型中訓練，最終輸出檔案`Result.csv`

## 待優化
* 檔案整合
* 直接讀檔用pandas抓出資料
