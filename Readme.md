# Toxic comment classification - BERT 

> [Kaggle競賽連結](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)
>
> [資料下載區](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

Google對話AI團隊研究計劃，目前正在開發有助於改善在線對話的工具。重點研究領域之一是負面網絡行為的研究，例如有害評論（即粗魯，無禮或可能使某人離開討論的評論）。到目前為止，他們已經建立了通過Perspective API服務的一系列公開可用的模型。但是，當前的模型仍然會出錯。
在這場比賽中，您面臨的挑戰是建立一個多頭模型，該模型能夠比Perspective的當前模型更好地檢測各種類型的有害性，例如威脅，淫穢，侮辱和基於身份的仇恨。您將使用Wikipedia對話頁編輯中的評論數據集，其改進有望幫助在線討論變得更加富有成效。


## 執行方法1
```python train.py```

```python test.py```
- **Attention!!** model parameters in test.py should be same as train.py e.g:max_length,num_class,model_name

## 執行方法2
```python preprocess.py```

```python main.py```

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

## 參數說明

- ```-h, --help```
  
  - show this help message and exit

- ```--max-length```

  - control the bert max length (default: 256)

- ```--train-path```

  - training data file path (default: ./data/train.csv)

- ```--test-path```

  - test dataset path (default: ./data/test.csv)

- ```--batch-size```

  - set the batch size (default: 16)

- ```--num-class```

  - 分類類別數 (預設值: 2)

- ```--prob```

  - 產生機率檔案(預設產生分類結果檔)

- ```--model-name```

  - 設定要用Bert or Roberta model (default='Bert')

- ```--gpu-id```
  - set the model to run on which gpu (default: 1)

- ```--lr```
  - set the learning rate (default: 2e-5)

- ```--weight-decay```
  - set weight decay (default: 1)

- ```--epochs```
  - set the epochs (default: 4)

- ```--save-path```
  - set the output file  path (default: ./output)

## 資料架構
- data/
  - train.csv
  - test.csv
  - sample_submission.csv
- logistics_regression.py
- preprocess.py
- main.py
- train.py
- test.py
---

## 解說
* 此次專案使用hugging face/transformers 進行Bert model的fine-tune 以達到文本分類的功能，主要利用BertForSequenceClassification 來完成任務，為了更快達到效果，採用`bert-base-uncased`做為pretrain-model，並將資料進行預處理後，丟入模型中訓練，最終輸出logits經過softmax轉換機率值後寫檔，檔案`Result.csv`
* 可選擇使用Roberta model
* 輸出可為機率值或是分類結果
* logistics_regression.py 使用了sklearn庫中的TfidfVectorizer來提取TF-IDF特徵，並利用Logistic Regression模型來進行分類

## 待優化

> [文本處理](https://cloud.tencent.com/developer/article/1616750)
* Data Augmentation
* Training with Roberta

## Reference
* https://toutiao.io/posts/cq9k9i/preview
* https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
* https://www.cnblogs.com/zingp/p/11696111.html#_label5
* https://zhuanlan.zhihu.com/p/80986272
* https://zhuanlan.zhihu.com/p/33925599
* https://juejin.cn/post/6844904167257931783
