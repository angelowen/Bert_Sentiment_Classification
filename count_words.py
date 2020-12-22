import pandas as pd
import numpy as np
import re,os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('./data/train.csv')

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
def clean_text(df,comment_text):
    comment_list = []
    # print(len(comment_text))
    for idx,text in enumerate(comment_text):
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text)
        
        # 恢复常见的简写
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        
        # 恢复特殊符号的英文单词
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')
        
        comment_list.append(text)
    return comment_list

train["comment_text"] = clean_text(train,train['comment_text'])
# test['comment_text'] = clean_text(test,test['comment_text'])


sentence = train["comment_text"].values
print(len(sentence))
max_len = 0
for item in sentence:
    n = len(item.split()) 
    if n>max_len :
        max_len = n
print(max_len)