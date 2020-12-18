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
test = pd.read_csv('./data/test.csv')
# sample_submission = pd.read_csv('./sample_submission.csv')

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
test['comment_text'] = clean_text(test,test['comment_text'])
train_org, valid_org= train_test_split(train, test_size=0.1, random_state=42)
# x_test = test_vec
# ===


for label in labels:
    if not os.path.isdir(os.path.join('data',label)):
        os.mkdir(os.path.join('data',label))
    test.to_csv(os.path.join('data',label,'test.tsv'), index=False,sep='\t')
    # train = train.loc[:20000,["id",label,"comment_text"]]
    train = train_org.reindex(columns = ["id",label,"comment_text"])
    valid = valid_org.reindex(columns = ["id",label,"comment_text"])
    l=['a'] * train.shape[0]
    train = pd.DataFrame(train,columns=["id",label,"nouse","comment_text"]) 
    train["nouse"]= l
    l=['a'] * valid.shape[0]
    valid = pd.DataFrame(valid,columns=["id",label,"nouse","comment_text"]) 
    valid["nouse"]= l  
    print(train)
    train.to_csv(os.path.join('data',label,'train.tsv'), index=False,sep='\t', header=False)# train.iloc[:60000,:]
    valid.to_csv(os.path.join('data',label,'dev.tsv'), index=False,sep='\t', header=False)

