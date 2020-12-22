from transformers import BertForSequenceClassification, AdamW, BertConfig

# 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # 小写的 12 层预训练模型
    num_labels = 2, # 分类数 --2 表示二分类
                    # 你可以改变这个数字，用于多分类任务  
    output_attentions = False, # 模型是否返回 attentions weights.
    output_hidden_states = False, # 模型是否返回所有隐层状态.
)

# 在 gpu 中运行该模型
model.cuda()
# 将所有模型参数转换为一个列表
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
