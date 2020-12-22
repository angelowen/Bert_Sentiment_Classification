from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer,RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import time
import datetime
import torch
from sklearn.metrics import matthews_corrcoef
import os,re
import torch.nn.functional as F
from argparse import Action, ArgumentParser, Namespace


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
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

def test(args): 
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')  
    df_ans = pd.read_csv('./data/sample_submission.csv')
    df_test = pd.read_csv(args.test_path)
    headers = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] 
    print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))
    sentences = clean_text(df_test['comment_text'])

    if args.model_name.lower()=="bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # 分词、填充或截断
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = args.max_length,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    batch_size = args.batch_size
    # 准备好数据集
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    if args.model_name.lower()=="bert":
        # 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层 
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # 小写的 12 层预训练模型
            num_labels = args.num_class, # 分类数 --2 表示二分类
                            # 你可以改变这个数字，用于多分类任务  
            output_attentions = False, # 模型是否返回 attentions weights.
            output_hidden_states = False, # 模型是否返回所有隐层状态.
        )
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", # 小写的 12 层预训练模型
            num_labels = args.num_class, # 分类数 --2 表示二分类
                            # 你可以改变这个数字，用于多分类任务  
            output_attentions = False, # 模型是否返回 attentions weights.
            output_hidden_states = False, # 模型是否返回所有隐层状态.
        )
    model = model.to(device)

    for emotion in headers:
        model.load_state_dict(torch.load(f'./{args.save_path}/model-{emotion}.pth'))
        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
        model.eval()
        # Tracking variables 
        predictions,sent_id  = [],[]
        # 预测
        for batch in prediction_dataloader:
            # 将数据加载到 gpu 中
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            # 不需要计算梯度
            with torch.no_grad():
                # 前向传播，获取预测结果
                outputs = model(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

            logits = outputs[0]
            # 将结果加载到 cpu 中
            logits = logits.detach().cpu().numpy()
            # 存储预测结果和 labels
            predictions.append(logits)
        print('    DONE.')

        ###############
        ### 結果評估 ###
        ###############
        # 合并所有 batch 的预测结果
        flat_predictions = np.concatenate(predictions, axis=0)

        if args.prob == True:
            x = torch.Tensor(flat_predictions)
            y = F.softmax(x,dim =1) #对每一行进行softmax
            pred = []
            for idx,item in enumerate(y.numpy()):
                pred.append(item[1])
            df_ans[emotion] = pred
        else:
            flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
            df_ans[emotion] = flat_predictions
    
    today = datetime.datetime.now().strftime("%m-%d-%X")
    df_ans.to_csv(f'{args.save_path}/submission_{today}.csv',index=None)

def test_argument(inhert=False):
    """return test arguments
    Args:
        inhert (bool, optional): return parser for compatiable. Defaults to False.
    Returns:
        parser_args(): if inhert is false, return parser's arguments
        parser(): if inhert is true, then return parser
    """

    # for compatible
    parser = ArgumentParser(add_help=not inhert)

    # doc setting
    parser.add_argument('--doc', type=str,  default='./doc',
                        help='load document file by position ')

    # dataset setting
    parser.add_argument('--max-length', type=int, default=256,
                        help='control the bert max length (default: 256)')
    parser.add_argument('--test-path', type=str, default='./data/test.csv',
                        help='test dataset path (default: ./data/test.csv)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='set the batch size (default: 16)')
    parser.add_argument('--num-class', type=int, default=2,
                        help='set the class number (default: 2)')
    parser.add_argument('--prob', action='store_true', default=False,
                        help='output the probability file (default: False)')
    # model setting
    parser.add_argument('--model-name', type=str, default='Bert',
                        metavar='Bert, Roberta' ,help="set model name (default: 'Bert')")
    # parser.add_argument('--load', action='store_true', default=False,
    #                     help='load model parameter from exist .pt file (default: False)')
    # parser.add_argument('--version', type=int, dest='load',
    #                     help='load specific version (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='set the learning rate (default: 2e-5)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1,
                        help="set weight decay (default: 1)")

    # save setting
    parser.add_argument('--save-path', type=str, default='./output',
                        help='set the output file  path (default: ./output)')
    # for the compatiable
    if inhert is True:
        return parser
    
    return parser.parse_args()



if __name__ == '__main__':
    # argument setting
    test_args = test_argument()
    test(test_args)