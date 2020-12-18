from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import time
import datetime
import torch
from sklearn.metrics import matthews_corrcoef

CUDA_DEVICES = 1
device = torch.device(f'cuda:{CUDA_DEVICES}' if torch.cuda.is_available() else 'cpu')

# 根据预测结果和标签数据来计算准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_test(exp,df_ans):
    
    print(f'Now training for {exp} data!!!')
    # 加载数据集到 pandas 的 dataframe 中
    df = pd.read_csv(f"./data/{exp}/train.tsv", delimiter='\t', header=None, names=['id', 'label', 'nouse', 'sentence'])

    # 打印数据集的记录数
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    sentences = df.sentence.values
    labels = df.label.values
    # 加载 BERT 分词器
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    # 将数据集分完词后存储到列表中
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # 输入文本
                            add_special_tokens = True, # 添加 '[CLS]' 和 '[SEP]'
                            max_length = 128,           # 填充 & 截断长度
                            pad_to_max_length = True,
                            return_attention_mask = True,   # 返回 attn. masks.
                            return_tensors = 'pt',     # 返回 pytorch tensors 格式的数据
                    )
        
        # 将编码后的文本加入到列表  
        input_ids.append(encoded_dict['input_ids'])
        
        # 将文本的 attention mask 也加入到 attention_masks 列表
        attention_masks.append(encoded_dict['attention_mask'])

    # 将列表转换为 tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # 输出第 1 行文本的原始和编码后的信息
    #print('Original: ', sentences[0])
    #print('Token IDs:', input_ids[0])


    # 将输入数据合并为 TensorDataset 对象
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # 计算训练集和验证集大小
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # 按照数据大小随机拆分训练集和测试集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))


    # 在 fine-tune 的训练中，BERT 作者建议小批量大小设为 16 或 32
    batch_size = 32

    # 为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = DataLoader(
                train_dataset,  # 训练样本
                sampler = RandomSampler(train_dataset), # 随机小批量
                batch_size = batch_size # 以小批量进行训练
            )

    # 验证集不需要随机化，这里顺序读取就好
    validation_dataloader = DataLoader(
                val_dataset, # 验证样本
                sampler = SequentialSampler(val_dataset), # 顺序选取小批量
                batch_size = batch_size 
            )

    # 加载 BertForSequenceClassification, 预训练 BERT 模型 + 顶层的线性分类层 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # 小写的 12 层预训练模型
        num_labels = 2, # 分类数 --2 表示二分类
                        # 你可以改变这个数字，用于多分类任务  
        output_attentions = False, # 模型是否返回 attentions weights.
        output_hidden_states = False, # 模型是否返回所有隐层状态.
    )
    # if device == 'cpu':
    model = model.to(device)
    # else:
    #     model = model.cuda(1)
        
    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                    )


    # 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合 
    epochs = 4

    # 总的训练样本数
    total_steps = len(train_dataloader) * epochs

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

        
    #######################
    ##### 訓練分類模型 #####
    #######################

    # 设定随机种子值，以确保输出是确定的
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # 存储训练和评估的 loss、准确率、训练时长等统计指标, 
    training_stats = []

    # 统计整个训练时长
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 统计单次 epoch 的训练时间
        t0 = time.time()

        # 重置每次 epoch 的训练总 loss
        total_train_loss = 0

        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):
            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()        
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            # 累加 loss
            total_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

        # 平均训练误差
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        avg_acc=1-avg_train_loss
        print("  Training epcoh took: {:}".format(training_time))
        # torch.save(model.state_dict(), f'model-{avg_acc:.02f}-acc.pth')    
        # ========================================
        #               Validation
        # ========================================
        # 完成一次 epoch 训练后，就对该模型的性能进行验证

        print("")
        print("Running Validation...")

        t0 = time.time()

        # 设置模型为评估模式
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # 将输入数据加载到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            # 累加 loss
            total_eval_loss += loss.item()

            # 将预测结果和 labels 加载到 cpu 中计算
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 计算准确率
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # 打印本次 epoch 的准确率
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # 统计本次 epoch 的 loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # 统计本次评估的时长
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # 记录本次 epoch 的所有统计信息
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


    ##########################

    # 保留 2 位小数
    pd.set_option('precision', 2)
    # 加载训练统计到 DataFrame 中
    df_stats = pd.DataFrame(data=training_stats)
    # 使用 epoch 值作为每行的索引
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

    ##########################

    # 加载数据集
    df = pd.read_csv(f"./data/{exp}/dev.tsv", delimiter='\t', header=None, names=['id', 'label', 'nouse', 'sentence'])

    # 打印数据集大小
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    # 将数据集转换为列表
    sentences = df.sentence.values
    labels = df.label.values

    # 分词、填充或截断
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 128,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    batch_size = 32  

    # 准备好数据集
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # 预测测试集

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    # 依然是评估模式
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # 预测
    acc = 0
    for batch in prediction_dataloader:
        # 将数据加载到 gpu 中
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
    
        # 不需要计算梯度
        with torch.no_grad():
            # 前向传播，获取预测结果
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # 将结果加载到 cpu 中
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # 存储预测结果和 labels
        predictions.append(logits)
        true_labels.append(label_ids)
        
        acc += flat_accuracy(logits, label_ids)
                
        # 打印本次 epoch 的准确率
        avg_dev_accuracy = acc / len(prediction_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_dev_accuracy))
        print('    DONE.')

    ###############
    ### 結果評估 ###
    ###############
    # 合并所有 batch 的预测结果
    flat_predictions = np.concatenate(predictions, axis=0)
    # 取每个样本的最大值作为预测值
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # 合并所有的 labels
    flat_true_labels = np.concatenate(true_labels, axis=0)
    # 计算 MCC
    # mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    # print('Total MCC: %.3f' % mcc)
    ##################################################
    df = pd.read_csv(f"./data/{exp}/test.tsv", delimiter='\t', header=0, names=['id', 'sentence'])

    # 打印数据集大小
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    # 将数据集转换为列表
    sentences = df.sentence.values
    test_id = df.id.values

    # 分词、填充或截断
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 128,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # test_id = torch.from_numpy(test_id)

    batch_size = 32  

    # 准备好数据集
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # 预测测试集

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    # 依然是评估模式
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
        #   label_ids = b_labels.to('cpu').numpy()
        
        # 存储预测结果和 labels
        predictions.append(logits)
        #   sent_id.append(label_ids)

    print('    DONE.')

    ###############
    ### 結果評估 ###
    ###############
    # 合并所有 batch 的预测结果
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # 合并所有的 序列id編號
    # flat_id = np.concatenate(sent_id, axis=0)

    print(flat_predictions)
    # output = np.vstack([flat_id,flat_predictions]).T
    # print(output)
    # df = pd.DataFrame(output, columns = ['id','ans'])
    # df.to_csv('Result.csv')

    df_ans[exp] = flat_predictions
    print(df_ans)
    df_ans.to_csv(f'Result_{exp}.csv',index=None)



if __name__ == '__main__':
    head = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']
    # head = ['id','small']
    df_test = pd.read_csv(f"./data/toxic/test.tsv", delimiter='\t', header=0, names=['id', 'sentence'])
    test_id = df_test.id.values
    df_ans = pd.DataFrame(index=None, columns=head)
    df_ans['id'] = test_id
    for idx,exp in enumerate(head):
        if idx!=0:
            train_test(exp,df_ans)

    df_ans.to_csv('Result.csv',index=None)


# https://juejin.cn/post/6844904167257931783