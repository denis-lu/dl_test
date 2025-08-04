# encoding=utf-8
import os
import sys
import json

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import pandas as pd
import logger
import glo
from dl_models import *
from utils import *
import re
from sklearn.utils.class_weight import compute_class_weight
from imbens.sampler._under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def save_model(epoch, model, training_stats, info, model_name):
    base_dir = './modelcard_'+ info + '_'+ model_name + '/epoch_' + str(epoch) + '/'
    out_dir = base_dir + 'model.ckpt'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print('Saving model to %s' % out_dir)
    torch.save(model.state_dict(), out_dir)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(base_dir + "training_stats.json")


def dl_container(modelcard_data, train_y, test_modelcard_data, test_y, logger, info, fold_num, model_type):
    config = Config()
    if model_type == "bi-lstm":
        model_name = "bi-lstm"
        model = bilstm_model(config).to(config.device)
    elif model_type == "textcnn":
        model_name = "textcnn"
        model = cnn_model(config).to(config.device)
    elif model_type == "transformer":
        model_name = "transformer"
        model = bert_trans_model(config).to(config.device)
    
    # 只在CUDA可用且有多个GPU时使用DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    else:
        print(f"使用 {config.device} 进行训练")

    logger.info("============begin %s training......" % model_name)
    train_dt = Data_processor(modelcard_data, train_y, config.batch_size)
    traindata_loader = train_dt.processed_dataloader
    test_dt = Data_processor(test_modelcard_data, test_y, config.batch_size)
    testdata_loader = test_dt.processed_dataloader
    print("model created.")
    epochs = config.num_epochs
    # 降低学习率以提高训练稳定性
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    total_steps = len(traindata_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # # 计算类别权重来处理不平衡问题
    # # 计算类别权重，但限制最大权重避免过度补偿
    # classes = np.unique(train_y)
    # class_weights = compute_class_weight('balanced', classes=classes, y=train_y)
    
    # # 限制正类权重最大为3.0，避免过度补偿
    # max_weight = 3.0
    # if class_weights[1] > max_weight:
    #     ratio = max_weight / class_weights[1]
    #     class_weights = class_weights * ratio
    
    # class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.device)
    
    # print(f"类别权重: {class_weights}")
    print(f"正样本数量: {np.sum(np.array(train_y) == 1)}")
    print(f"负样本数量: {np.sum(np.array(train_y) == 0)}")

    training_stats = []
    total_t0 = time.time()
    model.eval()
    loss_fn = F.cross_entropy

    # # 选择损失函数 (可以在这里切换)
    # use_focal_loss = True  # 设置为False使用加权交叉熵，设置为True使用Focal Loss
    
    # if use_focal_loss:
    #     # 使用Focal Loss (对困难样本给予更多关注)，降低gamma值提高稳定性
    #     loss_fn = FocalLoss(alpha=1, gamma=1, weight=class_weights)
    #     print("使用Focal Loss")
    # else:
    #     # 使用加权交叉熵损失函数
    #     loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    #     print("使用加权交叉熵损失函数")
    
    best_epoch_acc = 0.0
    # 添加预测监控
    epochs_without_positive_prediction = 0
    max_epochs_without_positive = 3  # 连续3个epoch不预测正类就调整
    
    print("Begin training...")
    progress_bar = tqdm(range(total_steps))
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        epcho_train_loss = 0
        model.train()
        for step, batch in enumerate(traindata_loader):
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(traindata_loader), elapsed))

            input_ids = batch[0].to(config.device)
            input_mask = batch[1].to(config.device)
            batch_labels = batch[2].to(config.device)

            model.zero_grad()
            modelcard_input = (input_ids, input_mask)
            batch_outputs = model(modelcard_input)
            loss = loss_fn(batch_outputs, batch_labels)
            epcho_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        avg_train_loss = epcho_train_loss / len(traindata_loader)
        training_time = format_time(time.time() - t0)
        print("")
        print("====== Average training loss: {0:.2f}".format(avg_train_loss))
        print("====== Training epcoh took: {:}".format(training_time))
        print("Running Testing....")
        t0 = time.time()
        model.eval()
        prob_result_lst = []
        all_pre_label = []
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in testdata_loader:
            with torch.no_grad():
                input_ids = batch[0].to(config.device)
                input_mask = batch[1].to(config.device)
                batch_labels = batch[2].to(config.device)
                modelcard_input = (input_ids, input_mask)
                b_outputs = model(modelcard_input)
            loss = loss_fn(b_outputs, batch_labels)
            total_eval_loss += loss.item()
            lr = torch.nn.Softmax(dim=1)
            preds_prob = lr(b_outputs.data)
            preds_prob = preds_prob.cpu().detach().numpy()
            prob_result_lst.append(preds_prob)
            preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
            labels = batch_labels.to('cpu').numpy()
            all_pre_label.extend(preds.flatten())
            total_eval_accuracy += flat_accuracy(preds, labels)
        prob_result = np.vstack(prob_result_lst)
        print("prob_result:", type(prob_result), prob_result.shape)
        
        # 打印预测情况统计
        pred_counts = np.bincount(all_pre_label)
        test_counts = np.bincount(test_y)
        print(f"测试集标签分布: 负类={test_counts[0]}, 正类={test_counts[1] if len(test_counts) > 1 else 0}")
        print(f"预测结果分布: 负类={pred_counts[0]}, 正类={pred_counts[1] if len(pred_counts) > 1 else 0}")
        
        # # 监控正类预测情况
        # positive_predictions = pred_counts[1] if len(pred_counts) > 1 else 0
        # if positive_predictions == 0:
        #     epochs_without_positive_prediction += 1
        #     print(f"⚠️  连续{epochs_without_positive_prediction}个epoch未预测正类")
            
        #     # 如果连续多个epoch不预测正类，动态调整权重
        #     if epochs_without_positive_prediction >= max_epochs_without_positive:
        #         print("🔧 动态增加正类权重")
        #         class_weights[1] = class_weights[1] * 1.5  # 增加50%权重
        #         if use_focal_loss:
        #             loss_fn = FocalLoss(alpha=1, gamma=1, weight=class_weights)
        #         else:
        #             loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        #         epochs_without_positive_prediction = 0  # 重置计数器
        # else:
        #     epochs_without_positive_prediction = 0  # 重置计数器
        
        avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
        print("  Accuracy: {0:.3f}".format(avg_val_accuracy))
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(testdata_loader)
        test_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.3f}".format(avg_val_loss))
        print("  Validation took: {:}".format(test_time))
        training_stats.append(
            {'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': test_time
            })
        # evulates each epohc, save the metrics with the best F1 score instead of accuracy.
        metrics_ = cal_metrics(test_y, all_pre_label)
        glo_key = model_name + '_' + info + '_' + str(fold_num)
        
        # 使用加权F1分数作为评估指标，而不是准确率
        current_weighted_f1 = metrics_[6]  # weighted_f1
        if current_weighted_f1 >= best_epoch_acc:
            best_epoch_acc = current_weighted_f1
            glo.set_val(glo_key, metrics_)
            save_model(epoch_i + 1, model, training_stats, info, model_name)
            print(f"保存模型！当前最佳加权F1: {best_epoch_acc:.4f}")
    logger.info("============all epoches of %s trained finished......" % model_name)
    return


def cal_last_metrics(logger, model_name, info):
    # (acc_res, precision_res, recall_res, f1_res, weighted_precision, weighted_recall, weighted_f1)
    res = np.array([0.0] * 10)
    keylist = [model_name + '_' + info + '_' + str(fold_num) for fold_num in range(0,10)]
    
    # 创建字典来保存每个fold的详细指标
    fold_metrics = {}
    
    for k in keylist:
        metrics = glo.get_val(k)
        fold_num = int(k.split('_')[-1])
        
        # 保存每个fold的详细指标
        fold_metrics[f'fold_{fold_num}'] = {
            'accuracy': float(metrics[0]),
            'precision_positive': float(metrics[1][1]),
            'precision_negative': float(metrics[1][0]),
            'recall_positive': float(metrics[2][1]),
            'recall_negative': float(metrics[2][0]),
            'f1_positive': float(metrics[3][1]),
            'f1_negative': float(metrics[3][0]),
            'weighted_precision': float(metrics[4]),
            'weighted_recall': float(metrics[5]),
            'weighted_f1': float(metrics[6])
        }
        
        res[0] += metrics[0]
        res[1] += metrics[1][1]
        res[2] += metrics[1][0]
        res[3] += metrics[2][1]
        res[4] += metrics[2][0]
        res[5] += metrics[3][1]
        res[6] += metrics[3][0]
        res[7] += metrics[4]
        res[8] += metrics[5]
        res[9] += metrics[6]
    
    # 添加平均指标
    fold_metrics['mean'] = {
        'accuracy': float(res[0]/10.0),
        'precision_positive': float(res[1]/10.0),
        'precision_negative': float(res[2]/10.0),
        'recall_positive': float(res[3]/10.0),
        'recall_negative': float(res[4]/10.0),
        'f1_positive': float(res[5]/10.0),
        'f1_negative': float(res[6]/10.0),
        'weighted_precision': float(res[7]/10.0),
        'weighted_recall': float(res[8]/10.0),
        'weighted_f1': float(res[9]/10.0)
    }
    
    # 保存详细指标到文件
    output_dir = f'./results_{model_name}_{info}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f'{output_dir}fold_metrics.json', 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    
    logger.info(f"Detailed fold metrics saved to {output_dir}fold_metrics.json")
    
    info_res = "\n flod " + str(10) + "\tAccuaryMean: " + str(res[0]/10.0) \
          + "\tWeightedPrecisionMean: " + str(res[7]/10.0) + "\tWeightedRecallMean: " \
          + str(res[8]/10.0) + "\t WeightedF1Mean： " + str(res[9]/10.0) \
          + "\tPrecisonMean: " + str(res[1]/10.0) + "\t NegativePrecisionMean： " + str(
        res[2]/10.0) \
          + "\tRecallMean: " + str(res[3]/10.0) + "\t NegativeRecallMean： " + str(res[4]/10.0) \
          + "\tF1Mean: " + str(res[5]/10.0) + "\t NegativeF1Mean： " + str(res[6]/10.0)
    logger.info(info_res)
    print(info_res)


# def clean_todo_comment(todo_data):
#     #  # 123123123, todo ( b / 111289526 ) : sadasdsada
#     re1 = r"#\s*\d+"
#     re2 = r"todo\s*\(\s*[^()]+\s*\)"
#     data_processed = []
#     for todo in todo_data:
#         new = re.sub(re2, "todo <info_tag>", todo)
#         new = re.sub(re1, "<link_id>", new)
#         data_processed.append(new)
#     return data_processed


# def read_file(path):
#     """load lines from a file"""
#     sents = []
#     with open(path, 'r') as f:
#         for line in f:
#             sents.append(str(line.strip()))
#     return sents


def load_all_data(file_path):
    data = pd.read_json(file_path, orient='records', lines=True)
    data = data.iloc[0:765,:4]
    return data


def tmp_label_process(labeled_data):
    high_low_labels = labeled_data['quality'].apply(lambda x:1 if x == 1 else 0)
    return high_low_labels



def main(file_path, model_type):
    mylogger = logger.mylog("dl")
    labeled_data = load_all_data(file_path)
    high_low_labels = tmp_label_process(labeled_data)
    pos_num = np.sum(high_low_labels == 1)
    print("Positive samples:", pos_num)
    glo._init()
    fold_num = 0
    # kf = KFold(n_splits=10, random_state=3407, shuffle=True)
    kf = StratifiedKFold(n_splits=10, random_state=3407, shuffle=True)
    # for train_index, test_index in kf.split(labeled_data):
    for train_index, test_index in kf.split(labeled_data, high_low_labels):
        train_x, train_y = list(labeled_data.iloc[train_index, 2]), list(high_low_labels.iloc[train_index])
        test_x, test_y = list(labeled_data.iloc[test_index, 2]), list(high_low_labels.iloc[test_index])
        
        # 欠采样
        rus = RandomUnderSampler(random_state=3407)
        train_x, train_y = rus.fit_resample(train_x, train_y)

        
        # Use single input models (CNN, bi-lstm, transformer)
        dl_container(train_x, train_y, test_x, test_y, mylogger, "score", fold_num, model_type)
        fold_num += 1

    ## calculate 10-fold metrics at last
    mylogger.info("==================modelcard high_low classifers=============")
    cal_last_metrics(mylogger, model_type, "score")


if __name__ == '__main__':
    file_path = "./modelcard_data (update).json"
    # type 1: bi-lstm 2: textcnn 3: transformer
    # model_list = ["textcnn", "bi-lstm", "transformer"]
    model_list = ["transformer"]
    for model_type in model_list:
        main(file_path, model_type)
