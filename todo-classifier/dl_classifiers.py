# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import pandas as pd
import logger
import glo
from dl_models import *
from utils import *
import re
warnings.filterwarnings("ignore")


def save_model(epoch, model, training_stats, info, model_name):
    base_dir = './todo_'+ info + '_'+ model_name + '/epoch_' + str(epoch) + '/'
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
    model = nn.DataParallel(model, device_ids=[0, 1])

    logger.info("============begin %s training......" % model_name)
    train_dt = Data_processor(modelcard_data, train_y, config.batch_size)
    traindata_loader = train_dt.processed_dataloader
    test_dt = Data_processor(test_modelcard_data, test_y, config.batch_size)
    testdata_loader = test_dt.processed_dataloader
    print("model created.")
    epochs = config.num_epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(traindata_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()
    model.eval()
    loss_fn = F.cross_entropy
    best_epoch_acc = 0.0
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
        # evulates each epohc, save the metrics with the best accuracy.
        metrics_ = cal_metrics(test_y, all_pre_label)
        glo_key = model_name + '_' + info + '_' + str(fold_num)
        if metrics_[0] >= best_epoch_acc:
            best_epoch_acc = metrics_[0]
            glo.set_val(glo_key, metrics_)
            save_model(epoch_i + 1, model, training_stats, info, model_name)
    logger.info("============all epoches of %s trained finished......" % model_name)
    return


def cal_last_metrics(logger, model_name, info):
    # (acc_res, precision_res, recall_res, f1_res, weighted_precision, weighted_recall, weighted_f1)
    res = np.array([0.0] * 10)
    keylist = [model_name + '_' + info + '_' + str(fold_num) for fold_num in range(0,10)]
    for k in keylist:
        metrics = glo.get_val(k)
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
        # Use single input models (CNN, bi-lstm, transformer)
        dl_container(train_x, train_y, test_x, test_y, mylogger, "score", fold_num, model_type)
        fold_num += 1

    ## calculate 10-fold metrics at last
    mylogger.info("==================Todo comment good_bad classifers=============")
    cal_last_metrics(mylogger, model_type, "score")


if __name__ == '__main__':
    file_path = "./modelcard_data (update).json"
    # type 1: bi-lstm 2: textcnn 3: transformer
    model_list = ["textcnn", "bi-lstm"]
    for model_type in model_list:
        main(file_path, model_type)
