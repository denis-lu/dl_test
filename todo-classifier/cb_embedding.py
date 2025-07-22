import os
import warnings
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from logger import mylog

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_codeBERT():
    # TODO change the model_path of CodeBERT to fit your environment
    model_path = '/home/ac/workspace/TODO-analysis/test_classifier/Model/codebert-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    model.cuda()
    return model, tokenizer

def code_bert_embedding(input_lst):
    print("Loading codeBERT model...")
    bert_model, tokenizer = load_codeBERT()
    print('codeBERT loaded')
    encoded_input = tokenizer(input_lst, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    input_ids, attention_masks = encoded_input['input_ids'], encoded_input['attention_mask']
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    with torch.no_grad():
        model_output = bert_model(input_ids, attention_mask=masks)
    # print(model_output.pooler_output)
    return model_output.pooler_output