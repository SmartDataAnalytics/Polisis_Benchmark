from cnn import CNN
from privacy_policies_dataset import PrivacyPoliciesDataset as PPD
from os.path import join, isfile
from os import listdir
from collections import OrderedDict
import time
import torch
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
from data_processing import get_policy_of_interest_tokens

def _cm(y, y_hat):
    
    #Empty dict where the data will be stored
    cm = OrderedDict()
    
    #Computation fo true positives, false positives, true negatives and false negatives
    tp = (y * y_hat).sum()
    tn = ((1 - y) * (1 - y_hat)).sum()
    fp = (y_hat * (1 - y)).sum()
    fn = ((1 - y_hat) * y).sum()
    
    #Storage of results in the dictionary
    cm['TP'] = tp.item()
    cm['TN'] = tn.item()
    cm['FP'] = fp.item()
    cm['FN'] = fn.item()
    
    return cm

def _cms(y, y_hat):
    
    #Empty tensor where the data will be stored
    cms = torch.tensor([])
    
    #Computation of cm for every label and pack them in cms
    for label in range(12):
        cm = torch.tensor(_cm(y[:,label], y_hat[:,label]).values()).unsqueeze(1)
        cms = torch.cat([cms,cm],1)
        
    return cms

def _metrics(cm):  
    
    tp, tn, fp, fn = cm.values()
    eps = 1e-10
    
    #Computation of F1 score, precision and recall
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    
    return f1, p, r

def _metrics_t(y, y_pred, t):
    
    y_hat = y_pred > t
    cm = _cm(y, y_hat.double())
    
    return _metrics(cm)    
    

def _metrics_wrt_t(y, y_pred):    

    #Initialization of range of thresholds and empty lists to store results
    ts = np.arange(0, 1, 0.01)
    f1s = []
    ps = []
    rs = []

    #loop that computes metrics for every threshold
    for t in ts:

        f1, p, r = _metrics_t(y, y_pred, t)
        
        #Storage of results
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        
    return f1s, ps, rs, ts

def _best_t_idx(y, y_pred):
    
    idxs = []

    for label in range(12):
       
        f1, p, r, ts = _metrics_wrt_t(y[:,label], y_pred[:,label])
        index = np.array(f1).argmax().item()
        idxs.append(index)

    return idxs

def macro_metrics(y, y_hat):
    
    eps = 1e-10
    cms = _cms(y, y_hat)
    ps = cms[0] / (cms[0] + cms[2] + eps)
    rs = cms[0] / (cms[0] + cms[3] + eps)
    p = torch.mean(ps)
    r = torch.mean(rs)
    f1 = torch.mean(2 * ps * rs / (ps + rs + eps))
    
    return f1, p, r

def micro_metrics(y, y_hat):
    
    eps = 1e-10
    cms = _cms(y, y_hat)
    cm = cms.sum(1)
    p = cm[0] / (cm[0] + cm[2] + eps)
    r = torch.mean(cm[0] / (cm[0] + cm[3] + eps))
    f1 = torch.mean(2 * p * r / (p + r + eps))
    
    return f1, p, r

def best_metrics(y, y_pred):
    
    f1s = []
    ps = []
    rs = []
    ts = []
    idxs = _best_t_idx(y, y_pred)
    
    for idx, label in zip(idxs, range(12)):
        
        f1, p, r, t = _metrics_wrt_t(y[:,label], y_pred[:,label])
        f1s.append(f1[idx])
        ps.append(p[idx])
        rs.append(r[idx])
        ts.append(t[idx])
    
    return f1s, ps, rs, ts

def save_metrics(y, y_pred, path):
    
    def label_scores(y, y_pred, label, idx):          
    
        f1s, ps, rs, ts = predict._metrics_wrt_t(y[:,label], y_pred[:,label])
        best_scores = f1s[idx], ps[idx], rs[idx]
        scores_05 = predict._metrics_t(y[:,label], y_pred[:,label], 0.5)
        return scores_05 + best_scores
    
    with open(path, 'w') as f:
        writer = csv.writer(f)
        idxs = predict._best_t_idx(y, y_pred)
        for label, idx in zip(range(12), idxs):
            scores = label_scores(y, y_pred, label, idx)
            writer.writerows([scores])

def load_model(path, label, epochs_num):
    
    #We set the name of the model and its parameters
    models_files = join(path, 'cnn_300_200_[100, 25]_1_[3]_e{}_label{}_polisis_state.pt')
    model_file = models_files.format(epochs_num , label)
    params_files = join(path, 'cnn_300_200_[100, 25]_1_[3]_e{}_label{}_polisis_params.pkl')
    params_file = params_files.format(epochs_num , label)
    
    #We now load the parameters
    with open(params_file, 'rb') as f:
        params = pickle.load(f)
        
    #We now load the model and pass the parameters
    model = CNN(**params)
    model.load_state_dict(torch.load(model_file))
    
    return model

def load_12CNN_model(path):
    
    #We instantiate an empty dictionary that will contain the models
    model12cnn = OrderedDict()
    epochs_num = [60, 60, 150, 150, 70, 100, 150, 100, 70, 65, 80, 60]
    for label in range(12):
        model12cnn['model{}'.format(label)] = load_model(path, label, epochs_num[label])
        
    return model12cnn

def predict(data, models):
    
    #We instantiate an empty y and instantiate the x
    x = PPD.collate_data(data)[0]
    y = torch.tensor([])
    
    #We start a timer to compute predicions time and compute them
    start = time.time()
    for key, model in models.items():
        y_label = model(x)
        y = torch.cat([y, y_label],1)
    end = time.time()
        
    print("Prediction time: {} seconds". format(end - start))
    
    return



def main():

    #We set the folder path containing the models and load the labels
    # folder = 'trained_models/New folder'
    # models = load_12CNN_model(folder)
    # data_folder = 'datasets'
    # data_file = join(data_folder, 'test_dataset_label6.pkl')
    # data = PPD.unpickle_dataset(data_file)
    # predictions = predict(data, models)

    # We set the name of the model and its parameters
    path = 'trained_models'
    model_file = join(path, 'cnn_300_200_[100]_12_[3]_zeros_60-20-20_polisis_state.pt')
    params_file = join(path, 'cnn_300_200_[100]_12_[3]_zeros_60-20-20_polisis_params.pkl')

    #We set the folder containing the data already prepared for predicting
    data_folder = 'datasets'
    data_file = join(data_folder, 'test_dataset_label6.pkl')


    # We now load the parameters
    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    model = CNN(**params)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    

    #We load 8the labels
    #with open('labels.pkl', 'rb') as f:
        #labels = pickle.load(f)


    # labels = ('First Party Collection/Use', 'Third Party Sharing/Collection', 'User Access, Edit and Deletion', 'Data Retention',
    #           'Data Security', 'International and Specific Audiences', 'Do Not Track', 'Policy Change', 'User Choice/Control',
    #  'Introductory/Generic', 'Practice not covered', 'Privacy contact information')
    #
    # all_tokens , all_paragraphs = get_policy_of_interest_tokens("random_pp", "embeddings_data")
    # segments_tensor = dp.process_policy_of_interest(all_tokens , all_paragraphs)
    # predictions = model(segments_tensor)
    # y_pred = predictions > 0.5
    #
    # for row in range(len(all_paragraphs)):
    #     predictedValues = y_pred[row, :]
    #     for label in range(12):
    #         if predictedValues[label] == 1:
    #             print("paragraph " + str(row) + " : " + labels[label])
    #             print('--------------------------------------')
    #
    #

    data = PPD.unpickle_dataset(data_file)
    x = PPD.collate_data(data)[0]
    y_pred = model(x) > 0.5
    predictions = model(x)

    #
    # for row in range(len(y_pred)):
    #     predictedValues = y_pred[row, :]
    #     for label in range(12):
    #         if predictedValues[label] == 1:
    #             print("paragraph " + str(row) + " : " + labels[label])
    #             print('--------------------------------------')


    #Computation of all metrics

    f1s, ps, rs, ts = _metrics_wrt_t(data.labels_tensor,predictions)
    figure = plt.figure(figsize=(18,5))
    figure.suptitle('Micro Averages with respect to threshold')
    ax_f1 = figure.add_subplot(131)
    ax_f1.set_ylim(0.2,0.72)
    ax_p = figure.add_subplot(132)
    ax_p.set_ylim(0,1)
    ax_r = figure.add_subplot(133)
    ax_r.set_ylim(0,1)
    ax_f1.plot(ts, f1s)
    ax_p.plot(ts, ps)
    ax_r.plot(ts, rs)
    plt.show()


if __name__ == '__main__':

    main()
