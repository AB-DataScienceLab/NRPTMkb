#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import json
from sklearn.metrics import  average_precision_score

import time

from src.utils import  handle_flags, limit_gpu_memory_growth, PTMDataGenerator
from src.model import  LSTMTransFormer

t0 = time.time()
handle_flags()

def build_model(FLAGS, optimizer , unique_labels, model_name, num_model):

    model = LSTMTransFormer(FLAGS,FLAGS.model,optimizer,  \
        num_layers=[num_model * 3, num_model * 3 + 1, num_model * 3 + 2],  num_heads=8,dff=512, rate=0.1, binary=False,\
        unique_labels=unique_labels, fill_cont=FLAGS.fill_cont)
    
    model.create_model(graph=FLAGS.graph)
    pretrain_model = tf.keras.models.load_model(model_name)
    # pdb.set_trace()
    for layer in pretrain_model.layers:
        if layer.name != 'my_last_dense' and layer.name != 'intermediate_dense' and len(layer.get_weights()) != 0:
            model.model.get_layer(name=layer.name).set_weights(layer.get_weights())  
            model.model.get_layer(name=layer.name).trainable = False 
            
    print(model.model.summary())
    return model 

def build_model_graph(FLAGS, optimizer , unique_labels, pretrain_model):
    if FLAGS.model=='LSTMTransformer':
        model = LSTMTransFormer(FLAGS,FLAGS.model,optimizer,  \
            num_layers=FLAGS.n_lstm, num_heads=8,dff=512, rate=0.1,binary=False,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
        for layer in pretrain_model.layers:
            if len(layer.get_weights())!=0 and layer.name!='my_last_dense':
                model.model.get_layer(name=layer.name).set_weights(layer.get_weights())   
                if layer.name != 'my_last_dense':
                    model.model.get_layer(name=layer.name).trainable = False
    print(model.model.summary())
    return model     

def ensemble_get_weights(PR_AUCs, unique_labels):
    weights = {ptm:None for ptm in unique_labels}
    for ptm in unique_labels:
        weight = np.array([PR_AUCs[str(i)][ptm] for i in range(len(PR_AUCs))])
        weight = weight/np.sum(weight)
        weights[ptm] = weight
    return weights # {ptm_type}


def save_model(model, FLAGS, fold=None):
    model_name = './new_model/'
    model_name+= 'MIND_fold_' + str(fold)
    model.model.save(model_name)


def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)
    # tf.config.run_functions_eagerly(True)#TODO remove

    class_weights = None
    # =============================================================
    train_dat_aug = PTMDataGenerator(f'train_data_new.json', FLAGS, shuffle=True,ind=None, eval=False, class_weights=class_weights)
    unique_labels = train_dat_aug.unique_labels
    input(unique_labels)
    train_dat_aug.train_val_split()
    test_dat_aug = PTMDataGenerator(f'test_data_new.json', FLAGS, shuffle=True,ind=None, eval=True)#
    test_dat_aug.update_unique_labels(unique_labels)
    # =============================================================

    training_callbacks = [
            #keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
            keras.callbacks.EarlyStopping(monitor='loss',patience = 2, restore_best_weights = True),
            #keras.metrics.Accuracy(),
        ] 
    
    PR_AUCs = {}
    
    for i in range(FLAGS.n_fold):       
        val_dat_aug = train_dat_aug.init_fold(i)
        train_dat_aug.on_epoch_end()

        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=True)
        model = build_model(FLAGS, optimizer, unique_labels, f'./saved_model/MIND_fifteenfold_fold_{i}', num_model=i) 
        model.train( train_dat_aug, val_dat_aug, FLAGS.num_epochs, callbacks=training_callbacks)
        
        AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len, test_dat_aug, FLAGS.batch_size, unique_labels)
        for u in unique_labels:
            print('%.3f'%PR_AUC[u])
        for u in unique_labels:
            print(u)
            print(confusion_matrixs[u])       
        save_model(model, FLAGS, fold=i)
        PR_AUCs[i] = PR_AUC
        # save PR_AUC 
    
    with open('./new_model/MIND_PRAU.json', 'w') as fw:
        json.dump(PR_AUCs, fw)

    print('----------------------Ensemble evaluation -----------------------')
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, amsgrad=True)
    with open("./new_model/MIND_PRAU.json",'r') as fw:
        PR_AUCs = json.load(fw)

    y_preds = []
    for i in range(FLAGS.n_fold):
        model = build_model(FLAGS, optimizer, unique_labels, f'./new_model/MIND_fold_{i}', num_model=i) 
        model.model = tf.keras.models.load_model(f'./new_model/MIND_fold_{i}')
        y_true, y_pred = model.predict(FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels, binary=False)
        y_preds.append(y_pred)
    weights = ensemble_get_weights(PR_AUCs, unique_labels)

    for u in unique_labels:
        y_pred = np.stack([np.array(y_preds[i][u]*weights[u][i]) for i in range(len(y_preds))])
        y_pred = np.sum(y_pred, axis=0)
        pr_auc = average_precision_score(y_true[u], y_pred)
        print('%.3f'%pr_auc)

    # AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len, test_dat_aug, FLAGS.batch_size, unique_labels)
    
    # group_labels = {
    #     'Methylation': ['Methylation_K', 'Methylation_R', 'Methylation_G'],
    #     'Acetylation': ['Acetylation_K', 'Acetylation_M'],
    #     'Palmitoylation': ['Palmitoylation_C'],
    #     'Monomethylation': ['Monomethylation_K'],
    #     'Phosphorylation': ['Phosphorylation_S', 'Phosphorylation_T', 'Phosphorylation_Y', 'Phosphorylation_C', 'Phosphorylation_E', 'Phosphorylation_K', 'Phosphorylation_R'],
    #     'Sumoylation': ['Sumoylation_K', 'Sumoylation_F', 'Sumoylation_L'],
    #     'Ubiquitination': ['Ubiquitination_K'],
    #     'Glycosylation': ['Glycosylation_N', 'Glycosylation_S', 'Glycosylation_T'],
    #     'S-Nitrosylation': ['S-Nitrosylation_C'],
    #     'O-glcNAcylation': ['O-glcNAcylation_S'],
    #     'Thiol oxidation': ['Thiol oxidation_C']
    # }

    # L = []
    # for k, v in group_labels.items():
    #     tp, fp, tn, fn = 0, 0, 0, 0
    #     for ptm in v:
    #         if ptm not in unique_labels:
    #             continue
    #         tn += confusion_matrixs[ptm]['0']['0']
    #         fp += confusion_matrixs[ptm]['0']['1']
    #         fn += confusion_matrixs[ptm]['1']['0']
    #         tp += confusion_matrixs[ptm]['1']['1']
    #     L.append({
    #         'PTM': k,
    #         'True Positive': tp,
    #         'False Positive': fp,
    #         'True Negative': tn,
    #         'False Negative': fn,
    #         'Precision': tp / (tp + fp) if tp + fp != 0 else np.nan,
    #         'Recall': tp / (tp + fn) if tp + fn != 0 else np.nan,
    #         'Accuracy': (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn != 0 else np.nan
    #     })
    # pd.DataFrame(L).to_csv('PTM_after_training_test_data.csv', index=False)
    


if __name__ == '__main__':
    app.run(main)

