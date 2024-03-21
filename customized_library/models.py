from sklearn.metrics import f1_score, log_loss, classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd
from customized_library import plot_lib
import itertools



#######################F1###################################################

def pre_model(model, X_train, y_train, X_valid, y_valid):
    y_train_pred, y_valid_pred, model = plot_lib.train(model, X_train, y_train, X_valid)
    train_f1 = f1_score(y_train, np.argmax(y_train_pred, axis=1), average='weighted')
    valid_f1 = f1_score(y_valid, np.argmax(y_valid_pred, axis=1), average='weighted')
    return train_f1, valid_f1


def best_model(X_train,y_train,X_valid, y_valid, model, param, threshold=0.0, show=False):
    v = 0; t = 0
    length = len(param)
    names = param.keys()
    params = None
    
    for i in itertools.product(*list(param.values())):
        dict = {k:v for k, v in zip(names, i)}
        model.set_params(**dict)
        train_f1, valid_f1 = pre_model(model, X_train, y_train, X_valid, y_valid)
        if train_f1 - valid_f1 < threshold and valid_f1 > v:
            v = valid_f1, 
            t = train_f1
            t1 = classification_report(y_train_lab, np.argmax(y_train_pred, axis=1), zero_division=0.0)
            v1 = classification_report(y_valid_lab, np.argmax(y_valid_pred, axis=1), zero_division=0.0)
            params = model.get_params()

    if params == None:
        return 'Threshold is small!'
    for i in names:
        print(i + ':  ' + str(params[i]), end='    ')
    print()
    print('train f1: '+str(t) + ' valid f1: '+str(v))
    if show == True:
        print('train report: '+str(t1) + ' valid report: '+str(v1))

##########################loss##############################

def pre_model_loss(model, X_train, y_train, X_valid, y_valid):
    y_train_pred, y_valid_pred, model = plot_lib.train(model, X_train, y_train, X_valid)
    train_f1 = log_loss(y_train.values, model.predict_proba(X_train))
    valid_f1 = log_loss(y_valid.values, model.predict_proba(X_valid))
    return train_f1, valid_f1
    
def best_model_loss(X_train,y_train, y_train_ohe,X_valid, y_valid, y_valid_ohe, model, param, threshold=0.0, show_all=False, show_weighted=True):
    v = 10; t = 0
    length = len(param)
    names = param.keys()
    params = None
    
    for i in itertools.product(*list(param.values())):
        dict = {k:v for k, v in zip(names, i)}
        model.set_params(**dict)
        y_train_pred, y_valid_pred, model = plot_lib.train(model, X_train, y_train, X_valid)
        train_loss = log_loss(y_train_ohe.values, model.predict_proba(X_train))
        valid_loss = log_loss(y_valid_ohe.values, model.predict_proba(X_valid))
        
        if valid_loss - train_loss < threshold and valid_loss < v:
            v = valid_loss, 
            t = train_loss
            precision, recall, fscore, _ = precision_recall_fscore_support(y_train, np.argmax(y_train_pred, axis=1), average='weighted')
            precision_valid, recall_valid, fscore_valid, _ = precision_recall_fscore_support(y_valid, np.argmax(y_valid_pred, axis=1), average='weighted')
            acc, acc_valid = model.score(X_train, y_train), model.score(X_valid, y_valid)
            data = {'train':[t, acc, precision, recall, fscore], 'valid':[v[0], acc_valid, precision_valid, recall_valid, fscore_valid]}
            result = pd.DataFrame(data=data, index=['loss', 'accuracy', 'precision', 'recall', 'f1_score'])
            if show_weighted == True:
                t1 = classification_report(y_train, np.argmax(y_train_pred, axis=1), zero_division=0.0)
                v1 = classification_report(y_valid, np.argmax(y_valid_pred, axis=1), zero_division=0.0)
            params = model.get_params()

    if params == None:
        return 'Threshold is small!'
    print('The best parameters of ' + type(model).__name__, end=',  ')
    for i in names:
        print(i + ':  ' + str(params[i]), end='    ')
    print('\n')
    print(result)

    if show_all == True:
        print('\ntrain report: \n\n'+str(t1) + '\n valid report: \n\n'+str(v1))


