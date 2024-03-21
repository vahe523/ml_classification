from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_classification_report(y_train, y_train_pred, y_valid, y_valid_pred):
    print('train\n')
    print(classification_report(y_train, np.argmax(y_train_pred, axis=1), zero_division=0.0), end='\n\n\n')

    print('validation\n')
    print(classification_report(y_valid, np.argmax(y_valid_pred, axis=1), zero_division=0.0), end='\n\n')
    
    
def plot_precision_recall_curve(y_train_encoder, y_train_pred, y_valid_encoder, y_valid_pred):
    fig, ax = plt.subplots(3,2,figsize=(20,30))
    
    for i, ax_row in enumerate(ax):
        precision, recall, thresholds = precision_recall_curve(y_train_encoder.iloc[:,i].values, y_train_pred[:,i])
        
        ax_row[0].plot(recall, precision)
        ax_row[0].set_xlabel('recall')
        ax_row[0].set_ylabel('precision')
        ax_row[0].set_title('Train ' + str(i))
        
        precision, recall, thresholds = precision_recall_curve(y_valid_encoder.iloc[:,i].values, y_valid_pred[:,i])
        
        ax_row[1].plot(recall, precision)
        ax_row[1].set_xlabel('recall')
        ax_row[1].set_ylabel('precision')
        ax_row[1].set_title('Validation ' + str(i))
        
        
def train(model, X_train, y_train, X_valid):
    model.fit(X_train, y_train)
    
    return model.predict_proba(X_train), model.predict_proba(X_valid), model