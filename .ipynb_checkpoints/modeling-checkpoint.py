from sklearn.metrics import auc, confusion_matrix, PrecisionRecallDisplay, classification_report, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.metrics import Recall, Precision, R2Score


def train(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train) 
    
    y_test_pred_prob = model.predict_proba(X_test)
    y_train_pred_prob = model.predict_proba(X_train)
    
    return y_train_pred, y_test_pred, model, y_train_pred_prob, y_test_pred_prob


def train_tf(model, X_train, y_train, X_test, y_test, optimizer='adam', batch_size=32, epochs=10, verbose=1, p=0.5, restart=True):
    if restart == True:
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision()])   
    
    X_train_np = X_train.iloc[:,:].values 
    y_train_np = y_train.iloc[:].values
    X_test_np = X_test.iloc[:,:].values
    y_test_np = y_test.iloc[:].values
    
    history = model.fit(X_train_np, y_train_np, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test_np, y_test_np))
    
    y_test_pred_prob = model.predict(X_test_np, verbose=verbose)
    y_train_pred_prob = model.predict(X_train_np, verbose=verbose) 
    
    y_train_pred = [int(i) for i in (y_train_pred_prob.reshape(y_train_pred_prob.shape[0],)>p)]
    y_test_pred = [int(i) for i in (y_test_pred_prob.reshape(y_test_pred_prob.shape[0],)>p)]
    
    y_train_pred_prob = [i for i in y_train_pred_prob.reshape(y_train_pred_prob.shape[0],)]
    y_test_pred_prob = [i for i in y_test_pred_prob.reshape(y_test_pred_prob.shape[0],)]
    
    return y_train_pred, y_test_pred, model, y_train_pred_prob, y_test_pred_prob, history


class Ploting_And_Report:
    
    def __init__(self, y_train, y_train_pred, y_test, y_test_pred):
        self.__y_train = y_train.iloc[:].values
        self.__y_train_pred = y_train_pred
        self.__y_test = y_test.iloc[:].values
        self.__y_test_pred = y_test_pred
        
    
    @property
    def y_train(self):
        return self.__y_train
    
    @property
    def y_train_pred(self):
        return self.__y_train_pred
    
    @property
    def y_test(self):
        return self.__y_test
    
    @property
    def y_test_pred(self):
        return self.__y_test_pred
    
    def a(self,):
        print(y_train)
    
    
    def plot_confusion_matrix(self,):

        fig, ax = plt.subplots(1, 2, figsize=(15,6))

        cf_train = confusion_matrix(self.y_train, self.y_train_pred)
        cf_train_plot = sns.heatmap(cf_train, annot=True, fmt='d', ax=ax[0])
        cf_train_plot.set_title('Train confusion matrix')

        cf_test = confusion_matrix(self.y_test, self.y_test_pred)
        cf_test_plot = sns.heatmap(cf_test, annot=True, fmt='d', ax=ax[1])
        cf_test_plot.set_title('Test confusion matrix')

        plt.plot()


    def plot_precision_Recal_curve(self,y_train_pred_prob, y_test_pred_prob):

        fig, ax = plt.subplots(1, 2, figsize=(15,6))

        PrecisionRecallDisplay.from_predictions(self.y_train, y_train_pred_prob, plot_chance_level=True, pos_label=1, ax=ax[0])
        ax[0].set_title("Train Precision-Recall curve for Transported")

        PrecisionRecallDisplay.from_predictions(self.y_test, y_test_pred_prob, plot_chance_level=True, pos_label=1, ax=ax[1])
        ax[1].set_title("Test Precision-Recall curve for Transported")

        plt.show()

    
    
    def show_classification_report(self,):

        print('Train classification_report\n')
        print(classification_report(self.y_train, self.y_train_pred))

        print('\n\nTest classification_report\n')
        print(classification_report(self.y_test, self.y_test_pred))

    
    def plot_roc_curve(self,y_train_pred_prob, y_test_pred_prob):

        fpr_train, tpr_train, thresholds = roc_curve(self.y_train, y_train_pred_prob)
        fpr_test, tpr_test, thresholds = roc_curve(self.y_test, y_test_pred_prob)

        fig, ax = plt.subplots(1, 2, figsize=(15,6))
        lin = np.linspace(0, 1, 10)


        ax[0].plot(fpr_train, tpr_train)
        ax[0].plot(lin, lin)
        ax[0].set_title('ROC Curve for Train')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')


        ax[1].plot(fpr_test, tpr_test)
        ax[1].plot(lin, lin)
        ax[1].set_title('ROC Curve for Test')

        fig.tight_layout()
        plt.show()

        
    def show_auc(self,y_train_pred_prob, y_test_pred_prob):
        fpr_train, tpr_train, thresholds = roc_curve(self.y_train, y_train_pred_prob)
        fpr_test, tpr_test, thresholds = roc_curve(self.y_test, y_test_pred_prob)
        
        print('Train AUC: ' + str(auc(fpr_train, tpr_train)) + '   Test AUC: ' + str(auc(fpr_test, tpr_test)) + '\n\n\n\n')


    def ploting_and_scores(self, average='macro'): 
        prf_train = precision_recall_fscore_support(self.y_train, self.y_train_pred, average=average)
        prf_test = precision_recall_fscore_support(self.y_test, self.y_test_pred, average=average)
        
        dic = {'train':[prf_train], 'test':[prf_test]}

        return dic



    
