import numpy as np
import pandas as pd
import pickle
from scipy import interp
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score
import matplotlib.pyplot as plt

# Write submission based on sample file
def write_submission(y_submit,filename):
    sample_submission = pd.read_csv('../../data/sample_submission.csv', index_col='TransactionID')
    sample_submission['isFraud'] = y_submit[:,1]
    sample_submission.to_csv('../../submissions/'+filename)

# save the model to disk
def save_model(clf,mean_auc,std_auc,filename):
    pickle.dump(clf, open('../../runs/'+filename+'.model', 'wb'))
    auc_file = filename + '.results'
    file2 = open(auc_file,'w')
    for line in ["mean_auc_score: "+str(mean_auc),"\nstd_auc_score: "+str(std_auc)]:
        file2.writelines(line)
    file2.close()

# roccin function generates the data for the ROC curve.
def roccin(y_train,y_pred,mean_fpr,tprs,aucs):
    fpr, tpr, thresholds = roc_curve(y_train, y_pred) 
    tprs.append(interp(mean_fpr, fpr, tpr)) #Interpolates tpr at the mean_fpr (for ROC curve)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    return fpr, tpr, tprs, roc_auc, aucs

# Runs CV, plots AUC curves, generates accuracy scores
def auc_plot(clf, cv, X_train, y_train):
    tprs = [] #roc curve translated to the 100 point 0-1 linspace
    aucs = []
    tprs_train = [] #roc curve translated to the 100 point 0-1 linspace
    aucs_train = []
    test_accuracy_scores = []
    train_accuracy_scores = []
    mean_fpr = np.linspace(0, 1, 1000)


    i = 0
    for train, test in cv.split(X_train, y_train):
        y_pred = clf.fit(X_train.iloc[train], y_train.iloc[train]).predict_proba(X_train.iloc[test])[:,1]
        y_pred_train = clf.predict_proba(X_train.iloc[train])[:,1]
        y_pred_binary = clf.predict(X_train.iloc[test])
        y_pred_train_binary = clf.predict(X_train.iloc[train])

        # Test AUC curve 
        fpr, tpr, tprs, roc_auc, aucs = roccin(y_train.iloc[test],y_pred,mean_fpr,tprs,aucs) 
        test_accuracy_scores.append(accuracy_score(y_train.iloc[test],y_pred_binary))

        #Train AUC curve
        fpr_train, tpr_train, tprs_train, roc_auc_train, aucs_train = roccin(y_train.iloc[train],
                                                                             y_pred_train,mean_fpr,
                                                                             tprs_train,aucs_train)    
        train_accuracy_scores.append(accuracy_score(y_train.iloc[train],y_pred_train_binary))

        #Print the ROC plot
        print("Fold {} complete.".format(i))    
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return mean_auc, std_auc, test_accuracy_scores, train_accuracy_scores

