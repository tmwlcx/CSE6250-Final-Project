# -*- coding: utf-8 -*-
from __future__ import print_function


from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from scikitplot.metrics import plot_roc
import sys



class CVD_Diagnosis():
    
    def __init__(self, path):
        
        self.path = path
        
        # scala saves multiple files, get the .csv
        file = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f.endswith(".csv")][0]

        # read in the data
        data = []
        with open(self.path + file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=",")
            header = next(readCSV)
        #     print(header)
            for row in readCSV:
                data.append([row[2]] + [row[0]] + [float(num) for num in row[1].split(",")])
    
        # create the data dictionary
        data_dict = {}
        for row in data:
            pid = row[0]
            if pid not in data_dict:
                data_dict[pid]= {}
                data_dict[pid][row[1]] = row[2:]
            else:
                data_dict[pid][row[1]] = row[2:]

        # infer resolution from file
        resolution = len(list(data_dict[next(iter(data_dict))].values())[0])
        
        # infer channel_names from file
        channel_names = list(data_dict[next(iter(data_dict))].keys())
    
        # dataframe has column structure { NSRRID | CVD Diagnosis | x1 | x2| ... |xN } where N = resolution*len(channel_names)
        col_names = ["nsrrid"] + ["x"+str(i) for i in range(resolution * len(channel_names))]
        df = pd.DataFrame(
            [[int(key.split("-")[1])] + data_dict[key]['EEG'] + data_dict[key]['ECG'] + data_dict[key]['EMG'] for key in data_dict.keys()],
            columns = col_names)
        outcomes = pd.read_csv("./python/cvd_outcomes/shhs-cvd-summary-dataset-0.15.0.csv", usecols=["nsrrid", "any_cvd"])
        df = outcomes.merge(df)

        def eprint(*args, **kwargs):
            print(*args, file=sys.stderr, **kwargs)

        eprint("WARNING! the following NSRRIDs do not have a CVD diagnosis (diagnoses) info and will be dropped:")
        eprint(str(df[df.any_cvd.isna()].nsrrid.tolist())[1:-1])
        df = df[df.any_cvd.notna()]
        
        #save the X and y values for later
        self.X = df.values[:,2:]
        self.y = df.values[:,1]
        
        scaler = StandardScaler()
        Xs = scaler.fit_transform(self.X)
        
        # save the training and testing data for later
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            Xs, self.y, test_size=0.25, random_state=1738)
        
    def classify_patients(self, class_weights=None, n_splits=10):
    
        # create a class_weights dictionary to hold the different class weights for the function
        cws = {
            "balanced":{i:len(self.y)/(len(list(set(self.y)))*list(self.y==i).count(True)) for i in list(set(self.y))},
            "None": None,
            "favor_CVD": {0: 0.25, 1: 2.9}
        }
        if class_weights:
            assert class_weights in cws, "leave class_weights blank or specify appropriate \
            class weights ('balanced', 'None', or 'favor_CVD')"
            cw = cws[class_weights]
        else:
            cw = None
        
        # fit the classfier
        self.clf = LogisticRegressionCV(max_iter = 10000,cv=n_splits,multi_class='ovr', 
                                   random_state=0, class_weight=cw, refit = False,
                                   tol=1e-4, n_jobs=-1).fit(self.X_train, self.y_train)
        
        #print out some summary data
        print("Validation Set Numbers:")
        print("Number of CVD negative patients = {}".format(int(self.y_test.shape[0] - self.y_test.sum())))
        print("Number of CVD positive patients = {}".format(int(self.y_test.sum())))
        print("\nLogistic Regression Classification Accuracy on Training Set:\t{:.2f}%".format(self.clf.score(self.X_train, self.y_train)*100))
        print("Logistic Regression Classification Accuracy on Validation Set:\t{:.2f}%".format(self.clf.score(self.X_test,self.y_test)*100))
        
        # print confusion matrices
        print("\nTEST SET Confusion Matrix class totals")
        print(confusion_matrix(self.y_test, self.clf.predict(self.X_test)))
        print("\nTEST SET Confusion Matrix True value normalized")
        print(confusion_matrix(self.y_test, self.clf.predict(self.X_test), normalize='true'))

        
    def make_confusion_matrix(self,validation_set='test'):
        assert type(validation_set)==str, "Validation set must be string ('test' or 'train')"
        if validation_set.lower() == 'test':
            vs = self.X_test
            ys = self.y_test
        elif validation_set.lower() == 'train':
            vs = self.X_train
            ys = self.y_train
        else:
            raise ValueError("validation set should be either 'test' or 'train'")
        fig, ax = plt.subplots()
        plot_confusion_matrix(self.clf, vs, ys, normalize='true',
                                   cmap=plt.cm.Blues, display_labels=['without CVD', 'with CVD'], 
                                   ax=ax, include_values=True)  # doctest: +SKIP
        plt.title('Confusion Matrix for {} Set'.format(validation_set.upper()))
#         plt.savefig('CM_val.jpeg')
        plt.show()
    
    def make_roc_auc_plot(self, validation_set):
        assert type(validation_set)==str, "Validation set must be string ('test' or 'train')"
        if validation_set.lower() == 'test':
            Xs = self.X_test
            ys = self.y_test
        elif validation_set.lower() == 'train':
            Xs = self.X_train
            ys = self.y_train
        else:
            raise ValueError("validation set should be either 'test' or 'train'")
        probs = self.clf.predict_proba(Xs)
        preds = probs[:,1]

        # plot_roc(ys, probs, plot_macro=False, plot_micro=False)
        fig, ax = plt.subplots()
        plot_roc_curve(self.clf, Xs, ys, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',alpha=.8)

        plt.title('ROC Curve for {} set'.format(validation_set.upper()))
        plt.show()
    
    def show_graphics(self, validation_set='test'):
        self.make_confusion_matrix(validation_set)
        self.make_roc_auc_plot(validation_set)
        
            
        
        
