# CSE_6250
Big Data For Healthcare Final Project: Learning From Sleep Data
Team 24: Tim Wilcox


## Introduction
This repository contains programs in two languages: *Apache Spark* and *Python*. The Spark program processes **.edf** files into spectral density estimates of the following waveforms:
  * EEG
  * ECG
  * EMG
 
***
>Folder structure as follows:
```
LearningFromSleepData/
|-- Demo Notebook.ipynb
|-- python
|   |-- CVD_Diagnosis.py
|   `-- cvd_outcomes
|       `-- shhs-cvd-summary-dataset-0.15.0.csv
`-- scala
    |-- build.sbt
    |-- data
    |-- project
    |   |-- build.properties
    |   `-- plugins.sbt
    |-- sbt
    |   |-- sbt
    |   `-- sbt-launch.jar
    `-- src
        `-- main
            |-- resources
            |   |-- log4j.properties
            |   `-- logback.xml
            `-- scala
                `-- edu
                    `-- gatech
                        `-- cse6250
                            |-- EDFOps
                            |   `-- EDFOps.scala
                            |-- helper
                            |   `-- SparkHelper.scala
                            `-- main
                                `-- Main.scala

16 directories, 13 files
```

***

**Note:** Before you run either program, you must put the polysomnogram data (.edf files) in the ./scala/data _AND_ the data must have EEG, EMG, and ECG signals present. 

## 1. Feature Engineering with Apache Spark
Open a terminal in the root directory and run the following:

```
(cd scala/ ; sbt compile run)
```

This process will take some time depending on the number of .edf files to read. On average, the full [Sleep Heart Health Study](https://sleepdata.org/datasets/shhs) SHHS1 dataset takes just under 9 hours to process approximately 5,800 files. At the conclusion of the operation, Spark has been configured to save the processed files in the './scala/output/Periodograms.csv/' directory. Use this location for the classification step.

## 2. Classification with Python3

You may choose to run the CVD_Diagnosis.py file on its own or through a Jupyter Notebook for greater interactivity and control. Either way, the structure is as follows:

*class* CVD_Diagnosis (*path*)
  The main class for classifying data from the spark output dataframe.
**Parameter**
>`path` \[String\] : 

the local path to the output of the Spark dataframe
  
**Returns**
>`self`
  
**Attributes**
>`X` \[numpy.ndarray\[numpy.float64\]\]: 

*(number of patients, number of signals * resolution of each signal)* The feature vector. Each row represents one patient and each column contains the PSD estimate for one frequency bin of the periodogram for each of three signals. X\[0:250\] represents the EEG periodogram, X\[250:500\] represents the ECG periodogram, and X\[500:750\] represents the EMG periodogram.

>`y` \[numpy.ndarray\[numpy.int\]\]:

*(number of patients,)* The labels. `y==1`: CVD positive. 

>`X_train` \[numpy.ndarray\[numpy.float64\]\]: 

*(number of patients, number of signals * resolution of each signal)* The training set portion of the feature vector. 

>`y_train` \[numpy.ndarray\[numpy.int\]\]:

*(number of patients,)* The training set labels. `y==1`: CVD positive. 

>`X_test` \[numpy.ndarray\[numpy.float64\]\]: 

*(number of patients, number of signals * resolution of each signal)* The validation set portion of the feature vector. 

>`y_test` \[numpy.ndarray\[numpy.int\]\]:

*(number of patients,)* The validation set labels. `y==1`: CVD positive. 

>**Methods**

>classify_patients(*class_weights=None, n_splits=100*)

>Uses a logisitic regression classifier with cross-validation (from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html?highlight=logist%20cv#sklearn.linear_model.LogisticRegressionCV)) to classify the patients and applies a class_weight to each class, if desired. 

>**Returns**






  

