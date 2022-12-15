
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import shutil

from azureml.core import Run, Dataset, Workspace, Experiment

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve, log_loss

# Calculate model performance metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

def getRuntimeArgs():
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file_location", type=str)
    parser.add_argument('--training_data', dest='training_data', required=True)
    parser.add_argument('--testing_data', dest='testing_data', required=True)
    parser.add_argument('--model_file', dest='model_file', required=True)
    parser.add_argument('--model_name', dest='model_name', required=True)
    args = parser.parse_args()
    return args

def buildpreprocessorpipeline(X_raw):
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('onehotencoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")
    
    return preprocessor

def model_train(LABEL, df, run, training_data, testing_data):  
#     y_raw = df[LABEL]
#     X_raw = df.drop([LABEL], axis=1)
    
     # Train test split
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    
    train1 = train.dropna()
    test1 = test.dropna()
    
    X_train = train1
    y_train = train1[LABEL]
    
    X_test = test1
    y_test = test1[LABEL]

    
    #save train and test datasets
    os.makedirs(training_data, exist_ok=True)
    os.makedirs(testing_data, exist_ok=True)

    train1.to_csv(os.path.join(training_data, 'training_data.csv'), index=False )
    test1.to_csv(os.path.join(testing_data, 'testing_data.csv'), index=False)

    
    lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    preprocessor = buildpreprocessorpipeline(X_train)
    
    #estimator instance
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lg)])

    clf.fit(X_train, y_train)

    
    
    # calculate AUC
    y_scores = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    run.log('AUC', np.float(auc))

    
    # calculate test accuracy
    y_hat = clf.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    run.log('Accuracy', np.float(acc))


    precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_test, y_hat)
    
    run.log('precision', np.float(precisions[1]))
    run.log('recall', np.float(recall[1]))
    y_pred_prob = clf.predict_proba(X_test)[::,1]
    run.log('Log-Loss', np.float(log_loss(y_test, y_pred_prob)))
    
    run.parent.log('precision', np.float(precisions[1]))
    run.parent.log('recall', np.float(recall[1]))
    run.parent.log('Log-Loss', np.float(log_loss(y_test, y_pred_prob)))


    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_hat)
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    run.log_image(name = "ROC", plot = fig)
    plt.show()

    # plot confusion matrix
    # Generate confusion matrix
    cmatrix = confusion_matrix(y_test, y_hat)
    cmatrix_json = {
        "schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {
               "class_labels": ["0", "1"],
               "matrix": [
                   [int(x) for x in cmatrix[0]],
                   [int(x) for x in cmatrix[1]]
               ]
           }
    }
    
    run.log_confusion_matrix('ConfusionMatrix_Test', cmatrix_json)

    return clf
    # Save the trained model
    
    
def main():
    # Create an Azure ML experiment in your workspace
    args = getRuntimeArgs()
    
    raw_file_location = args.raw_file_location
    training_data = args.training_data
    testing_data = args.testing_data
    model_file = args.model_file
    model_name = args.model_name
    
    
    run = Run.get_context()

    dataset_dir = './dataset/'
    os.makedirs(dataset_dir, exist_ok=True)
    ws = run.experiment.workspace
    ds = ws.get_default_datastore()
    
    print(ws)
    

    print("Loading Data...")
    dataset = Dataset.Tabular.from_delimited_files(path = [(ds, raw_file_location)])
    # Load a TabularDataset & save into pandas DataFrame
    df = dataset.to_pandas_dataframe()
    
    print(df.head(5))
 
    model = model_train('Survived', df, run, training_data, testing_data)
    
    # Save the trained model
#     model_file = 'titanic_model.pkl'
#     joblib.dump(value=model, filename='./outputs/' + model_file)
#     os.makedirs(model_file_output, exist_ok=True)
#     shutil.copyfile('./outputs/' + model_file, os.path.join(model_file_output, 'titanic_model.pkl'))

    os.makedirs('./outputs', exist_ok=True)

    model_file_name = model_name  + '.pkl'
    file_name = './outputs/' +model_file_name

    joblib.dump(value=model, filename=file_name)

    os.makedirs(model_file, exist_ok=True)

    shutil.copyfile(file_name, os.path.join(model_file, model_file_name))
    

if __name__ == "__main__":
    main()
