
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath

import joblib
import os
import argparse
import shutil
import pandas as pd
import shutil


from azureml.core.model import InferenceConfig
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import Webservice, AksWebservice


parser = argparse.ArgumentParser("Evaluate model and register if more performant")

parser.add_argument('--model_name', dest='model_name', required=True)
parser.add_argument('--model_file', dest = 'model_file',  required=True)
parser.add_argument('--model_desc', dest = 'model_desc',  required=True)
parser.add_argument('--deploy_file', dest='deploy_file', required=True)


args, _ = parser.parse_known_args()
model_name = args.model_name
model_file = args.model_file
model_desc = args.model_desc

deploy_file = args.deploy_file

#Get current run
run = Run.get_context()

#Get associated AML workspace
ws = run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

#Get metrics associated with current parent run
metrics = run.get_metrics()

print('current run metrics')
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')


print('parent run metrics')
#Get metrics associated with current parent run
metrics = run.parent.get_metrics()

for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')

current_model_precision = float(metrics['precision'])
current_model_recall = float(metrics['recall'])
current_model_logloss = float(metrics['Log-Loss'])
# Get current model from workspace

model_description = model_desc
model_list = Model.list(ws, name=model_name, latest=True)
first_registration = len(model_list)==0

updated_tags = {'precision': current_model_precision, 'recall': current_model_recall}


print('updated tags')
print(updated_tags)


#upload model to the outputs directory
relative_model_path = 'outputs'
run.upload_folder(name=relative_model_path, path=model_file)

model_file_name = model_name  + '.pkl'


#If no model exists register the current model
if first_registration:
    print('First model registration.')
    model_reg = run.register_model(model_path='outputs/' + model_file_name, model_name=model_name,
                   tags=updated_tags,
                   properties=updated_tags)
else:
    #If a model has been registered previously, check to see if current model 
    #performs better. If so, register it.
    print(dir(model_list[0]))
    if float(model_list[0].tags['precision']) < current_model_precision:
        print('New model performs better than existing model. Register it.')

        model_reg = run.register_model(model_path='outputs/' + model_file_name, model_name=model_name,
                   tags=updated_tags,
                   properties=updated_tags)
        
        # Output accuracy to file
        with open(deploy_file, 'w+') as f:
            f.write(('deploy_model'))
    
    else:
        print('New model does not perform better than existing model. Cancel run.')
        
        with open(deploy_file, 'w+') as f:
            f.write(('do not deploy model'))
            
        run.complete()
