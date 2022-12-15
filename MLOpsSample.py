#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Remove below from your notebook, example of setting variables in key vault


# In[ ]:





# In[ ]:





# In[126]:


registered_env_name = "Prediction_env"
experiment_folder = 'devOps_train_pipeline'
cluster_name = "wipfli-cluster"
conda_yml_file = './Environment/Prediction_env.yml'
model_name = 'wipfli'
prefix = 'wipfli'


# In[16]:


# Import required packages
from azureml.core import Workspace, Experiment, Datastore, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute, DataFactoryCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineParameter, PipelineData
from azureml.data.output_dataset_config import OutputTabularDatasetConfig, OutputDatasetConfig, OutputFileDatasetConfig
from azureml.data.datapath import DataPath
from azureml.data.data_reference import DataReference
from azureml.data.sql_data_reference import SqlDataReference
from azureml.pipeline.steps import DataTransferStep
import logging
from azureml.core.model import Model
from azureml.exceptions import WebserviceException


# In[17]:


import azureml.core
import os, shutil
from azureml.core import Workspace, Experiment, Datastore, Environment, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication

def get_environment_variables():
    global envs
    global run_by_notebook 
    run_by_notebook = False
    environment_variables = ['tenantId', 'servicePrincipalId', 'servicePrincipalPassword', 'wsName', 
                         'subscriptionId', 'resourceGroup']
    envs = {}
    for x in environment_variables:
        if os.environ.get(x) == None:
            #get the values from keyvault
            run_by_notebook = True
            print('retrieve from key vault, value is None: ' + x)
            ws = Workspace.from_config()
            keyvault = ws.get_default_keyvault()
            kv_results = keyvault.get_secrets(environment_variables)
            envs = kv_results
            for x in envs:
                os.environ.setdefault(x, envs[x])
            exit
        else:
            envs[x] = os.environ.get(x)
    return run_by_notebook



get_environment_variables()
sp = ServicePrincipalAuthentication(tenant_id=envs['tenantId'], # tenantID
                                    service_principal_id=envs['servicePrincipalId'], # clientId
                                    service_principal_password=envs['servicePrincipalPassword']) # clientSecret
ws = Workspace.get(name=envs['wsName'],
                       auth=sp,
                       subscription_id=envs['subscriptionId'],
                       resource_group=envs['resourceGroup'])
ws.get_details()

print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
print('Run By Notebook:' + str(run_by_notebook))
# Get the default datastore
default_ds = ws.get_default_datastore()


# In[18]:


#Select AML Compute Cluster
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException


try:
    # Check for existing compute target
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)


# In[ ]:


try:
    initial_model = Model(ws, model_name)
    inital_model_version = initial_model.version
except WebserviceException :
    inital_model_version = 0
print('inital_model_version = ' + str(inital_model_version))


# In[20]:


import os
import shutil
# Create a folder for the pipeline step files
os.makedirs(experiment_folder, exist_ok=True)

print(experiment_folder)

run_path = './run_outputs'
try:
    shutil.rmtree(run_path)
except:
    print('continue run_outputs directory does not exits')


# In[ ]:





# In[21]:


env = Environment.from_conda_specification("Prediction-env", conda_yml_file)
env.register(workspace=ws)


# In[22]:


registered_env_name


# In[24]:


# Create a Python environment for the experiment (from a .yml file)

try:
    env = Environment.from_conda_specification("Prediction_env", conda_yml_file)
    env.register(workspace=ws)
    registered_env = Environment.get(ws, registered_env_name)
    pipeline_run_config = RunConfiguration()
    
    # Use the compute you created above. 
    pipeline_run_config.target = pipeline_cluster

    # Assign the environment to the run configuration
    pipeline_run_config.environment = registered_env
    print ("Run configuration created.")
except Exception as e: 
    print(e)


# In[ ]:





# In[141]:


print('about to create pipeline data and parameters')
model_file         = PipelineData(name='model_file', datastore=default_ds)
model_name         = PipelineParameter("model_name", default_value='wipfli')
model_desc         = PipelineParameter("model_desc", default_value='description')
raw_file_location  = PipelineParameter(name="raw_file_location", default_value='Wipfli/raw_data.csv')

print('creating testing & training data')

training_data  = OutputFileDatasetConfig(name='training_data', destination=(default_ds, 'training_data/{run-id}')).read_delimited_files().register_on_complete(name=  'training_data')
testing_data   = OutputFileDatasetConfig(name='testing_data', destination=(default_ds, 'testing_data/{run-id}')).read_delimited_files().register_on_complete(name=  'testing_data')



# In[137]:


train_model_step = PythonScriptStep(
    name='Get Data and Create Model',
    script_name='training.py',
    arguments =['--raw_file_location', raw_file_location,
                '--training_data', testing_data,
                '--testing_data', testing_data,
                '--model_file', model_file,
                '--model_name', model_name
               ],
    inputs=[],
    outputs=[model_file, training_data, testing_data],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# In[139]:


deploy_file = PipelineData(name='deploy_file', datastore=default_ds)

evaluate_and_register_step = PythonScriptStep(
    name='Evaluate and Register Model',
    script_name='eval.py',
    arguments=[
        '--model_name', model_name,
        '--model_file', model_file,
        '--model_desc', model_desc,
        '--deploy_file', deploy_file,       
    ],
    inputs=[model_file.as_input('model_file'),
            training_data.as_input(name='training_data'),
            testing_data.as_input(name='testing_data')
           ],
    outputs=[ deploy_file],
    compute_target=pipeline_cluster,
    source_directory='./' + experiment_folder,
    allow_reuse=False,
    runconfig=pipeline_run_config
)


# In[140]:


## Create Pipeline Steps
pipeline = Pipeline(workspace=ws, steps=[train_model_step, evaluate_and_register_step])
if run_by_notebook:
    experiment = Experiment(ws, 'AML_Manual_PipelineTraining')
else:
    experiment = Experiment(ws, 'AML_AutoDevOps_PipelineTraining')
run = experiment.submit(pipeline)
run.wait_for_completion(show_output=True)


# In[ ]:


import json

try:
    final_model = Model(ws, model_name)
    final_model_version = final_model.version
except WebserviceException :
    final_model_version = 0
    
print('inital_model_version = ' + str(inital_model_version))
print('final_model_version= ' + str(final_model_version))

status = run.get_status()
run_details = run.get_details()

print((run_details))
print(run_details['runId'])


# In[ ]:


final_model


# In[ ]:


if final_model_version > 0 and (inital_model_version != final_model_version):
    deploy = 'deploy'
    model_details = {
        "name" : final_model.name,
        "version": final_model.version,
        "properties": final_model.properties,
        "nextstep": "deploy"
    }
    print(model_details)
else:
    deploy = 'no'


# In[ ]:


for x in final_model.properties:
    print(x)
    print(final_model.properties[x])


print (final_model.properties)


# In[ ]:


import json
import shutil
import os

outputfolder = 'run_outputs'
os.makedirs(outputfolder, exist_ok=True)

if (final_model_version != inital_model_version):
    print('new model registered')
    with open(os.path.join(outputfolder, 'deploy_details.json'), "w+") as f:
        f.write(str(model_details))
    model_name = prefix
    model_list = Model.list(ws, name=model_name, latest=True)
    model_path = model_list[0].download(exist_ok=True)
    model_file_name = prefix + '.pkl'
    shutil.copyfile(model_file_name,  os.path.join(outputfolder,model_file_name))
    
    #create model.yml file.
    with open(os.path.join(outputfolder, 'model.yml'), "w+") as f:
        f.write('$schema: https://azuremlschemas.azureedge.net/latest/model.schema.json \n')
        f.write('name: ' + model_details['name'] + '\n')
        #f.write('version: ' + str(final_model.version)  + '\n')
        f.write('path: ' + prefix + '.pkl \n')
        f.write('description: Model created from local file. \n')
        if len(final_model.properties) > 0:
            f.write('properties: ')
            f.write(json.dumps(final_model.properties))
            f.write('\n')
            f.write('tags: ')
            f.write(json.dumps(final_model.properties))
            
    
with open(os.path.join(outputfolder, 'run_details.json'), "w+") as f:
    print(run_details)
    f.write(str(run_details))

with open(os.path.join(outputfolder, "run_number.json"), "w+") as f:
    f.write(run_details['runId'])
    
with open(os.path.join(outputfolder, "deploy.txt"), "w+") as f:
    f.write(deploy)


# In[ ]:


from azureml.pipeline.core import PipelineEndpoint

def published_pipeline_to_pipeline_endpoint(
    workspace,
    published_pipeline,
    pipeline_endpoint_name,
    pipeline_endpoint_description="Endpoint to Training pipeline",
):
    try:
        pipeline_endpoint = PipelineEndpoint.get(
            workspace=workspace, name=pipeline_endpoint_name
        )
        print("using existing PipelineEndpoint...")
        pipeline_endpoint.add_default(published_pipeline)
    except Exception as ex:
        print(ex)
        # create PipelineEndpoint if it doesn't exist
        print("PipelineEndpoint does not exist, creating one for you...")
        pipeline_endpoint = PipelineEndpoint.publish(
            workspace=workspace,
            name=pipeline_endpoint_name,
            pipeline=published_pipeline,
            description=pipeline_endpoint_description
        )

if deploy == 'deploy':
    print('deploy Training Pipeline')
    pipeline_endpoint_name = 'Training Pipeline'
    pipeline_endpoint_description = 'Endpoint to Training pipeline'

    published_pipeline = pipeline.publish(name=pipeline_endpoint_name,
                                         description=pipeline_endpoint_description,
                                         continue_on_step_failure=False)

    published_pipeline_to_pipeline_endpoint(
        workspace=ws,
        published_pipeline=published_pipeline,
        pipeline_endpoint_name=pipeline_endpoint_name,
        pipeline_endpoint_description=pipeline_endpoint_description
    )
else:
    print('do not publish pipeline')

