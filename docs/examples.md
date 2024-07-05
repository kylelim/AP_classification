
# Prompt Optimization Examples

This document provides proposed workflow for the repository.


1. Dataset Generation
2. Initial Prompt Generation
3. Auto Prompting for improving prompt quality for classification task



### Generating Dataset for auto prompting pipeline

In this example, we aim to create training and testing dataset based on the labelled dataset.

Required Resources: Annotated dataset, with no header and two column: question and corresponding answer.

#### Steps to Run Example

1. Goes to DScreator.ipynb

2. Fill up the path of the annotated dataset in the jupyter notebook
```
excel_data = pd.read_excel('path pf your dataset here', header = None, names=['text', 'label'])
```

3. Specify the batch size for training set in the jupyter notebook
```
batch_size = 60 ##Specify here

df.batch_id = [_//batch_size for _ in range(len(df))]
```

4. Complete and obtain the dataset for auto prompting.

### Generate initial prompt for auto prompting:

In this example, we want to generate an well-structured initial prompt for auto prompting

1. Goes to initialPromptCreator.ipynb

2. Set up the environment for LLM gateway here
```
def getMlopKey():
    url = os.getenv("KEYCLOAK_TOKEN_URL")
    client_id = os.getenv("KEYCLOAK_CLIENT_ID")
    client_secret = os.getenv("KEYCLOAK_CLIENT_SECRET")
    header = {"Request-client": "IAM_PORTAL", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    response = requests.post(url, headers=header, data=data)
    responseJson = response.json()
    access_token = responseJson['access_token']
    os.environ['MLOP_ACCESS_TOKEN'] = access_token

def set_env_var(): #to be filled
        
        os.environ["KEYCLOAK_BASE_URL"] = "" 
        os.environ["KEYCLOAK_CLIENT_ID"] = ""
        os.environ["KEYCLOAK_CLIENT_SECRET"] = ""
        os.environ["KEYCLOAK_REALM"] = ""
        os.environ["KEYCLOAK_TOKEN_URL"] = ""
        os.environ["ENV"] = "local"
        os.environ["MLOP_HOST"] = ""


def get_llm_client():
    set_env_var()
    getMlopKey()
    llmClient = AzureChatOpenAI(
                deployment_name="",
                azure_endpoint=os.getenv("MLOP_HOST"),
                openai_api_version="2024-02-01",
                temperature=0.8,
                openai_api_key=os.getenv("MLOP_ACCESS_TOKEN"),
    )
    
    return llmClient
```

3. Input user enquiry
Based on the need, input the user enquiry.

4. Obtain the initial prompt
    Please note that initial prompt might require manual modification to make it conform to the requirement of the business use case


### Auto prompting for improving prompt quality for classification task
In this example, we aim to demonstrate the workflow for auto prompting.

To begin with, the following is required:
- Initial Prompt: Prompt that needs to be improved
- Task Description: Decribe the task in concise way
- Training and testing dataset


#### Steps to Run Example

1. Configure your labels by editing `config/config_default.yml`. Modify the `label_schema` in the `dataset` section to include the label schemas.

```
dataset:
    name: 'dataset'
    records_path: null
    initial_dataset: '' #specify the path of training set here
    initial_testset: '' #specify the path of testing set here
    label_schema: ["Yes", "No"] #specify the label schema here
```

2. Goes to run_pipeline.py and fill in the initial prompt, task description and max_iteration
```
# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--basic_config_path', default='./config/config_default.yml', type=str, help='Configuration file path')
parser.add_argument('--batch_config_path', default='',
                    type=str, help='Batch classification configuration file path')
parser.add_argument('--prompt',
                    default="""
**#Initial prompt to be provided here**
                    """,
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--task_description',
                    default='**task description to be provided here**',
                    required=False, type=str, help='Describing the task') #Task description is only used for updating the prompt
...
parser.add_argument('--num_steps', default=10, type=int, help='Number of iterations') **##Max no. of iteration can be specified here**
```

4. After completion, the pipeline outputs a **refined (calibrated) prompt** tailored for the task and a reference **benchmark** with challenging samples. 

- The framework automatically saves the benchmark, run log, and a checkpoint file (which stores the state of the optimization, enabling seamless continuation from a previous run) in a default `dump` path, adjustable with the `--output_dump` command line argument.
- Note that the steps above are relevant to all classification and generation tasks. See the following examples for more use cases. 
