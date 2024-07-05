
# Prompt Optimization Examples

This document provides practical examples of using the AutoPrompt pipeline across various scenarios. It focuses on movie review and chat moderation tasks to demonstrate the flexibility and effectiveness of the AutoPrompt framework.


1. Dataset Generation
2. Initial Prompt Generation
3. [Rating Movie Reviews (Scoring task)](#rating-movie-reviews-scoring-task)
4. [Generating Movie Reviews (Generation task)](#generating-movie-reviews-generation-task)
5. [Single Topic Moderation](#single-topic-moderation)
6. [Multi-Topic Moderation (Prompt squeezing task)](#multi-topic-moderation-prompt-squeezing)

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


### Generating Movie Reviews (Generation task):
Here, we aim to generate good (insightful and comprehensive) movie reviews from scratch. The initial prompt might look something like this: 
 - Initial prompt: “Write a good and comprehensive movie review about a specific movie.”
 - Task description: “Assistant is a large language model that is tasked with writing movie reviews.”

This time, we'll need to use the `run_generation_pipeline.py` to initiate a generation run. This pipeline is different from but builds on the classification pipeline in our earlier examples.

The generation pipeline starts by taking the initial prompt and modifying it for a scoring task, similar to the scoring example above. Once it establishes a robust estimtor for high-quality content, in this instance movie reviews, it runs the generation pipeline without the need for human annotation. 

To facilitate this, two distinct input config files are employed: `config/config_diff/config_ranking.yml`, and `config/config_diff/config_generation.yml`.

Note that the `annotator` section in the generation config yaml file remains empty: 
```
annotator:
    method : ''
```

#### Run Example

Run the generation pipeline with appropriate arguments: 
```bash
> python run_generation_pipeline.py \
    --prompt "Write a good and comprehensive movie review about a specific movie." \
    --task_description "Assistant is a large language model that is tasked with writing movie reviews."
```

As the pipeline runs, the user will be prompted to annotate ranking examples of movie reviews. The final output will be a calibrated prompt for the generation task.

### Single Topic Moderation:

In this example, we aim to monitor user interactions on an Enterprise's chat platform to moderate (filter out) any unsolicited advertisements. This ensures a focused and relevant communication environment.

The initial prompt could be as follows:

- Initial prompt: “Assess whether the message contains advertising. Answer 'Yes' or 'No'.”
 - Task description: “As a moderation expert at FabricFantasia, an online store selling clothes, you meticulously review customer inquiries and support tickets.”

#### Run Example
For the moderation, update the `label_schema` in `config/config_default.yml`
```
dataset:
    label_schema: ["Yes", "No"]
```
And then execute the pipeline with the specified input parameters:
```bash
> python run_pipeline.py \
    --prompt "Assess whether the message contains advertising. Answer 'Yes' or 'No'." \
    --task_description "As a moderation expert at FabricFantasia, an online store selling clothes, you meticulously review customer inquiries and support tickets."
```
Please follow the same annotation and monitoring procedures as shown in the previous examples.

### Multi Topic Moderation (Prompt squeezing task):
In this example, our goal is to monitor user interactions on an enterprise's chat platform and moderate (filter out) any problematic topics, including disclosing personal information, deceptive practices, spam, illegal activities, conflict of interest, and off-topic content.

The initial prompt could be structured as follows:

- Initial prompt: “Does this message contain any problematic topics such as disclosing personal information, deceptive practices, spam, illegal activities, conflict of interest, or off-topic content? Respond with 'Yes' or 'No'.”
 - Task description: “As a moderation expert at FabricFantasia, an online store selling clothes, you meticulously review customer inquiries and support tickets.”


#### Run Example
In a multi-topic moderation setting, we use various moderation rules to annotate a sample. Each rule is evaluated independently, and the outcomes are combined to generate the final labels. We employ an LLM annotator to avoid time-intensive manual annotation.

This task utilizes two distinct input configuration files: `config/config_default.yml`, used previously, and `config/config_diff/config_batch_classification.yml`, which specifies the individual moderation rules, the policy for aggregating results, and LLM configuration. The available aggregation policies are 'exist', 'majority', and 'all'. The 'exist' and 'all' policies are suited for scenarios with 'Yes' or 'No' labels, while the 'majority' policy assigns the final label based on the most frequently occurring outcome across the rules.

In our case, it can look like this:
```
dataset:
    label_schema: ["Yes", "No"]

annotator:
    method : 'llm_batch'
    config:
        instructions: 
        ['Does the message disclosure sensitive personal information? Answer Yes or No', 
         'Does the message involve deceptive practices? Answer Yes or No',
         'Is this message an example of spam? Answer Yes or No',
         'Does the message reference or promote any illegal activities? Answer Yes or No',
         'Does the message come from someone with a potential conflict of interest? Answer Yes or No',
         'Is this message completely irrelevant to the services or products offered? Answer Yes or No'
         ]
        aggregation_mode: 'exist'  #'majority',  'exist', or 'all'. exist/all is working only in case label_schema: ["Yes", "No"]!
        estimator_config:
            num_workers: 2
            prompt: 'prompts/predictor/prediction.prompt'
            mode: 'annotation'
            mini_batch_size: 1
            llm:
              type: 'OpenAI'
              name: 'gpt-4-1106-preview'
```

Also, update the `label_schema` in `config/config_default.yml`
```
dataset:
    label_schema: ["Yes", "No"]
```

#### Run Example
As before, we'll use the `run_pipeline.py` to initiate a multi-topic moderation run.
```bash
> python run_pipeline.py \
    --batch_config_path "config/config_diff/config_batch_classification.yml" \
    --prompt "Assess whether the message contains any of the following problematic topics:  disclosing personal information, deceptive practices, spam, illegal activities, conflict of interest, off-topic content. Answer 'Yes' if it does or 'No' otherwise." \
    --task_description "As a moderation expert at FabricFantasia, an online store selling clothes, you meticulously review customer inquiries and support tickets."
```
Please follow the same annotation and monitoring procedures as shown in the previous examples.
