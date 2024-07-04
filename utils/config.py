import yaml
from easydict import EasyDict as edict
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from pathlib import Path
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import logging
import os
import requests

LLM_ENV = yaml.safe_load(open('config/llm_env.yml', 'r'))


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'  # Reset to default color

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

def set_env_var():
        
        os.environ["KEYCLOAK_BASE_URL"] = "https://iamfw.home-np.oocl.com"
        os.environ["KEYCLOAK_CLIENT_ID"] = "mro_kf_chui"
        os.environ["KEYCLOAK_CLIENT_SECRET"] = "HAjxgG4EwcnYkCj7D68oticjW9w75rSO"
        os.environ["KEYCLOAK_REALM"] = "oocl-dev" 
        os.environ["KEYCLOAK_TOKEN_URL"] = "https://iamfw.home-np.oocl.com" + '/auth/realms/' + "oocl-dev" + '/protocol/openid-connect/token' 
        os.environ["ENV"] = "local" 
        os.environ["MLOP_HOST"] = "https://mlop-llm-gateway-dev.home-np.oocl.com" 
        os.environ["SUPP_DATA_HOST"] = "https://iacb-supp-data-dev.a.home-np.oocl.com"

def get_llm(config: dict):
    """
    Returns the LLM model
    :param config: dictionary with the configuration
    :return: The llm model
    """
    if 'temperature' not in config:
        temperature = 0
    else:
        temperature = config['temperature']
    if 'model_kwargs' in config:
        model_kwargs = config['model_kwargs']
    else:
        model_kwargs = {}
    if config['type'] == 'OpenAI':
        if LLM_ENV['openai']['OPENAI_ORGANIZATION'] == '':
            return ChatOpenAI(temperature=temperature, model_name=config['name'],
                              openai_api_key=config.get('openai_api_key', LLM_ENV['openai']['OPENAI_API_KEY']),
                              openai_api_base=config.get('openai_api_base', 'https://api.openai.com/v1'),
                              model_kwargs=model_kwargs)
        else:
            return ChatOpenAI(temperature=temperature, model_name=config['name'],
                              openai_api_key=config.get('openai_api_key', LLM_ENV['openai']['OPENAI_API_KEY']),
                              openai_api_base=config.get('openai_api_base', 'https://api.openai.com/v1'),
                              openai_organization=config.get('openai_organization', LLM_ENV['openai']['OPENAI_ORGANIZATION']),
                              model_kwargs=model_kwargs)
    elif config['type'] == 'Azure':
        return AzureChatOpenAI(temperature=temperature, azure_deployment=config['name'],
                        openai_api_key=config.get('openai_api_key', LLM_ENV['azure']['AZURE_OPENAI_API_KEY']),
                        azure_endpoint=config.get('azure_endpoint', LLM_ENV['azure']['AZURE_OPENAI_ENDPOINT']),
                        openai_api_version=config.get('openai_api_version', LLM_ENV['azure']['OPENAI_API_VERSION']))

    elif config['type'] == 'Google':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(temperature=temperature, model=config['name'],
                              google_api_key=LLM_ENV['google']['GOOGLE_API_KEY'],
                              model_kwargs=model_kwargs)


    elif config['type'] == 'HuggingFacePipeline':
        device = config.get('gpu_device', -1)
        device_map = config.get('device_map', None)

        return HuggingFacePipeline.from_model_id(
            model_id=config['name'],
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": config['max_new_tokens']},
            device=device,
            device_map=device_map
        )
    
    elif config['type'] == 'AzureViaGateway':
        set_env_var()
        getMlopKey()
        llmClient = AzureChatOpenAI(
                deployment_name="gpt-4-turbo-deploy",
                azure_endpoint=os.getenv("MLOP_HOST"),
                openai_api_version="2024-02-01",
                temperature=0.0,
                openai_api_key=os.getenv("MLOP_ACCESS_TOKEN"),
            )
        return llmClient
    
    else:
        raise NotImplementedError("LLM not implemented")


def load_yaml(yaml_path: str, as_edict: bool = True) -> edict:
    """
    Reads the yaml file and enrich it with more vales.
    :param yaml_path: The path to the yaml file
    :param as_edict: If True, returns an EasyDict configuration
    :return: An EasyDict configuration
    """
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        if 'meta_prompts' in yaml_data.keys() and 'folder' in yaml_data['meta_prompts']:
            yaml_data['meta_prompts']['folder'] = Path(yaml_data['meta_prompts']['folder'])
    if as_edict:
        yaml_data = edict(yaml_data)
    return yaml_data


def load_prompt(prompt_path: str) -> PromptTemplate:
    """
    Reads and returns the contents of a prompt file.
    :param prompt_path: The path to the prompt file
    """
    with open(prompt_path, 'r') as file:
        prompt = file.read().rstrip()
    return PromptTemplate.from_template(prompt)


def validate_generation_config(base_config, generation_config):
    if "annotator" not in generation_config:
        raise Exception("Generation config must contain an empty annotator.")
    if "label_schema" not in generation_config.dataset or \
            base_config.dataset.label_schema != generation_config.dataset.label_schema:
        raise Exception("Generation label schema must match the basic config.")


def modify_input_for_ranker(config, task_description, initial_prompt): ##modified to format the prompt
    modifiers_config = yaml.safe_load(open('prompts/modifiers/modifiers.yml', 'r'))
    task_desc_setup = load_prompt(modifiers_config['ranker']['task_desc_mod'])
    init_prompt_setup = load_prompt(modifiers_config['ranker']['prompt_mod'])

    llm = get_llm(config.llm)
    task_llm_chain = LLMChain(llm=llm, prompt=task_desc_setup)
    task_result = task_llm_chain(
        {"task_description": task_description})
    mod_task_desc = task_result['text']
    logging.info(f"Task description modified for ranking to: \n{mod_task_desc}")

    prompt_llm_chain = LLMChain(llm=llm, prompt=init_prompt_setup)
    prompt_result = prompt_llm_chain({"prompt": initial_prompt, 'label_schema': config.dataset.label_schema})
    mod_prompt = prompt_result['text']
    logging.info(f"Initial prompt modified for ranking to: \n{mod_prompt}")
    fixed_prompt = f"""Classification Quality Evaluation Instruction:

You are tasked with evaluating the quality of a generated sample based on a given user prompt. The evaluation should be carried out by assigning one of the following classification labels: ['1', '2', '3', '4', '5'], where '1' indicates the lowest quality and '5' indicates the highest quality.

Specifically, you need to assess how well the generated sample adheres to the user prompt and meets the provided criteria. Here is the prompt for the generative instruction:
{initial_prompt}
Your evaluation should consider the following aspects:
1. **Adherence to User Input:** How well does the generated prompt follow the provided user input?
2. **Relevance and Specificity:** Are the suggested questions relevant to the topic of vessel shipping and specific enough to be helpful?
3. **Clarity and Engagement:** Are the questions clear and engaging, contributing to a productive discussion?

Assign a classification label ['1', '2', '3', '4', '5'] based on the overall quality of the generated prompt, with '1' being poor quality and '5' being excellent quality.
"""

    return mod_prompt, mod_task_desc, fixed_prompt


def override_config(override_config_file, config_file='config/config_default.yml'):
    """
    Override the default configuration file with the override configuration file
    :param config_file: The default configuration file
    :param override_config_file: The override configuration file
    """

    def override_dict(config_dict, override_config_dict):
        for key, value in override_config_dict.items():
            if isinstance(value, dict):
                if key not in config_dict:
                    config_dict[key] = value
                else:
                    override_dict(config_dict[key], value)
            else:
                config_dict[key] = value
        return config_dict

    default_config_dict = load_yaml(config_file, as_edict=False)
    override_config_dict = load_yaml(override_config_file, as_edict=False)
    config_dict = override_dict(default_config_dict, override_config_dict)
    return edict(config_dict)
