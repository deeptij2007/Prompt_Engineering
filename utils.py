from typing import List, Dict
import openai
import os
import IPython
from langchain.llms import OpenAI
from dotenv import load_dotenv

def set_openai_params(model="text-davinci-003",\
    temperature=0,\
    max_tokens=256,\
    top_p=1,\
    frequency_penalty=0,\
    presence_penalty=0)-> Dict:
    """
    Description: Setting the parameters of OpenAI API

    Args:
        model (str): name of the specific language model being used
        temeprature (float): parametr that controls the randomness of the generated output
        max_tokens (int): length of the generated output
        top_p (float): parameter that controls the diversity of the generated output
        frequency_penalty (float): Controls the likelihood of the repeated tokens in the generated output
        presence_penalty (float): penlizes the presence of a certain token
    Returns:
        Dict: Dict of openai parameters
    """
    openai_parameters={}
    openai_parameters['model']=model
    openai_parameters['temperature']=temperature
    openai_parameters['max_tokens']=max_tokens
    openai_parameters['top_p']=top_p
    openai_parameters['frequency_penalty']=frequency_penalty
    openai_parameters['presence_penalty']=presence_penalty
    return openai_parameters


def get_response(params: Dict, prompt: str)-> str:
    """
    Description: Getting response from OpenAI API

    Args:
        params (Dict): Dictionary containing the parameters of the OpenAI model
        prompt (str): prompt used to generate the response

    Returns:
        str: response from the OpenAI API
    """
    response = openai.Completion.create(engine = params['model'],
        temperature = params['temperature'],
        max_tokens = params['max_tokens'],
        top_p = params['top_p'],
        frequency_penalty = params['frequency_penalty'],
        presence_penalty = params['presence_penalty'],prompt=prompt)
    return response


