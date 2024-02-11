from llamaapi import LlamaAPI
import json
import requests
from requests_toolbelt import MultipartEncoder
import os
from typing import Dict, List
import pathlib
import os
from PIL import Image
import pprint
import matplotlib.pyplot as plt


def get_file_type(filename: str) -> str:
    type_mapper = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        ".mp4": "video/mp4",
    }
    file_extension = pathlib.Path(filename).suffix
    if file_extension in type_mapper:
        return type_mapper[file_extension]
    raise Exception(f"Unknown file extension: {file_extension}")


def post_files(filename: str, api_key: str):
    url = "https://api.archetypeai.dev/v0.3/files"
    auth_headers = {"Authorization": f"Bearer {api_key}"}
    with open(filename, "rb") as file_handle:
        encoder = MultipartEncoder(
            {
                "file": (
                    os.path.basename(filename),
                    file_handle.read(),
                    get_file_type(filename),
                )
            }
        )
        response = requests.post(
            url,
            data=encoder,
            headers={**auth_headers, "Content-Type": encoder.content_type},
        )
        response_data = response.json() if response.status_code == 200 else {}
        return response.status_code, response_data


def summarize(query: str, file_ids: List[str], api_key: str):
    url = "https://api.archetypeai.dev/v0.3/summarize"
    auth_headers = {"Authorization": f"Bearer {api_key}"}
    data_payload = {"query": query, "file_ids": file_ids}
    response = requests.post(url, data=json.dumps(data_payload), headers=auth_headers)
    response_data = response.json() if response.status_code == 200 else {}
    return response.status_code, response_data


def describe(query: str, file_ids: List[str], api_key: str):
    url = "https://api.archetypeai.dev/v0.3/describe"
    auth_headers = {"Authorization": f"Bearer {api_key}"}
    data_payload = {"query": query, "file_ids": file_ids}
    response = requests.post(url, data=json.dumps(data_payload), headers=auth_headers)
    response_data = response.json() if response.status_code == 200 else {}
    return response.status_code, response_data


def do_everything(image):
    api_key = "gt51b6ea"

    PATH = "image.jpg"

    # Need to save image here
    img = Image.open(image)
    img.save(PATH)

    # Post to VQA
    post_files_response = post_files(PATH, api_key)

    prompt_responses = {}

    # Example prompts
    prompts = [
        "Describe many characteristics of the image?",
        "Describe for each of the item in the house, what is good and what is impacted?"
        "Is the damage really bad? Please describe it how bad it is?",
    ]

    # Loop through prompts and retrieve text responses
    for prompt in prompts:
        status_code, data = describe(prompt, [PATH], api_key)
        prompt_responses[prompt] = data

    fullresponse = ""
    for prompt, response in prompt_responses.items():
        fullresponse += response["response"][0]["description"] + " "

    llama = LlamaAPI(
        "LL-xthoWEJGAR2iKroCOBiAYK0Fzvu4dt1FnkXWI0AeS1qcF1Hz1xtuLjLtgosEgTd7"
    )

    api_request_json = {
        "model": "llama-13b-chat",
        "messages": [
            {
                "role": "system",
                "content": """You are an inspector designed to assist homeowners with house-related damage diagnosis for insurance purposes. When provided with a description of an issue within the house based on a picture,
        you should diagnose the problem and offer potential solutions along with estimated costs. Ensure that you are capable of understanding various types of house issues 
        such as plumbing leaks, electrical faults, structural damages, and more. Additionally, you should predict and highlight potential issues likely to arise within a 3-6 month timeframe
        and generate 3 insights as tailored maintenance and protection recommendations for the homeowners based on the individual condition/damage of their house. The response
        to a single house problem should be a comprehensive list of general costs and solutions. Do not ask any questions that require a user response, but make sure the report is extremely detailed, professional, 
        and actionable.""",
            },
            {"role": "user", "content": fullresponse},
        ],
    }

    response = llama.run(api_request_json)

    return response.json()["choices"][0]["message"]["content"]
