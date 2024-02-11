#!/usr/bin/env python
# coding: utf-8

# In[5]:


import requests
import json
from requests_toolbelt import MultipartEncoder
import os
from typing import Dict, List
import pathlib


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


# ##### Evaluating the classifaction of interior vs exterior images using LLM ##########

# In[140]:


directory = "DamagedHouseImage/DamagedHouse_Exterior_iStock/"

# List all files in the directory
api_key = "gt51b6ea"
files = os.listdir(directory)
img = list()

# Filter out non-image files if needed
image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

image_responses_ext_vs_int = {}

# Loop through the image files and load them one by one
for filename in image_files:
    # Construct the full path to the image
    filepath = os.path.join(directory, filename)
    print(filename)

    post_files_response = post_files(filepath, api_key)
    print(post_files_response)
    prompt_responses = {}
    
    # Example prompts
    prompts = [
        "Describe if the image is either interior or exterior of a house?"
    ]
    
    # Loop through prompts and retrieve text responses
    for prompt in prompts:
        status_code, data = describe(prompt, [filename], api_key)
        prompt_responses[prompt] = data
        #print(f"{status_code}: {data}")
        
        
    # Store the prompt responses in the dictionary with the image filename as the key
    image_responses_ext_vs_int[filename] = prompt_responses


# In[158]:


# Keywords to search for
keywords_exterior = ['exterior', 'aerial']
keywords_interior = ['interior']

# Initialize variables to count relevant responses and total responses for each keyword
relevant_responses_count_exterior = 0
total_responses_count_exterior = 0
relevant_responses_count_interior = 0
total_responses_count_interior = 0
relevant_responses_count_both = 0
total_responses_count_both = 0
# Iterate over each image description
for image_filename, description_data in image_responses_ext_vs_int.items():
    for query, query_response in description_data.items():
        # Extracting the description
        description = query_response['response'][0]['description']
        # Checking if any of the exterior-related keywords is present in the description
        if any(keyword in description.lower() for keyword in keywords_exterior):
            total_responses_count_exterior += 1
            # Checking if the interior keyword is also present in the description
            if any(keyword in description.lower() for keyword in keywords_interior):
                total_responses_count_both += 1
        else:
            # Checking if the interior keyword is present in the description
            if any(keyword in description.lower() for keyword in keywords_interior):
                total_responses_count_interior += 1

# Calculate accuracy for each keyword
total_responses_count = len(image_responses_ext_vs_int.items())
accuracy_exterior = (total_responses_count_exterior / total_responses_count) if total_responses_count > 0 else 0
accuracy_interior = (total_responses_count_interior / total_responses_count) if total_responses_count > 0 else 0
accuracy_both = (total_responses_count_both / total_responses_count) if total_responses_count > 0 else 0

print("Exterior-related responses:")
print("Total responses:", total_responses_count)
print("Relevant responses:", total_responses_count_exterior)
print("Accuracy:", accuracy_exterior)

print("\nInterior-related responses:")
print("Total responses:", total_responses_count)
print("Relevant responses:", total_responses_count_interior)
print("Accuracy:", accuracy_interior)

print("\nResponses containing both exterior and interior keywords:")
print("Total responses:", total_responses_count)
print("Relevant responses:", total_responses_count_both)
print("Accuracy:", accuracy_both)

for image_filename, description_data in image_responses_ext_vs_int.items():
    for query, query_response in description_data.items():
        # Extracting the description
        description = query_response['response'][0]['description']
        # Checking if any of the exterior-related keywords is present in the description
        if any(keyword in description.lower() for keyword in keywords_exterior):
            # Checking if the interior keyword is also present in the description
            if any(keyword in description.lower() for keyword in keywords_interior):
                print("Response containing both exterior and interior keywords:")
                print(description)


# In[167]:


directory = "DamagedHouseImage/DamagedHouse_Interior_iStock/"

# List all files in the directory
api_key = "gt51b6ea"
files = os.listdir(directory)
img = list()

# Filter out non-image files if needed
image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

image_responses_ext_vs_int = {}

# Loop through the image files and load them one by one
for filename in image_files:
    # Construct the full path to the image
    filepath = os.path.join(directory, filename)
    #print(filename)

    post_files_response = post_files(filepath, api_key)
    #print(post_files_response)
    prompt_responses = {}
    
    # Example prompts
    prompts = [
        "Describe if the image is either taken inside the house or outside?"
    ]
    
    # Loop through prompts and retrieve text responses
    for prompt in prompts:
        status_code, data = describe(prompt, [filename], api_key)
        prompt_responses[prompt] = data
        #print(f"{status_code}: {data}")
        
        
    # Store the prompt responses in the dictionary with the image filename as the key
    image_responses_ext_vs_int[filename] = prompt_responses

    
# Keywords to search for
keywords_exterior = ['exterior', 'aerial','outside']
keywords_interior = ['interior','inside','yes']

# Initialize variables to count relevant responses and total responses for each keyword
relevant_responses_count_exterior = 0
total_responses_count_exterior = 0
relevant_responses_count_interior = 0
total_responses_count_interior = 0
relevant_responses_count_both = 0
total_responses_count_both = 0
# Iterate over each image description
for image_filename, description_data in image_responses_ext_vs_int.items():
    for query, query_response in description_data.items():
        # Extracting the description
        description = query_response['response'][0]['description']
        # Checking if any of the exterior-related keywords is present in the description
        if any(keyword in description.lower() for keyword in keywords_exterior):
            total_responses_count_exterior += 1
            # Checking if the interior keyword is also present in the description
            if any(keyword in description.lower() for keyword in keywords_interior):
                total_responses_count_both += 1
        else:
            # Checking if the interior keyword is present in the description
            if any(keyword in description.lower() for keyword in keywords_interior):
                total_responses_count_interior += 1

# Calculate accuracy for each keyword
total_responses_count = len(image_responses_ext_vs_int.items())
accuracy_exterior = (total_responses_count_exterior / total_responses_count) if total_responses_count > 0 else 0
accuracy_interior = (total_responses_count_interior / total_responses_count) if total_responses_count > 0 else 0
accuracy_both = (total_responses_count_both / total_responses_count) if total_responses_count > 0 else 0

print("Exterior-related responses:")
print("Total responses:", total_responses_count)
print("Relevant responses:", total_responses_count_exterior)
print("Accuracy:", accuracy_exterior)

print("\nInterior-related responses:")
print("Total responses:", total_responses_count)
print("Relevant responses:", total_responses_count_interior)
print("Accuracy:", accuracy_interior)

print("\nResponses containing both exterior and interior keywords:")
print("Total responses:", total_responses_count)
print("Relevant responses:", total_responses_count_both)
print("Accuracy:", accuracy_both)

for image_filename, description_data in image_responses_ext_vs_int.items():
    for query, query_response in description_data.items():
        # Extracting the description
        description = query_response['response'][0]['description']
        #if any(keyword in description.lower() for keyword in keywords_interior):
        #print("Response containing both exterior and interior keywords:")
        #print(description)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to the directory containing images
#directory = "DamagedHouseImage/test/"
directory = "DamagedHouseImage/DamagedHouse_Exterior_iStock/"

# List all files in the directory
api_key = "gt51b6ea"
files = os.listdir(directory)
img = list()

# Filter out non-image files if needed
image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

image_responses = {}

# Loop through the image files and load them one by one
for filename in image_files:
    # Construct the full path to the image
    filepath = os.path.join(directory, filename)
    print(filename)
    #status_code, data = summarize('Describe the image.', filename, api_key)

    post_files_response = post_files(filepath, api_key)
    print(post_files_response)
    prompt_responses = {}
    
    # Example prompts
    prompts = [
        "Describe many characteristics of the image?",
        "If the house or building is damaged, what will happen to it if it's left untreated?"
    ]
    
    # Loop through prompts and retrieve text responses
    for prompt in prompts:
        status_code, data = describe(prompt, [filename], api_key)
        prompt_responses[prompt] = data
        #print(f"{status_code}: {data}")
        
    # Store the prompt responses in the dictionary with the image filename as the key
    image_responses[filename] = prompt_responses

    
    
    # Load the image
    image = Image.open(filepath)
    img.append(image)
    
    # Process the image, for example, display it
    #image.show()
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    #plt.plot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')


# In[71]:


for filename, prompt_responses in image_responses.items():
    print("Image:", filename)
    for prompt, response in prompt_responses.items():
        print("Prompt:", prompt)
        print("Text Response:", response)


# In[19]:


img[0]
#plt.plot(img[1])#, plt.imshow(image, cmap='gray'), plt.title('Original Image')


# In[47]:


img[1]


# In[ ]:





# ######  Interior Images ####### 

# In[22]:


import os
from PIL import Image
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Path to the directory containing images
directory = "DamagedHouseImage/DamagedHouse_Interior_iStock/"

# List all files in the directory
api_key = "gt51b6ea"
files = os.listdir(directory)
img_interior = list()

# Filter out non-image files if needed
image_files_interior = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]

image_responses_interior = {}
image_sentiment_interior = {}

# Initialize the SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Loop through the image files and load them one by one
for filename in image_files_interior:
    # Construct the full path to the image
    filepath = os.path.join(directory, filename)
    #print(filename)
    #status_code, data = summarize('Describe the image.', filename, api_key)

    post_files_response = post_files(filepath, api_key)
    #print(post_files_response)
    prompt_responses_interior = {}
    
    # Example prompts
    prompts_interior = [
        "Describe many characteristics of the image?",
        "Describe for each of the item in the house, what is good and what is impacted?",
        "Is the damage really bad? Please describe it how bad it is?"
    ]
    
    # Loop through prompts and retrieve text responses
    for prompt in prompts_interior:
        status_code, data = describe(prompt, [filename], api_key)
        prompt_responses_interior[prompt] = data
        #print(f"{status_code}: {data}")
        
    # Store the prompt responses in the dictionary with the image filename as the key
    image_responses_interior[filename] = prompt_responses_interior
    
    # Load the image
    image = Image.open(filepath)
    img_interior.append(image)
    
    # Get the response for the first prompt
    response = prompt_responses_interior['Describe many characteristics of the image?']['response'][0]['description']
    
    # Calculate sentiment score for the response1
    sentiment_score = sid.polarity_scores(response)['compound']
    response = prompt_responses_interior['Describe for each of the item in the house, what is good and what is impacted?']['response'][0]['description']
    
    # Calculate sentiment score for the response2
    sentiment_score2 = sid.polarity_scores(response)['compound']

    response = prompt_responses_interior['Is the damage really bad? Please describe it how bad it is?']['response'][0]['description']
    
    # Calculate sentiment score for the response2
    sentiment_score3 = sid.polarity_scores(response)['compound']
    
    avg_score = (sentiment_score + sentiment_score2 + sentiment_score3 )/3
    image_sentiment_interior[filename] = (sentiment_score + sentiment_score2 + sentiment_score3 )/3
    # Print or save the sentiment score
    print(f"Sentiment Score for {filename}: {avg_score}")

    # If you want to save the sentiment scores to a file, you can do something like this:
    # with open('sentiment_scores.txt', 'w') as file:
    #     file.write(f"{filename}: {sentiment_score}\n")


# In[23]:


i=0
#for i in range(len(image_responses_interior.items())):
for i in range(3):    
    for filename, prompt_responses_interior in image_responses_interior.items():
        print("Image:", filename)
        for prompt, response in prompt_responses_interior.items():
            print("Prompt:", prompt)
            print("Text Response:", response)
            plt.figure(figsize=(10, 5))
    plt.plot(1, 4), plt.imshow(img_interior[i], cmap='gray'), plt.title('Original Image')

        #i += 1


# In[133]:


for i, (filename, prompt_responses_interior) in enumerate(image_responses_interior.items()):
    print("Image:", filename)
    
    # Access the image from the img_interior array
    image = img_interior[i]
    
    # Display the image
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    # Loop through each prompt and its response
    for prompt, response in prompt_responses_interior.items():
        # Add text annotation to the image with the response
        description = response['response'][0]['description']
        print(description)
        # Add text annotation to the image with the prompt and description
        #plt.text(10, 20, f"{prompt}: {description}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        #plt.text(10, 20, {description }, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Show the image with annotations
    plt.show()


# In[28]:


#img_interior[1]


# In[32]:


# Plot images with sentiment scores in subplots
num_images = 10#len(img_interior)
num_cols = 3  # Number of columns for subplots
num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))

# Flatten axes if needed
if num_rows == 1:
    axes = [axes]

# Iterate through images and sentiment scores
i = 0
for ax, image, filename in zip(axes.flat, img_interior, image_files_interior):
    # Get sentiment score for the current image
    sentiment_score = image_sentiment_interior.get(filename, None)
    print(sentiment_score)
    
    # Plot the image
    ax.imshow(image, cmap='gray')
    
    # Add title with sentiment score
    if sentiment_score is not None:
        ax.set_title(f'Sentiment Score: {sentiment_score}')
    

# Hide any unused subplots
for i in range(num_images, num_rows * num_cols):
    axes.flat[i].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:





# In[42]:


import os
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

def analyze_interior_images(directory: str, api_key: str) -> dict:
    """
    Analyze interior damage images and save sentiment scores for each prompt for each image.

    Args:
    directory (str): Path to the directory containing images.
    api_key (str): API key for accessing the ArchetypeAI API.

    Returns:
    dict: A dictionary containing sentiment scores for each prompt for each image.
    """
    # Initialize the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Initialize dictionaries to store prompt responses and sentiment scores
    image_responses_interior = {}
    image_sentiment_interior = {}

    # Loop through the first 10 image files
    for filename in os.listdir(directory)[:100]:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full path to the image
            filepath = os.path.join(directory, filename)

            # Post the image file and retrieve prompt responses
            post_files_response = post_files(filepath, api_key)
            prompt_responses_interior = {}

            # Example prompts
            prompts_interior = [
                "Describe many characteristics of the image?",
                "Describe for each of the item in the house, what is good and what is impacted?",
                "Is the damage really bad? Please describe it how bad it is?"
            ]
            
            # Loop through prompts and retrieve text responses
            for prompt in prompts_interior:
                status_code, data = describe(prompt, [filename], api_key)
                prompt_responses_interior[prompt] = data

            # Calculate sentiment scores for each prompt response
            sentiment_scores = {}
            for prompt, response_data in prompt_responses_interior.items():
                response = response_data['response'][0]['description']
                sentiment_score = sid.polarity_scores(response)['compound']
                sentiment_scores[prompt] = sentiment_score

            # Calculate average sentiment score for the image
            avg_score = np.mean(list(sentiment_scores.values()))
            image_sentiment_interior[filename] = avg_score

            # Store prompt responses and sentiment scores for the image
            image_responses_interior[filename] = {
                'prompt_responses': prompt_responses_interior,
                'sentiment_scores': sentiment_scores
            }

            # Print sentiment score for the image
            print(f"Sentiment Score for {filename}: {avg_score}")

    return image_responses_interior

def plot_sentiment_distribution(image_responses: dict, prompt: str) -> None:
    """
    Plot the distribution of sentiment scores for a specific prompt.

    Args:
    image_responses (dict): Dictionary containing sentiment scores for each prompt for each image.
    prompt (str): Prompt for which sentiment scores distribution is to be plotted.

    Returns:
    None
    """
    sentiment_scores = []
    for response_data in image_responses.values():
        sentiment_scores.append(response_data['sentiment_scores'][prompt])

    plt.hist(sentiment_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Sentiment Score Distribution for Prompt: {prompt}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Path to the directory containing images
directory = "DamagedHouseImage/DamagedHouse_Interior_iStock/"
api_key = "gt51b6ea"

# Analyze interior images and retrieve sentiment scores for each prompt for each image
image_responses_interior = analyze_interior_images(directory, api_key)

# Plot the distribution of sentiment scores for each prompt
for prompt in image_responses_interior[next(iter(image_responses_interior))]['prompt_responses']:
    plot_sentiment_distribution(image_responses_interior, prompt)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




