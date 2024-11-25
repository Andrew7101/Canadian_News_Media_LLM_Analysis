#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 18:29:07 2024

@author: jeongwoohong
"""

import os
import glob
import re
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from striprtf.striprtf import rtf_to_text
import google.generativeai as genai


api_key = os.getenv('GEMINI_API_KEY')


if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")


genai.configure(api_key=api_key)


temperature = 0.85 

generation_config = {
    "temperature": temperature,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 50,
}

def initialize_model(model_name):
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

model_sequence = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
current_model_index = 0
model = initialize_model(model_sequence[current_model_index])


gemini_1_0_pro_429_errors = 0
max_429_errors = 5  


stop_api_calls = False


rtf_directory = '/Users/jeongwoohong/Desktop/school/ECON 424/A5/424_F2024_PC5_Newspaper_data_v1/'


rtf_files = glob.glob(os.path.join(rtf_directory, '*.rtf'))

articles = []


date_patterns = [
    re.compile(r'\b(\d{1,2})\s(January|February|March|April|May|June|July|August|September|October|November|December)\s(19\d{2}|20\d{2})\b'),
    re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s(\d{1,2}),\s(19\d{2}|20\d{2})\b'),
    re.compile(r'\b(19\d{2}|20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12][0-9]|3[01])\b'),
    re.compile(r'\b(\d{1,2})[-/](0[1-9]|1[0-2])[-/](19\d{2}|20\d{2})\b')
]


def split_articles(text):
    article_splits = re.split(r'\nDocument [^\n]*\n', text)
    return [article.strip() for article in article_splits if article.strip()]


def find_date_in_content(text):
    for pattern in date_patterns:
        match = pattern.search(text)
        if match:
            date_str = " ".join(match.groups())
            try:
                if len(match.groups()) == 3:
                    date = datetime.strptime(date_str, '%d %B %Y').date()
                elif ',' in date_str:
                    date = datetime.strptime(date_str, '%B %d, %Y').date()
                else:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                return date
            except ValueError:
                continue
    return None


for file_path in rtf_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        rtf_content = file.read()
        text_content = rtf_to_text(rtf_content)
        articles_in_file = split_articles(text_content)
        
        for article in articles_in_file:
            date = find_date_in_content(article)
            if date:
                articles.append({'date': date, 'content': article})
            else:
                print(f"No date found in one article within file: {file_path}")


df = pd.DataFrame(articles)


prompt_market = (
    "You are to evaluate the following article and determine if it is pro-markets (pro-capitalism) "
    "or pro-government intervention. Respond in valid JSON format with a single key-value pair: "
    '{"result": "0"} for pro-markets and {"result": "1"} for pro-government intervention. '
    "Provide only the JSON object as your response, with no additional text or formatting."
)


def evaluate_article(content, prompt):
    global model
    global current_model_index
    global gemini_1_0_pro_429_errors
    global stop_api_calls
    try:
        max_content_length = 100000  
        if len(content) > max_content_length:
            content = content[:max_content_length]
    
        full_prompt = prompt + "\n\n" + content
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(full_prompt)
        output = response.text.strip()
        print(f"Raw output for current article: '{output}'")
    
        import json
        try:
            if not output.endswith('}'):
                output += '}'
            if not output.startswith('{'):
                output = '{' + output
            result = json.loads(output)
            value = result.get('result')
            if value == '0' or value == 0:
                return 0
            elif value == '1' or value == 1:
                return 1
            else:
                print("Invalid value in JSON.")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    except Exception as e:
        print(f"Error processing article: {e}")
        if "429 Resource has been exhausted" in str(e):
            if current_model_index >= len(model_sequence) - 1:
                if model.model_name == "gemini-1.0-pro":
                    gemini_1_0_pro_429_errors += 1
                    if gemini_1_0_pro_429_errors > max_429_errors:
                        print("Maximum retries with 'gemini-1.0-pro' reached. Stopping API calls.")
                        stop_api_calls = True
                        return None
                else:
                    current_model_index += 1
                    model = initialize_model(model_sequence[current_model_index])
                    print(f"Quota exhausted. Switching model to '{model.model_name}'.")
            else:
                current_model_index += 1
                model = initialize_model(model_sequence[current_model_index])
                print(f"Quota exhausted. Switching model to '{model.model_name}'.")
            return evaluate_article(content, prompt)
        else:
            return None

market_attitudes = []
dates = []


for idx, row in df.iterrows():
    if stop_api_calls:
        print("Stopping API calls due to repeated quota exhaustion.")
        break
    date = row['date']
    content = row['content']
    market_attitude = evaluate_article(content, prompt_market)
    if market_attitude is not None:
        market_attitudes.append(market_attitude)
        dates.append(date)
    else:
        print(f"Skipping article at index {idx} due to invalid output.")
    time.sleep(7)


results_df = pd.DataFrame({
    'date': dates,
    'market_attitude': market_attitudes,
})


results_df['date'] = pd.to_datetime(results_df['date'])
results_df = results_df.sort_values('date')

market_trend = results_df.groupby('date')['market_attitude'].mean()

plt.figure(figsize=(12, 6))
plt.plot(market_trend.index, market_trend.values, marker='o')
plt.title('Attitude Toward Government Intervention Over Time')
plt.xlabel('Date')
plt.ylabel('Attitude (0 = Pro-Markets, 1 = Pro-Government Intervention)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
