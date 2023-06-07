# Streamlit app for news articles

# Fabian Schubö
# June 2023

################## Libraries ##################  
import streamlit as st
import pickle
import time
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline
import spacy as sp

################## Title and input ##################  
st.title("Text Classification")

# Request input text for instant classification
user_input = st.text_area("Enter text:",key="input", height=300)

################## Text preprocessing ##################   

def text_cleaning(text):
    clean_text = re.sub('<.*?>', '', text) # Remove HTML tags
    clean_text = re.sub(r'[^\w\s]|_|[0-9]', '', clean_text) # Remove punctuation and sp. characters
    clean_text = clean_text.lower() # lowercasing
    clean_text = re.sub(r'\s+', ' ', clean_text) # remove extra whitespace
    clean_text = clean_text.strip()
    return clean_text

def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    #return lemmas
    return ' '.join(lemmas)

# Define a dictionary for character replacement
character_replacements = {
    'ä': 'ae',
    'ö': 'oe',
    'ü': 'ue',
    'ß': 'ss'
}

# Function for character replacement
def replace_characters(text):
    for old_char, new_char in character_replacements.items():
        text = text.replace(old_char, new_char)
    return text

# Function for stopword removal
def remove_stopwords(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    #return tokens
    return ' '.join(tokens)
    
################## Language detection ##################    

# Open model for language detection
with open('my_lang_model.pkl', 'rb') as file:
    loaded_lang_model = pickle.load(file)

# Detect language
if user_input:
    # Call text_cleaning function for cleaning user input
    clean_text = text_cleaning(user_input)
    language = loaded_lang_model.predict([clean_text])    

################## Topic classification ##################

# Open models for topic classification
with open('de_class_svm_model.pkl', 'rb') as file:
    loaded_model_de = pickle.load(file)
with open('my_en_model.pkl', 'rb') as file:
    loaded_model_en = pickle.load(file)

# Create a placeholder using st.empty
output = st.empty()    

# Classify topic
if user_input:
    if language[0]=="German":
        
        # Load the German language model
        nlp = sp.load("de_core_news_sm")
        
        # Call functions for preprocessing
        lemmatized_text = lemmatize_text(clean_text)
        replaced_text = replace_characters(lemmatized_text)
        removed_text = remove_stopwords(replaced_text)
        
        # Apply classification
        topic = loaded_model_de.predict([removed_text])
        
    elif language[0]=="English":
        
        # Load the English language model
        nlp = sp.load("en_core_web_sm")
        
        # Call functions for preprocessing
        lemmatized_text = lemmatize_text(clean_text)
        replaced_text = replace_characters(lemmatized_text)
        removed_text = remove_stopwords(replaced_text)
        
        # Apply classification
        topic = loaded_model_en.predict([removed_text])
        
    elif language[0]=="French":
        topic = ["NA"]
        # Call text cleaning function to clean translated text
        # clean_translated_text = text_cleaning([translated_text])
        # topic = loaded_model_en.predict([clean_translated_text]) # check if this works
        
    else:
        topic = ["NA"]
        


################## Output ##################

# Laguage

if user_input:
    output = st.empty()
    
    output.write(f"### Language: ")
    time.sleep(0.1)
    
    output.write(f"### Language: .")
    time.sleep(0.1)

    output.write(f"### Language: ..")
    time.sleep(0.1)

    output.write(f"### Language: ...")
    time.sleep(0.2)

    output.write(f"### Language: {language[0]}")

# Topic
    output = st.empty()
    
    output.write(f"### Topic: ")
    time.sleep(0.1)

    output.write(f"### Topic: .")
    time.sleep(0.1)

    output.write(f"### Topic: ..")
    time.sleep(0.1)

    output.write(f"### Topic: ...")
    time.sleep(0.2)

    output.write(f"### Topic: {topic[0]}")
    
# Translation
#if input_language == "available":


################## Translation ##################   

input_language = ""
if user_input:
    
    if language[0]=="German":
        lang_model='Helsinki-NLP/opus-mt-de-en'
        input_language="available"

    elif language[0]=="French":
        lang_model='Helsinki-NLP/opus-mt-fr-en'
        input_language="available"
        
    else:
        input_language="NA"
    
    if input_language == "available":
        
        output = st.empty()
    
        output.write(f"### Translating")
        output.write(f"### Translating.")

        # Create a translation pipeline with the Helsinki-NLP/opus-mt-de-en model
        translator = pipeline('translation', model=lang_model)
        
        output.write(f"### Translating..")
        
        # Use the pipeline to translate the text
        translated_text = translator(user_input, max_length=90000)[0]['translation_text']
        output.write(f"### Translating...")
        output.write(f"### Translation:\n")
        output = st.empty()
        output.write(translated_text)
