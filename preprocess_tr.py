# This is my library for my preprocessing stage. 
# Since I have created lots of subfolders for each version/model, I need to create a general library that I can use it for all of my model training notebook
# One can find most of general preprocessing methods for text. I configire most of them for Turksih text content.
# I hope it is clear, ENJOY


import re
from zeyrek import MorphAnalyzer  

# Dictionary for stop words
dictionary = {}

# Function to load stop words from a file
def load_stop_words(filepath):
    global dictionary
    with open(filepath, 'r', encoding='utf-8') as fdict:
        for line in fdict:
            if line and line[0] in ['0', '1']:
                freq, word = line.strip().split()
                dictionary[word] = int(freq)

dictionary = {}

# Function to check if a word is a stop word
def is_stop_word(word):
    return word in dictionary

# Function to remove stop words from a sentence
def remove_stop_words(sentence):
    try:
        load_stop_words("./dosyalar/stop_words.txt")
    except FileNotFoundError:
        print("Stop words file not found")
    words = sentence.split()
    return " ".join([word for word in words if not is_stop_word(word)])

# Function to keep only alphanumeric characters (Turkish)
def keep_alpha_turkish(sentence):
    alpha_sent = re.sub('[^a-zA-ZçğıöşüÇĞİÖŞÜ ]+', '', sentence)
    return alpha_sent.strip()

# Function for stemming using Zeyrek MorphAnalyzer
def stemming(sentence):
    sentence = lower(sentence)
    zeyrek = MorphAnalyzer()
    words = sentence.split()
    result = []
    for word in words:
        try:
            result.append(zeyrek.lemmatize(word)[0][1][0])
        except:
            result.append(word)
    return " ".join(result)

# Function to clean punctuation
def clean_punctuation(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#|-|®|™]', '', sentence)
    cleaned = re.sub(r'[.|,|)|:|;|"|(|\|/|-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def lower(sentence):
    return sentence.replace('I', 'ı').lower()

def preprocess_turkish(sentence):
    sentence = lower(sentence)
    cleaned_sentence = clean_punctuation(sentence)
    without_stopwords = remove_stop_words(cleaned_sentence)
    alpha_turkish = keep_alpha_turkish(without_stopwords)
    stemmed_sentence = stemming(alpha_turkish)
    
    return stemmed_sentence
