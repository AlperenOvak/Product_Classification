from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from preprocess_tr import *
import operator
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


app = Flask(__name__)

# Load Keras model
model = load_model('v2.3_general_class/model-neural-net.keras')
df2 = pd.read_csv('cleaned_data.csv')
tokenizer = Tokenizer(num_words=5000, lower=True) # lower : boolean. Whether to convert the texts to lowercase. , num_words : the maximum number of words to keep, based on word frequency.
tokenizer.fit_on_texts(df2['description'])

classes = ['bilgisayar',
 'cep telefonu',
 'bilgisayar bileşenleri',
 'küçük ev aletleri',
 'çevre birimleri',
 'tüketici elektroniği',
 'yazılım',
 'elektronik',
 'oyun - hobi',
 'aksesuar ürünleri',
 'tüketim malzemeleri',
 'aksesuar & sarf malz.',
 'bilgisayar tablet yazıcı']  # categories

# Utility function to get predictions using Neural Net model
def categoryPredictionNN(description):
    
    description = description.lower()
    description = clean_punctuation(description)
    description = keep_alpha_turkish(description)
    description = remove_stop_words(description)
    description = stemming(description)

    information = description

    sequences = tokenizer.texts_to_sequences([information])
    x = pad_sequences(sequences, maxlen=500)
    prediction = model.predict(x)

    predScores = [score for pred in prediction for score in pred]
    predDict = {}
    for cla, score in zip(classes, predScores):
        predDict[cla] = score

    sortedPredictions = sorted(predDict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    print(sortedPredictions)
    return sortedPredictions

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Endpoint to predict category
@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    
    # Make prediction
    prediction = categoryPredictionNN(description)

    # Get predicted category
    #predicted_category = categories[np.argmax(prediction)]

    return render_template('home.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
