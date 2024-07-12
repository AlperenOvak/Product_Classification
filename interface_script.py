from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load Keras model
model = load_model('v2.3_general_class/model-neural-net.keras')

categories = ['bilgisayar', 'bilgisayar bileşenleri', 'cep telefonu',
       'aksesuar ürünleri', 'tüketici elektroniği',
       'bilgisayar tablet yazıcı', 'elektronik', 'tüketim malzemeleri',
       'aksesuar & sarf malz.', 'oyun - hobi', 'küçük ev aletleri',
       'çevre birimleri', 'yazılım', 'Others']  # categories

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Endpoint to predict category
@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']

    # Make prediction
    prediction = model.predict(np.array([description]))

    # Get predicted category
    predicted_category = categories[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
