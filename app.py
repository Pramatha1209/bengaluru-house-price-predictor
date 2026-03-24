from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

with open('bangalore_house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_locations')
def get_locations():
    return jsonify({'locations': sorted(locations)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    total_sqft = float(data['total_sqft'])
    bhk        = int(data['bhk'])
    bath       = int(data['bath'])
    location   = data['location'].lower()

    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    price = round(model.predict([x])[0], 2)
    return jsonify({'price': price})

if __name__ == '__main__':
    app.run(debug=True)