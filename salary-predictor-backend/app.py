from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import math
import json

app = Flask(__name__)
CORS(app)
with open(r"Linear_Model.pkl", 'rb') as f:
    model = pickle.load(f)

with open(r"Dec_Tree.pkl", 'rb') as f:
    dt = pickle.load(f)

with open(r"Rand_Forest.pkl", 'rb') as f:
    rf = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    model_name=data['model_name']
    if model_name== 'Linear':
        prediction = model.predict([features])
        prediction=prediction[0] 
    elif model_name== 'Decision_Tree':
        prediction = dt.predict([features])
    elif model_name== 'Random_Forest':
        prediction = rf.predict([features])

    return jsonify(prediction= prediction[0]*prediction[0])

if __name__ == '__main__':
    app.run(debug=True)