import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from predict.predict.run import TextPredictionModel

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_text():
    model = TextPredictionModel.from_artefacts(r"C:\Users\anhki\Documents\EPF\5A\From Poc to Prod\poc-to-prod-capstone\poc-to-prod-capstone\train\data\artefacts\2023-01-06-11-39-27")
    text = "text"

    model.predict(text)
    return text


if __name__ == '__main__':
    app.run(debug=True)





