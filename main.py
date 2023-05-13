from flask import Flask, jsonify, request
import os
import tensorflow as tf
import cv2

da_model = tf.keras.models.load_model('tfm/document_analysis/model.h5')
e2e_model = tf.keras.models.load_model('tfm/agnostic_end2end/model.h5')

app = Flask(__name__)

@app.route('/document_analysis', methods=['POST'])
def document_analysis():
    files = request.files
    image = files['image']
    image.save('temp.png')
    image = cv2.imread('temp.png')
    print(image.shape)
    os.remove('temp.png')
    return "ok uploaded"

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
