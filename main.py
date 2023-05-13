from flask import Flask, jsonify
import os
import tensorflow as tf

da_model = tf.keras.models.load_model('da.h5')

app = Flask(__name__)

@app.route('/document_analysis', methods=['GET'])
def document_analysis():
    return jsonify({"bounding_box": "This is a bounding box"})

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
