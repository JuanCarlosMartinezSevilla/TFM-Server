from flask import Flask, jsonify, request
import os
import tensorflow as tf
import cv2
from bounding_boxes import get_bounding_boxes, preprocess_image_document_analysis, after_processing

da_model = tf.keras.models.load_model('tfm/document_analysis/model.h5')
e2e_model = tf.keras.models.load_model('tfm/agnostic_end2end/model.h5')

app = Flask(__name__)



@app.route('/document_analysis', methods=['POST'])
def document_analysis():
    files = request.files
    image = files['image']
    image.save('temp.png')
    image = cv2.imread('temp.png', cv2.IMREAD_COLOR)
    # Get original image size
    height = image.shape[0]
    width = image.shape[1]

    image = preprocess_image_document_analysis(image)
    #image.save('temp_gray.png')

    prediction = da_model.predict(image)
    prediction = after_processing(prediction)

    #prediction_to_save = (prediction * 255)
    #cv2.imwrite('prediction.png', prediction_to_save)

    bounding_boxes = get_bounding_boxes(prediction, height, width)
    os.remove('temp.png')
    return "EVERYTHING WORKS"

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
