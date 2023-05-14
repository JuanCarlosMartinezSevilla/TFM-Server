from flask import Flask, jsonify, request
import os, json
import tensorflow as tf
import cv2
from bounding_boxes import get_bounding_boxes, preprocess_image_document_analysis, after_processing, preprocess_e2e, decode, preprocess_e2e_no
from bounding_boxes import create_json
da_model = tf.keras.models.load_model('tfm/document_analysis/model.h5')
e2e_model = tf.keras.models.load_model('tfm/agnostic_end2end/model.h5')

app = Flask(__name__)


def e2e(image, bounding_boxes):
    sequences = []
    for idx, box in enumerate(bounding_boxes):
        #bounding_box = (x, y, x + width, y + height)
        e2e_img = image[box[1]:box[3], box[0]:box[2]]
        # e2e_no = preprocess_e2e_no(e2e_img)
        # cv2.imwrite(f"e2e{idx}.png", e2e_no)
        e2e_img = preprocess_e2e(e2e_img)
        #print(f"Processed image: {e2e_img.shape}")
        prediction = e2e_model.predict(e2e_img)
        #print(f"Prediction: {prediction.shape}")

        # Load the JSON file as a dictionary
        with open('tfm/agnostic_end2end/i2w.json', 'r') as json_file:
            i2w = json.load(json_file)

        pred = decode(prediction, i2w)
        #print(len(pred))
        #print(pred)
        # i2w conversion
        # y_pred = [[i2w[int(i)] for i in b if int(i) != -1] for b in prediction]
        #print(pred)
        sequences.append(pred)
    return sequences

@app.route('/document_analysis', methods=['POST'])
def document_analysis():
    files = request.files
    image = files['image']
    image.save('temp.png')
    image = cv2.imread('temp.png', cv2.IMREAD_COLOR)
    # Get original image size
    height = image.shape[0]
    width = image.shape[1]

    image_da = preprocess_image_document_analysis(image)
    #image.save('temp_gray.png')

    prediction = da_model.predict(image_da)
    prediction = after_processing(prediction, height, width)

    #prediction_to_save = (prediction * 255)
    #cv2.imwrite('prediction.png', prediction_to_save)

    bounding_boxes = get_bounding_boxes(prediction)

    sequences = e2e(image, bounding_boxes)
    result = create_json(bounding_boxes, sequences)
    # Convert numpy.int32 values to int
    # result = convert_numpy_int32(result)
    # result = convert_numpy_int64(result)
    os.remove('temp.png')
    return jsonify(result)

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
