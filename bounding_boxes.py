import cv2
import numpy as np

def get_bounding_boxes(image):

    # Perform connected components labeling
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

    # Create an empty list to store bounding boxes
    bounding_boxes = []

    # Iterate through each connected component
    print("Calculating bounding boxes")
    for label in range(1, num_labels):  # Exclude background label 0
        # Get the statistics for the current connected component
        x, y, width, height, _ = stats[label]
        
        # Extract the bounding box coordinates
        bounding_box = (x, y, x + width, y + height)

        # Calculate the height increment (20% of the original height)
        height_increment = int(0.2 * height)

        # Adjust the top and bottom coordinates of the bounding box
        bounding_box_adjusted = (
            int(x),
            int(y - height_increment),         # Adjust top coordinate by subtracting height_increment
            int(x + width),
            int(y + height + height_increment)  # Adjust bottom coordinate by adding height_increment
        )

        print(bounding_box, bounding_box_adjusted)

        # Add the bounding box to the list
        bounding_boxes.append(bounding_box_adjusted)

    # orig_image = cv2.imread('temp.png', cv2.IMREAD_COLOR)
    # #orig_image = cv2.resize(orig_image, (512, 512), interpolation=cv2.INTER_AREA)
    # # Display the bounding boxes
    # for bounding_box in bounding_boxes:
    #     x1, y1, x2, y2 = bounding_box
    #     cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # cv2.imwrite('boxes.png', orig_image)
    return bounding_boxes


def preprocess_image_document_analysis(image_path):
    #img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (255. - img) / 255.
    new_height = 512
    img = cv2.resize(img, (new_height, new_height), interpolation=cv2.INTER_AREA)
    # Add a new dimension using np.expand_dims()
    img = np.expand_dims(img, axis=(0, 3))
    return img

def after_processing(prediction, height, width):
    # quito la dimensión del batch
    prediction = np.squeeze(prediction, axis=0)
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_AREA)
    
    # Perform thresholding to convert the image to binary
    threshold_value = 0.5
    # Binarize the image using the threshold value
    prediction = np.where(prediction > threshold_value, 0, 1).astype(np.uint8)
    return prediction

import itertools
def decode(prediction, i2w):
    out_best = np.argmax(prediction, axis=2)
    #print("out_best-> ", out_best)
    out_best = [k for k, g in itertools.groupby(list(out_best[0]))]
    return [i2w[f"{s}"] for s in out_best if s != len(i2w)]

def preprocess_e2e(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (255. - img) / 255.
    #img = img / 255.
    new_height = 64
    new_width = int(new_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=(0, 3))
    return img

def preprocess_e2e_no(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (255. - img)
    new_height = 64
    new_width = int(new_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def create_json(bounding_boxes, sequences):
    response = []

    # Extract the bounding box coordinates
    # bounding_box = (x, y, x + width, y + height)
    
    for b, s in zip(bounding_boxes, sequences):
        region = {}
        region["fromX"] = b[0]
        region["fromY"] = b[1]
        region["toX"] = b[2]
        region["toY"] = b[3]
        region["seq"] = s
        response.append(region)

    #res = {"prediction": response}
    return response