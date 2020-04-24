# import the necessary packages
import argparse
import cv2

from src.prep_data import PrepData
from src.load_data import LoadData

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True, help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True, help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28, help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28, help="target spatial dimension height")
args = vars(ap.parse_args())

loader = LoadData()
prep = PrepData()

# load the input image and resize it to the target spatial dimensions
output, image = prep.process_image(f'./images/{args["image"]}')
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = loader.load_model(args["model"])
lb = loader.load_labels(args["label_bin"])

# make a prediction on the image
preds = model.predict(image)

# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)
