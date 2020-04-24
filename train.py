# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from src.smallvggnet import SmallVGGNet
from src.load_data import LoadData
from src.prep_data import PrepData
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="filename to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="filename to output label binarizer")
ap.add_argument("-p", "--plot", required=True, help="filename to output accuracy/loss plot")
args = vars(ap.parse_args())


# Initialize the data and labels
print("[INFO] loading images...")
loader = LoadData()
prep = PrepData()
data = []
labels = []

# Grab the image paths and label names then randomly shuffle
df = loader.load_data_to_df(is_train=True)[['fpath',
                                                    'Models']].sample(frac=1).reset_index(drop=True)

for index, row in df.iterrows():
    # load the image, resize it to 64x64 pixels (the required input
    # spatial dimensions of SmallVGGNet), and store the image in the
    # data list
    raw, image, image_size = prep.process_image(row['fpath'])
    data.append(image_size)

    # extract the class label from the image path and update the
    # labels list
    label = row['Models']
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=680)

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.build(width=64, height=64, depth=3,
                          classes=len(lb.classes_))

# Initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 1
BS = 32

# Initialize the model and optimizer
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# Train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(f'./output/plots/{args["plot"]}')

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(f'./output/models/{args["model"]}')
f = open(f'./output/labels/{args["label_bin"]}', "wb")
f.write(pickle.dumps(lb))
f.close()