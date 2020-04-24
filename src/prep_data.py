import cv2

class PrepData():
    def __init__(self):
        pass

    def process_image(self, image_path):
        # Load the image, resize it to 64x64 pixels (the required input
        # spatial dimensions of SmallVGGNet), and store the image in the
        # data list

        image = cv2.imread(image_path)
        raw_image = image.copy()
        image = cv2.resize(image, (64, 64))
        image = image.astype("float") / 255.0
        return raw_image, image