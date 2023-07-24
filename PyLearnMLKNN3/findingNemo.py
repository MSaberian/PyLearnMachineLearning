import numpy as np
import cv2
from knn import KNN

class FindingNemo:
    def __init__(self, train_image):
        self.light_orange = (1, 100, 100)
        self.dark_orange = (60, 255, 255)
        self.light_white = (0, 0, 150)
        self.dark_white = (145, 60, 255)
        self.light_black = (0, 0, 0)
        self.dark_black = (255, 255, 50)
        self.knn = KNN(k=3)
        X_train, Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(X_train, Y_train)
    
    def convert_image_to_dataset(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        pixels_list_hsv = image_hsv.reshape(-1, 3)

        mask_orange = cv2.inRange(image_hsv, self.light_orange, self.dark_orange)
        mask_white = cv2.inRange(image_hsv, self.light_white, self.dark_white)
        mask_black = cv2.inRange(image_hsv, self.light_black, self.dark_black)

        final_mask = mask_orange + mask_white + mask_black

        X_train = pixels_list_hsv / 255
        Y_train = final_mask.reshape(-1,) // 255

        return X_train, Y_train

    def remove_background(self, test_image, background = (0, 0, 0)):
        test_image_hsv = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)
        X_test = test_image_hsv.reshape(-1, 3) / 255
        Y_pred = self.knn.predict(X_test)
        output = Y_pred.reshape(test_image.shape[:2])
        output = output.astype('uint8')
        final_result = cv2.bitwise_and(test_image, test_image, mask= output)
        back = np.ones(final_result.shape[:2])* (1 - output)
        back3 = np.ones(final_result.shape)
        back3[:,:,0] = back * background[0]
        back3[:,:,1] = back * background[1]
        back3[:,:,2] = back * background[2]
        final_result += back3.astype('uint8')

        return final_result