import sys

import cv2
from skimage import data
from matplotlib import pyplot as plt


def main():
    eye, manual_segmentation, mask = read_input()
    eye = convert_eye_color(eye)
    eye = apply_filters(eye)
    eye, manual_segmentation = apply_mask(eye, manual_segmentation, mask)
    accuracy = calculate_accuracy(eye, manual_segmentation)
    print('accuracy', accuracy, '%')
    show_result(eye, manual_segmentation)


def show_result(eye, manual_segmentation):
    plt.imshow(data.imread(sys.argv[1]), cmap=plt.gray())
    plt.show()
    plt.imshow(eye, cmap=plt.gray())
    plt.show()
    plt.imshow(manual_segmentation, cmap=plt.gray())
    plt.show()


def calculate_accuracy(eye, manual_segmentation):
    height, width, depth = manual_segmentation.shape
    total_count = 0
    hits_count = 0
    for i in range(0, height):
        for j in range(0, width // 4):
            for k in range(0, depth):
                if manual_segmentation[i, j, k] == eye[i, j]:
                    hits_count += 1
                total_count += 1
    accuracy = (hits_count / total_count) * 100
    return accuracy


def apply_mask(eye, manual_segmentation, mask):
    eye = cv2.bitwise_and(eye, eye, mask=mask)
    manual_segmentation = cv2.bitwise_and(manual_segmentation, manual_segmentation, mask=mask)
    return eye, manual_segmentation


def apply_filters(eye):
    eye_filter = cv2.morphologyEx(eye, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    eye_filter = cv2.morphologyEx(eye_filter, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    eye_filter = cv2.morphologyEx(eye_filter, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    eye_filter = cv2.morphologyEx(eye_filter, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    eye = cv2.subtract(eye_filter, eye)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dst = cv2.erode(eye, kernel, iterations=2)
    eye = dst
    return eye


def convert_eye_color(eye):
    eye_green = eye.copy()
    eye_green[:, :, 0] = 0
    eye_green[:, :, 2] = 0
    eye = eye_green
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    return eye


def read_input():
    eye = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    mask = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    manual_segmentation = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)
    return eye, manual_segmentation, mask


if __name__ == "__main__":
    main()
