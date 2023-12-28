from math import floor
import numpy as np
import cv2
def pad_matrix(matrix, pad, value):

    def pad_integers(matrix, pad, value):
        return cv2.copyMakeBorder(matrix, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)

    if isinstance(value, bool):
        matrix = matrix.astype(np.uint8)
        padded = pad_integers(matrix.astype(np.uint8), pad, 0)
        return padded.astype(bool)

    return pad_integers(matrix, pad, value)
def convolve(image, k_size, function, mask, features): 
    import cv2
    pad = floor(k_size / 2)
    (image_height, image_width) = image.shape
    result = np.full((image_height, image_width, features), None, dtype=float)
    image_padded = pad_matrix(image, pad, 0)
    mask_padded = pad_matrix(mask, pad, False) 

    def get_neighborhood(image, x_index, y_index, pad):
        return image[y_index - pad: y_index + 1 + pad, x_index - pad: x_index + 1 + pad]

    for y_index in np.arange(pad, image_height + pad):
        for x_index in np.arange(pad, image_width + pad):
            neighborhood = get_neighborhood(image_padded, x_index, y_index, pad)
            mask_neighborhood = get_neighborhood(mask_padded, x_index, y_index, pad)
            result[y_index - pad, x_index - pad] = function(neighborhood, mask_neighborhood)

    return result
