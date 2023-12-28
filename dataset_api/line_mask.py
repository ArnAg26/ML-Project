import math
import numpy as np

def radians_to_degrees(rads):
    return rads * math.pi / 180.0

class LineMask: 
    def __init__(self, k_size, angle, orthogonal_length=3):
        if orthogonal_length % 2 != 1 or orthogonal_length < 1:
            raise ValueError('Orthogonal line mask length must be a positive odd number')

        angle = angle % 180.0
        acute = angle % 90.0
        quarter_size = math.ceil(k_size / 2)
        quarter = np.zeros((quarter_size, quarter_size), dtype=bool)
        diagonal_difference = abs(45.0 - acute)
        rise = math.tan(radians_to_degrees(45.0 - diagonal_difference))
        for i in range(0, quarter_size):
            quarter[quarter_size - round(rise * i) - 1, i] = 1
        mask = np.zeros((k_size, k_size), dtype=bool)
        mask[:quarter_size, quarter_size - 1:] = quarter # Q1
        mask[quarter_size - 1:, :quarter_size] = np.rot90(quarter, 2) # Q3
        if 45.0 < angle <= 135.0:
            mask = np.rot90(np.fliplr(mask), -1)
        if angle > 90.0:
            mask = np.fliplr(mask)
        orthogonal_radius = math.floor(orthogonal_length / 2)
        center_mask = np.zeros((k_size, k_size), dtype=bool)
        center_mask[quarter_size - 1 - orthogonal_radius: quarter_size + orthogonal_radius,
                    quarter_size - 1 - orthogonal_radius: quarter_size + orthogonal_radius] = True

        self.mask = mask
        self.orthogonal_mask = np.logical_and(np.rot90(mask), center_mask)

def generate_line_mask_list(k_size, resolution):
    steps = math.floor(180.0 / resolution)
    mask_list = list()
    for i in range(0, steps):
        mask_list.append(LineMask(k_size, resolution * i))

    return mask_list
