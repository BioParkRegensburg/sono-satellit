#!/usr/bin/env python
# coding: utf-8

# In[34]:


# Import packages

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[35]:


# Add image path

path = "./extract_test_images"
img_folder = os.listdir(path)
img_folder = [path + "/" + file for file in img_folder]


# In[36]:


img_folder


# In[37]:


def create_curve_mask(img_shape, top_curve, thickness, shift_y=0):

    """
    Create a rectangular mask
    """

    h, w = img_shape

    # define horizontal boundaries
    x_min = int(w * 0.33)
    x_max = int(w * 0.66)

    y_top = np.min(top_curve[:, 1]) + shift_y

    # Rectangle
    rect = np.array([
        [x_min, y_top],
        [x_max, y_top],
        [x_max, y_top + thickness],
        [x_min, y_top + thickness]
    ], dtype=np.int32)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rect], 255) # image, co-ordinates, white colour mask

    return mask, rect


# In[38]:


def generate_shifted_masks(image_path, thickness=20, iterations=60):

    # Grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape

    top_curve = np.array([ # Initiate top mask
        [180, 45],
        [230, 55],
        [280, 65],
        [340, 72],
        [400, 65],
        [460, 55],
        [520, 45]
    ], dtype=np.int32)

    all_mask_values = [] # Pixel values of the masks

    cols = 3
    rows = int(np.ceil(iterations / cols))

    plt.figure(figsize=(15, rows * 5))

    for i in range(iterations):

        # Shift each mask downward by 5 pixels
        shift_y = i * 5

        mask, smooth_curve = create_curve_mask(
            img.shape,
            top_curve,
            thickness,
            shift_y
        )

        # Extract pixels inside the mask
        pixel_values = img[mask == 255]

        all_mask_values.append(pixel_values)

        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        green = np.zeros_like(overlay)
        green[:] = (0, 255, 0)

        alpha = 0.3

        overlay[mask == 255] = cv2.addWeighted(
            overlay,
            1 - alpha,
            green,
            alpha,
            0
        )[mask == 255]

        # Draw curve boundary
        cv2.polylines(
            overlay,
            [smooth_curve],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2
        )

        plt.subplot(rows, cols, i + 1)

        plt.imshow(cv2.cvtColor(overlay, cv2.IMREAD_GRAYSCALE))

        plt.title(f"Mask {i+1}")

        # plt.axis("off")

    plt.tight_layout()
    plt.show()

    return all_mask_values


# In[39]:


def array_to_dict(array):

    """
    Create a dictionary with pixel values sum and standard deviation
    """

    mask_values = dict()

    for i, value in enumerate(array):
        mask_values[i+1] = [np.sum(array[i]), np.std(array[i])]

    return mask_values


# In[40]:


def plot_brightness_and_std(mask_stats):

    """
    Plot the graphs from the extracted sum and standard deviation values of the pixels
    """

    keys = mask_stats.keys()

    brightness = [mask_stats[k][0] for k in keys]
    std_values = [mask_stats[k][1] for k in keys]


    plt.figure(figsize=(10, 4))

    plt.plot(keys, brightness, marker='o')
    plt.title("Brightness vs Mask Index")
    plt.xlabel("Mask Index")
    plt.ylabel("Brightness (Sum)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(keys, std_values, marker='o', color='orange')
    plt.title("Standard Deviation vs Mask Index")
    plt.xlabel("Mask Index")
    plt.ylabel("Standard Deviation")
    plt.grid(True)
    plt.show()


# In[41]:


all_values = generate_shifted_masks(
    image_path="./VAT 6.PNG"
)


# In[42]:


a_dict = array_to_dict(all_values)


# In[43]:


plot_brightness_and_std(a_dict)

