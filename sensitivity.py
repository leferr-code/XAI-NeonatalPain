"""
GradCAM.py

Author: Craig Pirie
Date: 12/09/2024

This file contains the main functions to calculate the pixel-level ann region-level 
agreement between two explainers. 
"""
import numpy as np


# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0.0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0, dtype="float32")


def calculate_robustness(images, xai_method, n_noise_samples=30):
    """
    Calculates the robustness of an explanation method by quantifying the variability of its attributions under Gaussian noise.

    For each image, the function computes an original explanation mask using the provided explanation method.
    It then generates a specified number of noisy versions of the image using a globally defined function
    `add_gaussian_noise`, computes the corresponding explanation masks for these noisy images, and measures
    the mean squared error (MSE) between the original mask and each noisy mask. The robustness score for each
    image is defined as the average MSE across all noisy samples.

    Parameters
    ----------
    images : iterable
        A collection of images (e.g., numpy arrays) to evaluate.
    xai_method : object
        An explanation method object with an `attribution_mask` method that accepts an image and returns
        an explanation mask.
    n_noise_samples : int, optional
        The number of noisy samples to generate per image. Default is 30.

    Returns
    -------
    list
        A list of robustness scores (average MSE values) corresponding to each input image.

    Notes
    -----
    This function relies on the existence of a globally defined `add_gaussian_noise` function that applies
    Gaussian noise to an image.
    """
    robustness = []

    for image in images:
        image_original = image

        # Compute the original explanation
        mask = xai_method.attribution_mask(image_original)

        mse_values = []

        for _ in range(n_noise_samples):
            image_noise = add_gaussian_noise(image)
            mask_noise = xai_method.attribution_mask(image_noise)

            mse = np.mean((mask - mask_noise) ** 2)
            mse_values.append(mse)

        robustness.append(np.mean(mse_values))

    return robustness
