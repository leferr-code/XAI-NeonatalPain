"""
GradCAM.py

Author: Craig Pirie
Date: 12/09/2024

This file contains the main functions to calculate the pixel-level ann region-level 
agreement between two explainers. 
"""
import numpy as np


def calculate_infidelity(images, model, xai_method, n_samples=30, noise_std=0.1):
    """
    Calculates the infidelity of an explanation method by measuring the discrepancy between input perturbations 
    and corresponding changes in the model's output.

    For each image, the function first obtains the original model prediction and explanation mask. It then 
    generates several perturbed versions of the image by subtracting a small Gaussian noise and computes the 
    perturbed predictions. The infidelity is quantified as the average squared difference between the change in 
    the model output and the change in the explanation (weighted by the perturbation).

    Parameters
    ----------
    images : iterable
        A collection of images to evaluate.
    model : object
        A model that implements a `predict` method. The method should accept an image with an added batch 
        dimension (e.g., using `unsqueeze(0)`) and return a prediction convertible to a numpy array.
    xai_method : object
        An explanation method that implements an `attribution_mask` method, which returns an explanation 
        mask for a given image.
    n_samples : int, optional
        The number of perturbed samples to generate per image. Default is 30.
    noise_std : float, optional
        The standard deviation of the Gaussian noise used for perturbation. Default is 0.1.

    Returns
    -------
    list
        A list of infidelity scores (one per image), each representing the average squared discrepancy 
        between the change in the model output and the change in the explanation due to the applied perturbations.

    Notes
    -----
    This function assumes that the images, model predictions, and explanation masks are appropriately scaled 
    and compatible for the computation of perturbation effects.
    """
    infidelity = []

    for image in images:
        # Original image and model prediction
        image_original = image
        pred_original = model.predict(image_original.unsqueeze(0)).cpu().detach().numpy()

        # Compute the original explanation mask
        mask = xai_method.attribution_mask(image_original)

        total_infidelity = 0

        for _ in range(n_samples):
            # Generate a small Gaussian perturbation
            perturbation = np.random.normal(0, noise_std, image.shape[:-1])
            perturbed_image_aux = np.zeros(image.shape, dtype='float32')

            # Perturb the image channel-wise and clip values to [0, 1]
            for i in range(image.shape[-1]):
                perturbed_image_aux[:, :, i] = np.clip(image[:, :, i] - perturbation, 0, 1)

            # Convert the perturbed image for model prediction
            perturbed_image = perturbed_image_aux
            pred_perturbed = model.predict(perturbed_image.unsqueeze(0)).cpu().detach().numpy()

            # Compute the effect of the perturbation on the model output and explanation
            output_perturbation = np.sum(pred_perturbed - pred_original)
            input_perturbation = np.sum(mask * perturbation)

            total_infidelity += (input_perturbation - output_perturbation) ** 2

        # Average the infidelity over the number of samples
        infidelity.append(total_infidelity / n_samples)

    return infidelity
