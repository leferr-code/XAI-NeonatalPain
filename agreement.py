"""
GradCAM.py

Author: Craig Pirie
Date: 12/09/2024

This file contains the main functions to calculate the pixel-level ann region-level 
agreement between two explainers. 
"""
import numpy as np


def get_top_k_pixels(importance_map, k_percent=10):
    """
    Extracts a binary mask of the top k percent of pixels from an importance map.

    This function computes the threshold value corresponding to the top k percent of pixel
    importance values in the input map. It returns a binary mask where pixels with importance
    values greater than or equal to the threshold are marked as 1, and all other pixels are 0.

    Parameters
    ----------
    importance_map : numpy.ndarray
        A 2D array representing pixel importance scores.
    k_percent : int, optional
        The percentage of top pixels to select (0-100). Default is 10.

    Returns
    -------
    numpy.ndarray
        A binary mask of the same shape as `importance_map`, with 1 indicating top k percent
        important pixels and 0 otherwise.
    """
    # Flatten the map and sort the pixels by importance
    flat_map = importance_map.flatten()
    threshold_value = np.percentile(flat_map, 100 - k_percent)
    top_k_mask = importance_map >= threshold_value

    return top_k_mask.astype('int')


def feature_agreement(mask_1, mask_2):
    """
    Computes the feature agreement between two binary masks using the Intersection over Union (IoU).

    This function calculates the similarity between two feature masks by computing the ratio 
    of the number of overlapping (intersection) pixels to the total number of pixels present 
    in either mask (union). A higher value indicates greater overlap between the two masks.

    Parameters
    ----------
    mask_1 : numpy.ndarray
        A binary mask representing the first set of features.
    mask_2 : numpy.ndarray
        A binary mask representing the second set of features.

    Returns
    -------
    float
        The feature agreement score, defined as the ratio of the sum of the intersection 
        of the masks to the sum of their union.
    """
    # Compute the Intersection
    intersection = np.logical_and(mask_1, mask_2)
    union = np.logical_or(mask_1, mask_2)

    # Calculate Feature Agreement
    feature_agreement = np.sum(intersection) / np.sum(union)

    return feature_agreement


def get_region_importances(masks_dict, heatmap, scale=True):
    """
    Computes the importance of specified regions using a heatmap and corresponding masks.

    This function iterates over a dictionary of region masks, computing the total importance
    for each region by summing the heatmap values where the mask is active (i.e., greater than 0).
    Optionally, it normalizes the computed importance by the number of active pixels in the region.

    Parameters
    ----------
    masks_dict : dict
        A dictionary mapping region names (str) to their corresponding binary masks (numpy.ndarray).
    heatmap : numpy.ndarray
        A 2D array representing pixel-wise importance scores.
    scale : bool, optional
        If True, scales the region importance by the number of pixels in the region.
        Default is True.

    Returns
    -------
    dict
        A dictionary where keys are region names and values are the computed importance scores.
    """
    region_importance = {}

    for region, mask in masks_dict.items():
        # Calculate the total importance for the region
        total_importance = np.sum(heatmap * (mask > 0))

        if scale:
            # Calculate the number of pixels in the region
            num_pixels = np.sum(mask > 0)
            # Scale importance by the number of pixels if scale=True
            region_importance[region] = total_importance / num_pixels if num_pixels > 0 else 0
        else:
            # If scaling is not requested, just use the total importance
            region_importance[region] = total_importance

    # Return the unsorted region importance dictionary
    return region_importance


def calculate_region_agreement(importance_dict1, importance_dict2, k, return_agreed_set=False):
    """
    Computes the agreement between two region importance dictionaries by comparing the top-k regions.

    This function sorts two dictionaries of region importance scores in descending order and selects the
    top k regions from each. It then calculates the intersection of these top-k region sets and computes
    an agreement score as the fraction of shared regions. Optionally, it can also return the set of agreed regions.

    Parameters
    ----------
    importance_dict1 : dict
        Dictionary mapping region names to their importance scores (first explainer).
    importance_dict2 : dict
        Dictionary mapping region names to their importance scores (second explainer).
    k : int
        Number of top regions to consider from each dictionary.
    return_agreed_set : bool, optional
        If True, returns a tuple containing the agreement score and the set of common top-k regions.
        Default is False.

    Returns
    -------
    float
        Agreement score between 0 and 1 representing the fraction of shared top-k regions.
    set, optional
        Set of regions that are common to both top-k lists (only returned if return_agreed_set is True).
    """
    # Sort each dictionary by importance and get the top k regions
    top_k_regions_1 = set(sorted(importance_dict1, key=importance_dict1.get, reverse=True)[:k])
    top_k_regions_2 = set(sorted(importance_dict2, key=importance_dict2.get, reverse=True)[:k])

    # Calculate the intersection of the top k regions
    agreed_regions = top_k_regions_1.intersection(top_k_regions_2)

    # Calculate the agreement score
    agreement_score = len(agreed_regions) / k

    if return_agreed_set:
        return agreement_score, agreed_regions
    else:
        return agreement_score
