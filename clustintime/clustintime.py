#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustintime is a python3 library meant to apply clustering algorithms over spatio-temporal fMRI data

It supports ``.nii.gz`` files as well as ``.txt`` files.

It requires python 3.6 or above, as well as the modules:
    - `numpy`
    - `matplotlib`
    - `pandas`
"""


import clustering as clus

# Libraries
import numpy as np
import processing as proc
import visualization as vis
from nilearn.input_data import NiftiMasker
from nilearn.masking import apply_mask
from os.path import exists
from pathlib import Path

INFOMAP = 'infomap'
NOT_INFOMAP = 'not_infomap'

VALID_ALGORITHMS = (
    INFOMAP,
    NOT_INFOMAP
)


PROC_DOUBLE = 'double'
PROC_THRESHOLD = 'thr'
PROC_RSS = 'rss'
PROC_WINDOW = 'window'

VALID_PROCESSING_ARGS = (
    PROC_DOUBLE,
    PROC_THRESHOLD,
    PROC_RSS,
    PROC_WINDOW
)


def validate_data_file(data_file: str) -> None:
    if not isinstance(data_file, str):
        raise ValueError('data_file should be str.')

    if not exists(data_file):
        raise ValueError('data_file does not exist.')

    if not Path(data_file).is_file():
        raise ValueError('data_file is not a file.')


def preprocess_data(
        data,
        processing,
        window_size,
        near,
        thr,
        contrast,
        task,
        TR
):

    correlation_map = np.nan_to_num(np.corrcoef(data))
    indexes = range(correlation_map.shape[0])

    if processing not in VALID_PROCESSING_ARGS:
        return correlation_map, indexes

    correlation_map, indexes = proc.preprocess(
        corr_map=correlation_map,
        analysis=processing,
        data=data,
        window_size=window_size,
        near=near,
        thr=thr,
        contrast=contrast,
        task=task,
        TR=TR,
    )

    return correlation_map, indexes





def clustintime(
    data_file,
    mask_file,
    *timings_file,
    processing=None,
    timings=True,
    window_size=1,
    near=1,
    thr=95,
    contrast=1,
    TR=0.5,
    algorithm=INFOMAP,
    thr_infomap=90,
    n_clusters=7,
    save_maps=False,
    saving_dir=".",
    prefix="",
):
    """
    Run main workflow of clustintime.

    Estimates the functional connectivity of the data, processes the data and performs the clustering analysis,
    It returns summary on screen as well as the graphs corresponding to the data, the processing and the results.


    Parameters
    ----------
    data_file : str or path
        Fullpath to the data to be analyzed.
    mask_file : str or path
        Fullpath to the corresponding mask.
    *timings_file : str or path
        path to `.txt` files containing timings of the analyzed task.
    processing : str, optional
        Desired type of processing, the options are `None`, `double`, `thr`, `RSS`, `window`.
        The default is `None`.
    timings : bool, optional
        Boolean that indicates whether the timings of the task are known or not.
        If `True`, a path to the timings must be specified in `timings_file`.
        The default is True.
    window_size : int, optional
        Window size for the `window` processing option.
        The default is 1.
    near : int, optional
        Nearby time-points to select when performing `RSS` processing.
        The default is 1.
    thr : int, optional
        Threshold percentile for the `thr` processing.
        The default is 95.
    contrast : float, optional
        RAnge of values for the correlation matrixes.
        The default is 1.
    TR : float, optional
        TR of the data.
        The default is 0.5.
    algorithm : str, optional
        Desired clustering algorithm for the analysis, the options are `infomap` and `KMeans`.
        The default is "infomap".
    thr_infomap : int, optional
        Threshold percentile to binarize the matrix in the infomap algorithm.
        The default is 90.
    n_clusters : int, optional
        Desired number of groups for the K Means algorithm.
        The default is 7.
    save_maps : bool, optional
        Boolean that indicates whether the results must be saved or not.
        The default is False.
    saving_dir : str or path, optional
        Fullpath to the saving directory.
        The default is ".".
    prefix: str
        Prefix for the saved outcomes

    Returns
    -------
    None.

    """

    validate_data_file(data_file)

    NiftiMasker(mask_img=mask_file).fit(data_file)
    data = apply_mask(data_file, mask_file)

    task = {idx: np.loadtxt(filename) for idx, filename in enumerate(timings_file)}

    correlation_map, indexes = preprocess_data(
        data=data,
        processing=processing,
        window_size=window_size,
        near=near,
        thr=thr,
        contrast=contrast,
        task=task,
        TR=TR
    )




    if algorithm == "infomap":
        # corr_map_2 = correlation_map.copy()
        correlation_map, labels = clus.Info_Map(
            correlation_map,
            indexes,
            thr_infomap,
            nscans=correlation_map.shape[0],
            task=task,
            TR=TR,
            saving_dir=saving_dir,
            prefix=prefix,
        )
        # vis.plot_two_matrixes(corr_map_2, correlation_map, "Original correlation map", "Binary correlation map")
    elif algorithm == "KMeans":
        labels = clus.K_Means(
            corr_map=correlation_map,
            indexes=indexes,
            nscans=correlation_map.shape[0],
            n_clusters=n_clusters,
            TR=TR,
            task=task,
            saving_dir=saving_dir,
            prefix=prefix,
        )

    if save_maps:
        clus.generate_maps(labels, saving_dir, data_file, mask_file, prefix)
    vis.Dyn(correlation_map, labels, output_file=f"{saving_dir}/dyneusr_{algorithm}.html")


if __name__ == '__main__':
    clustintime(
        data_file='../samples/S41vol.pySPFM.LARS.STC.beta_99.GROUPS.nii.gz',
        mask_file='../samples/S41vol.mask.nii.gz'
    )
