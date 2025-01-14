"""
Background subtraction


@author:     Jan Stransky

@copyright:  2019-2025 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""
import numpy

import aares
import aares.datafiles

import concurrent.futures
import os, logging
import freephil as phil

prog_short_description = 'Performs background subtraction.'

def subtract_reduced(data, background):
    '''
    Subtracts background intensities from data from two compatible files.
    '''

    if data.intensity.shape != background.intensity.shape:
        raise ValueError('Shape of data and background intensities do not match.')

    output = aares.datafiles.Subtract1D(data)

    output.intensity = data.intensity - background.intensity
    output.intensity_sigma = numpy.sqrt(data.intensity_sigma**2 + background.intensity_sigma**2)

    output.add_process(name='Subtraction', description='Background subtraction')
    output.parents = [data.path, background.path]

    output.update_attributes()

    return output