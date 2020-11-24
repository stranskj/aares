"""
Creating masks


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import numpy as np
import math
import PIL.Image
import PIL.ImageOps

def rough_beamstop(beam_xy, frame_size, beamstop_pixel_radius):
    """
    Creates rough beamstop mask based on position of beam and beamstop size (does not account for poteintialy rectangular pixels)

    :param beam_xy: tuple
    :param frame_size: tuple or np.dsize
    :param pixel_size: tuple
    :param beamstop_size: float
    :return: np.ndarray
    """

    #beamstop_pixel_radius = beamstop_size / pixel_size[0] / 2

    i = np.arange(frame_size[0])
    j = np.arange(frame_size[1])
    I, J = np.meshgrid(j, i)

    circle = (I - beam_xy[0]) ** 2 + (J - beam_xy[1]) ** 2 > beamstop_pixel_radius ** 2
    neck_base_center = (frame_size[1] / 2, 0)
    # beamstop is strip of points within distance from connection of beam center and middle of bottom edge.
    # the connection is ax + by + c =0

    try:
        a = (beam_xy[1]-neck_base_center[1])/(neck_base_center[0]-beam_xy[0])
        b = 1
        c = -a * beam_xy[0] - b * beam_xy[1]
        norm = math.sqrt(a**2 + b**2)
        stick_x = np.abs(a * I + b*J +c ) > beamstop_pixel_radius * norm
    except ZeroDivisionError:
        stick_x = np.abs(I-beam_xy[0]) > beamstop_pixel_radius


    # beamstop is below beam position only
    stick_y = J > beam_xy[1]

    final_mask = np.logical_and(np.logical_or(stick_x, stick_y), circle)
    return final_mask

def draw_mask(mask,output='mask.png'):
    """
    Draws a mask to an image

    :param mask: np.ndarray
    :param output: str
    :return:
    """
    size = mask.shape[::-1]
    databytes= np.packbits(mask, axis=1)
    img = PIL.Image.frombytes(mode='1', size=size, data=databytes)
    flipped = PIL.ImageOps.flip(img)
    flipped.save(output)


def pixel_mask_from_file(h5z_file):
    pixel_mask = h5z_file['entry/instrument/detector/pixel_mask'][:]
    return pixel_mask == 0

def combine_masks(*list_of_masks):
    '''
    Combines list of mask to a single one
    :param list_of_masks:
    :return:
    '''

    if len(list_of_masks) == 1:
        return list_of_masks[0]

    out_mask = list_of_masks[0]
    for mask in list_of_masks[1:]:
        out_mask = np.logical_and(mask,out_mask)

    return out_mask

def test():
    with h5z.FileH5Z('../data/AgBeh_826mm.h5z') as h5f:
        draw(h5f['entry/data/data'][0],'frame.png','5*average')

def main():
    test()

if __name__ == '__main__':
    main()