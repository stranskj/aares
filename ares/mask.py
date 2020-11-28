"""
Creating masks


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import ares

import numpy as np
import math
import PIL.Image
import PIL.ImageOps

detectors = {
    'Eiger R 1M' : {
        'shape' : (1065,1030),   # X,Y size of detector
        'Y-mask' : [
            255,256,257,258,
            513,514,515,516,
            771,772,773,774
                     ],
        'X-mask' : [
            255,256,257,258,
            806,807,808,809
                     ],
    }
}

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

def detector_chip_mask(det_type):
    '''
    Generates mask of pixels on chip boundaries

    TODO: User configurable?

    :param det_type: A detector from list "Detectors"
    :return: numpy array of booleans
    '''

    try:
        detector = detectors[det_type]
    except KeyError:
        raise ares.RuntimeErrorUser('Unknown detector type: {}\nAvailable types: {}'.format(det_type, list(detectors.keys())))

    mask = np.ones(detector['shape'],dtype=bool)

    for x in detector['X-mask']:
        mask[x] = False
    for y in detector['Y-mask']:
        mask[:,y] = False

    return mask

def read_mask_from_image(image_in, channel='A', threshold =128, invert=False):
    '''
    Creates frame mask from a image; preferably PNG; light number is
    :param image_in: Input file name
    :param channel: Channel to be used for mask creation
    :param threshold: Values higher than this threshold
    :param invert: Invert the mask
    :return:
    '''

    try:
        with PIL.Image.open(image_in) as img_in:
            if channel == 'A':
                img_mask = img_in.split()[-1]
            elif channel == 'RGB':
                img_mask = img_in.convert(mode='1')
            else:
                raise ares.RuntimeErrorUser('Unsupported channel: {}'.format(channel))
    except FileNotFoundError:
        raise ares.RuntimeErrorUser('File not found: {}'.format(image_in))
    except PIL.UnidentifiedImageError:
        raise ares.RuntimeErrorUser('Unsupported image for mask.')

    flipped = PIL.ImageOps.flip(img_mask)
    mask_np = np.array(flipped.convert(mode='1').getdata()).reshape(img_mask.size[1], img_mask.size[0])

    if invert:
        mask = mask_np > 0
    else:
        mask = mask_np < 1
    return mask

def draw_mask(mask,output='mask.png'):
    """
    Draws a mask to an image

    :param mask: np.ndarray
    :param output: str
    :return:
    """
    size = mask.shape[::-1]
    databytes= np.packbits(np.invert(np.ascontiguousarray(mask)), axis=1)
    img = PIL.Image.frombytes(mode='1', size=size, data=databytes)
    img_rgb = img.convert(mode='RGBA')
    img_rgb.putalpha(img)
    flipped = PIL.ImageOps.flip(img_rgb)
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
#    mask = detector_chip_mask('Eiger R 1M')
#    draw_mask(mask,'chip_mask.png')
    mask = read_mask_from_image('frame_alpha_mask.png', 'A')

    draw_mask(mask,'test_mask.png')


def main():
    test()

if __name__ == '__main__':
    main()