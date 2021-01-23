"""
Creating masks


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import ares
import logging

__all__ = []
__version__ = ares.__version__
prog_short_description = 'Generates frame mask'

import numpy as np
import math
import PIL.Image
import PIL.ImageOps
import freephil as phil
import h5z


phil_core = phil.parse('''
file = None
.type = path
#.multiple = True
#.help = File with a custom mask. It can be mask in H5-file, or PNG (than is eqiuvalent to custom.file). If this is H5 or H5Z data file, pixel_mask is extracted.
.help = File used to extract information about the experiment.

output = mask.png
.type = path
.help = Output mask for further usage. It can be of H5 or PNG format

beamstop 
    .help = Automatic beamstop masking. The beamstop is expected to be tilted rectangle with half-circle top.
{
    size = 0
    .type = float
    .help = Size of beamstop for automatic mask in millimeters. Set to 0, if no beamstop masking is required
    .expert_level = 0
    
    tilt = None
    .type = float
    .help = Angular offset of the beamstop from a vertical line in degrees. For SAXSpoint 2.0, vertical position is 8 degrees, this value is then the difference from 8.
    
    origin = None
    .type = ints(size = 2)
    .help = Center of the beamstop in pixels. If None, primary beam position is used. 
    
    semitransparent = None
    .help = Hole cut-out in the mask for primary, in case semitransparent beamstop was used. Pick slightly smaller number than beamstop size.
    .type = float
}

custom
    .help = Custom made mask stored as PNG image. The image has to have pixel-to-pixel size with the detector. A template image can be generated using ares.draw2d
    .multiple = True
    {
    file = None
    .help = File name
    .type = path
    
    channel = R G B *A
    .type = choice
    .help = Image channel to be extracted as a mask
    
    threshold = 128
    .type = int
    .help = If value is higher than the threshold, the pixel is masked out.
    } 
    
pixel_mask = True
    .help = Use pixel mask from data files
    .expert_level = 1
    .type = bool    

detector
.help = Masking parameters related to the detector
{
   
    chip_borders = True
    .help = Exclude border pixels of the chips
    .type = bool
    
    type = *None 'Eiger R 1M' custom
    .help = Specify type of the detector. If None, read from file header. If custom, specify excluded_lines and invalid_pixels.
    .type = choice

    invalid_pixels
    .help= Threshold limits, in which pixels are considered valid
    .expert_level = 1
    {
        min = 0
        .type = int
        max = None
        .type = int
    }
    
    exclude_lines
    .help = lines to be excluded during processing. For example, pixels on chip borders.
    .expert_level = 1
    {
        rows = None
        .type = ints
        columns = None
        .type = ints
    }
}
''')


detectors = {
    'Eiger R 1M' : {
        'shape' : (1065,1030),   # X,Y size of detector
        'y_mask' : [
            255,256,257,258,
            513,514,515,516,
            771,772,773,774
                     ],
        'x_mask' : [
            255,256,257,258,
            806,807,808,809
                     ],
        'range' : (0, int(math.pow(2,31)))
    }
}

def beamstop_hole(beam_xy, hole_pixel_radius, beamstop_mask):
    '''
    Creates hole in beamstop mask for semitransparent beamstops

    :param beam_xy:
    :param frame_size:
    :param hole_pixel_radius:
    :return:
    '''

    i = np.arange(beamstop_mask.shape[0])
    j = np.arange(beamstop_mask.shape[1])
    I, J = np.meshgrid(j, i)

    circle = (I - beam_xy[0]) ** 2 + (J - beam_xy[1]) ** 2 < hole_pixel_radius ** 2

    return np.logical_or(beamstop_mask,circle)

def rough_beamstop(beam_xy, frame_size, beamstop_pixel_radius, tilt = None):
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

def detector_chip_mask(shape = None, x_mask= [], y_mask = [], det_type = None):
    '''
    Generates mask of pixels on chip boundaries

    :param det_type: A detector from list "Detectors"
    :return: numpy array of booleans
    '''

    if det_type is not None:
        try:
            detector = detectors[det_type]
            shape = detector['shape']
            x_mask = detector['x_mask']
            y_mask = detector['y_mask']
        except KeyError:
            raise ares.RuntimeErrorUser('Unknown detector type: {}\nAvailable types: {}'.format(det_type, list(detectors.keys())))

    mask = np.ones(shape,dtype=bool)

    for x in x_mask:
        mask[x] = False
    for y in y_mask:
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

def composite_mask(work_phil, file_header=None):
    '''
    Produces composite mask based on input

    :param work_phil: input parameters
    :type work_phil: phil.extract
    :param file_header: File header to get the experimental information
    :return:
    '''

    import os

    masks = []

    # Get file header
    if (work_phil.file is not None) and (file_header is None):
        try:
            file_header = h5z.SaxspointH5(work_phil.file)
        except OSError:
            raise ares.RuntimeErrorUser('Wrong file format. Please provide data file.')

    if file_header is None:
        raise ares.RuntimeErrorUser('Missing file header. Please provide a data file.')


    # Pixel mask from the file
    if work_phil.pixel_mask:
        ares.my_print('Extracting pixel mask from the datafile...')
        masks.append(pixel_mask = pixel_mask_from_file(file_header))

    # Masks from PNG files
    for msk in work_phil.custom:
        ares.my_print('Creating mask from file {}'.format(msk.file))
        masks.append(read_mask_from_image(msk.file,msk.channel,msk.treshold))

    # Chip borders
    if work_phil.detector.chip_borders:
        ares.my_print('Creating mask for chip borders...')
        if work_phil.detector.type == 'None':
            detector_type = file_header['entry/instrument/detector/description'].item().decode()
        else:
            detector_type = work_phil.detector.type

        if detector_type == 'custom':
            ares.my_print('Custom detector type. Excluding:')
            if work_phil.detector.exclude_lines.rows is None:
                logging.warning('No pixel row defined to be excluded')
            else:
                ares.my_print('Rows:    '+', '.join(work_phil.detector.exclude_lines.rows))
            if work_phil.detector.exclude_lines.columns is None:
                logging.warning('No pixel column defined to be excluded')
            else:
                ares.my_print('Columns: '+', '.join(work_phil.detector.exclude_lines.rows))

            detector_type = None

        masks.append(detector_chip_mask(
            file_header.frame_size,
            work_phil.detector.exclude_lines.rows,
            work_phil.detector.exclude_lines.columns,
            detector_type
            ))

    # Beamstop

    if work_phil.beamstop.size > 0:
        ares.my_print('Creating beamstop mask...')
        if work_phil.beamstop.origin is not None:
            origin = work_phil.beamstop.origin
        else:
            origin = file_header.beam_center_px

        beamstop_mask = rough_beamstop(
            beam_xy=origin,
            frame_size=file_header.frame_size,
            beamstop_pixel_radius=work_phil.beamstop.size/file_header.pixel_size[0]/2000,
            tilt=work_phil.beamstop.tilt
            )

        if work_phil.beamstop.semitransparent is not None:
            ares.my_print('Applying cut-out for transparent beamstop...')
            beamstop_mask = beamstop_hole(
                beam_xy=origin,
                hole_pixel_radius=work_phil.beamstop.semitransparent/file_header.pixel_size[0]/2000,
                beamstop_mask=beamstop_mask
            )

        masks.append(beamstop_mask)

    ares.my_print('Merging masks...')
    return combine_masks(*masks)

class JobMask(ares.Job):
    """
    Run class based on generic Ares run class
    """

    def __set_meta__(self):
        '''
        Sets various package metadata
        '''
        import os, sys
        self._program_short_description = prog_short_description

        self._program_name = os.path.basename(sys.argv[0])
        self._program_version = __version__

    def __worker__(self):
        '''
        The actual programme worker
        :return:
        '''
        mask = composite_mask(self.params)

        ares.my_print('Saving the mask in the file: {}'.format(self.params.output))
        draw_mask(mask,self.params.output)

    def __set_system_phil__(self):
        '''
        Settings of CLI arguments. self._parser to be used as argparse.ArgumentParser()
        '''
        self._system_phil = phil_core

    def __argument_processing__(self):
        '''
        Adjustments of raw input arguments. Modifies self._args, if needed

        '''
        pass

    def __help_epilog__(self):
        '''
        Epilog for the help

        '''
        pass



def test():
#    mask = detector_chip_mask('Eiger R 1M')
#    draw_mask(mask,'chip_mask.png')
#    mask = read_mask_from_image('frame_alpha_mask.png', 'A')

    test_phil = phil.parse('''
    pixel_mask = False
    file = ../data/10x60s_826mm_010Frames.h5
    ''')

    work_phil = phil_core.fetch(test_phil)
    work_extract = work_phil.extract()
    composite_mask(work_extract)
    pass
#    draw_mask(mask,'test_mask.png')


def main():
    job = JobMask()
    return job.job_exit

if __name__ == '__main__':
    main()