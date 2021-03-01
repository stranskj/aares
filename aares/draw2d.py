"""
Draw frame


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import aares
import sys,os
import freephil as phil

__all__ = []
__version__ = aares.__version__
prog_short_description = 'Draws the frame into a PNG file'

import numpy
import math
import PIL.Image
import PIL.ImageOps
import matplotlib.cm
import h5z

phil_core = phil.parse("""
input = None
.type = path
.help = Input file to be drawn

paths = None
.type = path
.multiple = True
.help = Not implemented

frame = None
.type = str
.help = Frame or range of frames to be drawn. If None, an average of all frames is drawn

min = 0
.type = str
.help = Minimum threshold for the color map. A number, string with multiple of 'min', 'max', 'median' or 'average'. For example: '3*median'

max = 5*median
.type = str
.help = Minimum threshold for the color map. A number, string with multiple of 'min', 'max', 'median' or 'average'. For example: '3*median'

output = frame.png
.type = path
.help = Output file name

color_map = *jet hot
.type = choice
.help = Coloring scheme in the output

""")


def draw(frame, fiout, Imax= '2*median', Imin=0, cmap='jet'):
    '''
    Draws frame to a file

    :param frame:
    :param fiout:
    :param Imax:
    :param Imin:
    :return:
    '''

    threshold = [0,0]

    for i,tres in enumerate(list([Imin, Imax])):
        if isinstance(tres, str):
            splited = tres.split('*')
            if splited[-1] == 'median':
                val = numpy.median(frame)
            elif splited[-1] == 'average':
                val = numpy.average(frame)
            elif splited[-1] == 'max':
                val = numpy.nanmax(frame)
            elif splited[-1] == 'min':
                val = numpy.nanmin(frame)

            else:
                val = float(tres)

            if len(splited) == 2:
                threshold[i] = float(splited[0]) * val
            else:
                threshold[i] = val
        else:
            threshold[i] = float(tres)

    Imin, Imax = sorted(threshold)

    aares.my_print('''Used parameters:
    min: {min:.1f}
    max: {max:.1f}'''.format(min=Imin, max=Imax))

    normalized = (frame - Imin) / (Imax - Imin)

    normalized[normalized>1] = 1
    normalized[normalized < 0] = 0

    imshow_kwargs = {
        'vmax': threshold[1],
        'vmin': threshold[0],
        'cmap': 'RdYlBu',
        'extent': (0, frame.shape[0], 0 , frame.shape[1]),
    }


    colormap = matplotlib.cm.get_cmap(cmap)
    #colormap.
    #img = PIL.Image.fromarray(numpy.uint8(matplotlib.cm.gist_earth(normalized)*255))

    img = PIL.Image.fromarray(numpy.uint8(colormap(normalized) * 255))
    flipped = PIL.ImageOps.flip(img)
    flipped.save(fiout)

    # fig, ax = plt.subplots()
    # ax.imshow(frame, **imshow_kwargs)
    # ax.set_axis_off()
    # plt.savefig(fiout, bbox_inches='tight', pad_inches=0)

class JobDraw2D(aares.Job):
    def __set_meta__(self):
        '''
        Sets various package metadata
        '''

        self._program_short_description = prog_short_description

        self._program_name = os.path.basename(sys.argv[0])
        self._program_version = __version__

    def __worker__(self):
        '''
        The actual programme worker
        :return:
        '''
        header = h5z.SaxspointH5(self.params.input)
        frame = numpy.nanmean(header.data[:], axis=0)
        draw(frame, self.params.output,Imax=self.params.max,Imin=self.params.min, cmap=self.params.color_map)

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

    def __process_unhandled__(self):
        '''
        Process unhandled CLI arguments into self.params

        :return:
        '''
        self.params.paths=self.unhandled

def main(argv=None):
    job = JobDraw2D()
    return job.job_exit


def test():
    with h5z.FileH5Z('../data/AgBeh_826mm.h5z') as h5f:
        draw(h5f['entry/data/data'][0],'frame.png','0.5*max')

def main():
    test()

if __name__ == '__main__':
    main()