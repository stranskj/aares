"""
Draw frame


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""
import concurrent.futures
import logging
import multiprocessing

import aares
import sys,os
import freephil as phil

__all__ = []
__version__ = aares.__version__

from aares import my_print

prog_short_description = 'Draws the frame into a PNG file'

import numpy
import math
import PIL.Image
import PIL.ImageOps
import matplotlib.cm
import h5z
import aares.power as pwr
import aares.datafiles
from tqdm import tqdm


phil_core = phil.parse("""
input = None
.type = path
.help = Input file to be drawn

frame = None
.type = str
.help = Frame or range of frames to be drawn. If None, an average of all frames is drawn

by_frame = False
.type = bool
.help = Draws images of the individual frames. Value in "output" is converted to a folder name.

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

log_scale = False
.type = bool
.help = Put the data on logarithmic scale. Values of min and max are on the logscale too.

nproc = 0
.type = int
.help = Number of processes  to be used. When 0, set to number of CPU cores.

""")

def get_treasholds(frame, Imax= '2*median', Imin=0):
    '''
    Returns Imin and Imax to be used as a color range

    :param Imax, Imin: Range limits to be used, if some of them known. If string is provided, it has to be a number followed by '*' and one of keywords: min, max, median, average.
    :type Imax, Imin: float or int or str
    '''

    threshold = [0,0]

    for i,tres in enumerate(list([Imin, Imax])):
        if isinstance(tres, str):
            splited = tres.split('*')
            if splited[-1] == 'median':
                val = numpy.median(frame)
                if val == 0:
                    val = 1
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

    return Imin, Imax


def draw_file(file_in, output=None, params=None):
    if output is None:
        output = params.output

    if not h5z.is_h5_file(file_in):
        raise aares.RuntimeErrorUser('Unsupported file type: {}'.format(file_in))
    logging.info('Reading file...')
    header = h5z.SaxspointH5(file_in)
    if params.log_scale:
        frame = numpy.nanmean(header.data[:], axis=0)
        logging.info('Putting data on a logarithmic scale...')
        print(numpy.min(frame + 1))
        print(numpy.max(frame + 1))
        frame[frame < 0] = 0
        print(numpy.min(frame))
        frame_log = numpy.log(frame + 1)
        print(numpy.min(frame))
        print(numpy.max(frame))
        frame = frame_log.reshape((1, *frame_log.shape))
        params.by_frame = False
    else:
        frame = header.data[:]

    draw(frame,
         output,
         Imax=params.max, Imin=params.min,
         cmap=params.color_map,
         by_frame=params.by_frame)
def draw(frame, fiout, Imax= '2*median', Imin=0, cmap='jet', by_frame=False):
    '''
    Draws frame to a file

    :param frame:
    :param fiout:
    :param Imax:
    :param Imin:
    :return:
    '''

    mean = None
    try:
        Imin = float(Imin)
        Imax = float(Imax)

    except ValueError:
        logging.info('\nDetermining thresholds....')
        mean = numpy.nanmean(frame, axis=0)
        Imin, Imax = get_treasholds(mean, Imax, Imin)

    logging.info('''Used parameters:
    min: {min:.1f}
    max: {max:.1f}'''.format(min=Imin, max=Imax))

    if by_frame:
        aares.my_print('\nFrames will drawn to individual files.')
        folder_name = os.path.splitext(fiout)[0]
        no_frame = numpy.size(frame, axis=0)
        aares.my_print('Number of frames to be drawn: {}'.format(no_frame))
        fi_names = [os.path.join(folder_name,'frame{:04d}.png'.format(i)) for i in range(1, no_frame+1)]

        aares.create_directory(folder_name)

        logging.debug('Current working path: {}'.format(os.getcwd()))
        aares.my_print('Drawing frames to: {}'.format(folder_name))
        logging.debug('Full output path: {}'.format(os.path.realpath(folder_name)))

        with concurrent.futures.ProcessPoolExecutor() as ex:
            pwr.mp_worker(ex.map, frame_to_png,
                          frame,
                          fi_names,
                          [Imin]*no_frame,
                          [Imax]*no_frame,
                          [cmap]*no_frame
                          )
        pass

    else:
        if mean is None:
            mean = numpy.nanmean(frame, axis=0)
        frame_to_png(mean, fiout, Imin, Imax, cmap)

def frame_to_png(frame, fiout, Imin=0, Imax=1, cmap='jet'):
    '''
    Draws 2-dimensional array to a image file.
    '''

    normalized = (frame - Imin) / (Imax - Imin)

    normalized[normalized>1] = 1
    normalized[normalized < 0] = 0

    # imshow_kwargs = {
    #     'vmax': threshold[1],
    #     'vmin': threshold[0],
    #     'cmap': 'RdYlBu',
    #     'extent': (0, frame.shape[0], 0 , frame.shape[1]),
    # }


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
    long_description = ''

    short_description = prog_short_description

    system_phil = phil_core

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

        if h5z.is_h5_file(self.params.input):
            draw_file(self.params.input,self.params.output,self.params)
        elif aares.datafiles.is_fls(self.params.input):
            fls = aares.datafiles.DataFilesCarrier(file_phil=self.params.input,mainphil=self.system_phil)
            out_dir = os.path.splitext(self.params.output)[0]
            aares.create_directory(out_dir)
            if self.params.nproc == 0:
                self.params.nproc = multiprocessing.cpu_count()
            aares.my_print('Using {} processors.'.format(self.params.nproc))
            groups_num = len(fls.file_groups)
            if groups_num > 1:
                my_print('Multiple geometry groups detected, outputting to individual folders.')

            with (concurrent.futures.ProcessPoolExecutor(self.params.nproc) as ex,
                  tqdm(total=len(fls)) as pbar):
                jobs = {}
                try:

                    for group in fls.file_groups:
                        if groups_num > 1:
                            logging.info('Creating folder: {}'.format(group.name))
                            aares.create_directory(os.path.join(out_dir,group.name))
                        for name in group.files_by_name.keys():
                            fi_path = fls.get_file_scope(name).path
                       # aares.my_print('\nDrawing: {}'.format(name))
                            if groups_num > 1:
                                output_name = os.path.join(out_dir,group.name,name+'.png')
                            else:
                                output_name = os.path.join(out_dir,name+'.png')
                            logging.info('File to be drawn: {}\n to: {}'.format(fi_path, output_name))
                            jobs[ex.submit(draw_file,
                                              fi_path,
                                              output=output_name,
                                              params=self.params)] = name
                        # draw_file(fi_path,
                        #           output=os.path.join(out_dir,name+'.png'),
                        #           params=self.params)
                    for job in concurrent.futures.as_completed(jobs):
                        logging.info('File drawn: {}'.format(jobs[job]))
                        pbar.update(1)

                except KeyboardInterrupt:
                    print('Stopping...')
                    for job in jobs:
                        job.cancel()
                    raise KeyboardInterrupt
        else:
            raise aares.RuntimeErrorUser('Unsupported file type: {}'.format(self.params.input))


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
        if len(self.unhandled) > 1:
            raise aares.RuntimeErrorUser('Too much argument to be processed. Only 1 file can be drawn at a time.')

        self.params.input=self.unhandled[0]

def main(argv=None):
    job = JobDraw2D()
    return job.job_exit


def test():
    with h5z.FileH5Z('../data/AgBeh_826mm.h5z') as h5f:
        draw(h5f['entry/data/data'][0],'frame.png','0.5*max')

#def main():
#    test()

if __name__ == '__main__':
    main()