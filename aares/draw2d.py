"""
Draw frame


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import numpy
import math
import PIL.Image
import PIL.ImageOps
import matplotlib.cm
import h5z

def draw(frame, fiout, Imax= '2*median', Imin=0, cmap='jet'):
    '''
    Draws frame to a file

    :param frame:
    :param fiout:
    :param Imax:
    :param Imin:
    :return:
    '''

    treshold = [0,0]

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
                raise AttributeError('Unknown parameter: {}'.splited[-1])

            if len(splited) == 2:
                treshold[i] = float(splited[0]) * val
            else:
                treshold[i] = val
        else:
            treshold[i] = float(tres)

    Imin, Imax = sorted(treshold)

    normalized = (frame - Imin) / (Imax - Imin)

    normalized[normalized>1] = 1
    normalized[normalized < 0] = 0

    imshow_kwargs = {
        'vmax': treshold[1],
        'vmin': treshold[0],
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

def test():
    with h5z.FileH5Z('../data/AgBeh_826mm.h5z') as h5f:
        draw(numpy.log10(h5f['entry/data/data'][0]+1),'frame.png',5)

    with h5z.FileH5Z('../data/W_826mm_005Frames.h5z') as h5f:
        draw(numpy.average(h5f['entry/data/data'],axis=0),'beam.png',1)

def main():
    test()

if __name__ == '__main__':
    main()