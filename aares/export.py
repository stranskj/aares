import aares
import sys,os
import freephil as phil
import aares.import_file
import aares.q_transformation as q_trans
import aares.mask
import aares.power
import aares.integrate
import math
import h5z

__all__ = []
__version__ = aares.__version__
prog_short_description = 'Saves the resulting data using various formats.'


def write_atsas(q_val, avr, std, file_name, header=[], footer=[], separator=" "):
    """
    Writes 1D curve in ATSAS format
    :param q_val: q_values
    :param avr: avreages
    :param std: errors
    :param file_name: output file name
    :param header: lines to be written in the beginging
    :param footer: lines to be written after the data
    :param separator: column separator
    :return:
    """

    assert len(q_val) == len(avr) == len(std)

    try:
        with open(file_name,'w') as fiout:
            fiout.writelines(header)

            for q, a, e in zip(q_val, avr, std):
                line = separator.join(['{}']*3)+os.linesep
                fiout.write(line.format(q,a,e))

            fiout.writelines(footer)
    except PermissionError:
        raise aares.RuntimeErrorUser('Cannot write to {}.'.format(file_name))