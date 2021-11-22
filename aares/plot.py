#!/usr/bin/env python3
# encoding: utf-8
"""
    aares.plot -- Prepare Gnuplot script file for plotting of the SAXS data or fits.

The programme generates a file, which instruct GNUplot, how to plot given data. Many options are available :-)


@author:     Jan Stransky

@copyright:  2020 Institute of Biotechnology. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import sys
import os, glob, shutil
import itertools

import argparse
import time
import math

import aares

#from atsas_mp.atsas import CLIError, AtsasIOError
#import atsas_mp.dammin_mp
#import atsas_mp.dammif_mp
#import atsas_mp.damaver_mp
#import concurrent.futures
#import contextlib

__version__ = aares.__version__
prog_short_description = 'GNUplotting of the SAXS data'
prog_long_description = '''The programme generates a file, which instruct GNUplot, how to plot given data. Many options are available :-)
'''

DEBUG = 1
TESTRUN = 0
PROFILE = 0


class FileType(object):
    """
    Generic file type
    """
    pass


class FileTypeDAT(FileType):
    """
    ATSAS datafile

    self.separator: columns separator in the file
    """

    def __init__(self, separator=None, units=None):
        self.separator = separator
        self.units = units

    @property
    def separator(self):
        return self._separator

    @separator.setter
    def separator(self, val):
        if val is None:
            self._separator = 'whitespace'
        else:
            self._separator = val


class FileTypeFIT(FileType):
    """"
    File with data on fitting a model to data
    """

    def __init__(self, units='1/A'):
        self.units = units
        self.separator = 'whitespace'


def get_file_type(in_file):
    """
    Returns FileType class describing given file type
    """
    try:
        with open(in_file, 'r') as fin:
            lines = fin.readlines()
            first_line = lines[0]
            pref, ext = os.path.splitext(in_file)
            if ('Chi^2' in first_line):
                return FileTypeFIT()
            elif (ext == '.dat'):
                for line in reversed(lines):
                    for sep in [None, ',']:
                        line_items = line.strip().split(sep=sep)
                        try:
                            if (len(line_items) == 3) and (float('NaN') not in [float(x) for x in line_items]):

                                if float(line_items[0]) <= 1:
                                    units = '1/A'
                                else:
                                    units = '1/nm'
                                float(line_items[1])
                                float(line_items[2])

                                return FileTypeDAT(separator=sep, units=units)
                        except ValueError:
                            continue
            else:
                pass
            raise TypeError('Unsupported file format.')
    except Exception as e:
        raise Exception('Bad or unsupported file. ' + str(e))


def gnuplot_scattering_common(file_types, out_unit):
    """
    Common parts of scattering files
    """
    lines = []

    lines.append('set logscale y')
    if file_types[0].separator != 'whitespace':
        lines.append('set datafile separator "{}"'.format(file_types[0].separator))
    lines.append('set xlabel "{}"'.format(out_unit))
    lines.append('set ylabel "I"')
    return lines


def gnuplot_scattering(infiles, file_types, out_unit='1/nm', **kwargs):
    """
    Return lines for standard Intensity vs. q
    """

    if out_unit == '1/nm':
        out_unit_multiplier = 1.0
    elif out_unit == '1/A':
        out_unit_multiplier = 10.0
    else:
        raise TypeError('Unknown output units.')

    lines = []

    lines.append('plot \\')
    for fi, tp in zip(infiles, file_types):
        if tp.units == '1/A':
            unit_multiplier = 10 / out_unit_multiplier
        elif tp.units == '1/nm':
            unit_multiplier = 1 / out_unit_multiplier
        lines.append(
            "'{file}' using ($1*{unit}):2  title '{title}', \\".format(file=fi, title=fi, unit=unit_multiplier))
    lines.append('')
    return lines


def gnuplot_fit(infiles, file_types, out_unit='1/nm', difference=False, **kwargs):
    """
    Gnuplotting of model fits (e.g. DAMMI[N,F], CRYSOL, SASREF, ...)
    """
    if out_unit == '1/nm':
        out_unit_multiplier = 1.0
    elif out_unit == '1/A':
        out_unit_multiplier = 10.0
    else:
        raise TypeError('Unknown output units.')

    lines = []

    if difference:
        lines.extend('''
set multiplot layout 2,1
set lmargin at screen 0.12
set tmargin at screen 0.9
set bmargin at screen 0.3
set ylabel "I" offset 1,0
unset xtics
unset xlabel
'''.split('\n')
                     )

    data_lines = []
    fit_lines = []

    lines.append('plot \\')
    for fi, tp in zip(infiles, file_types):
        if tp.units == '1/A':
            unit_multiplier = 10 / out_unit_multiplier
        elif tp.units == '1/nm':
            unit_multiplier = 1 / out_unit_multiplier
        data_lines.append(
            "'{file}' using ($1*{unit}):2  title '{title} data', \\".format(file=fi, title=fi, unit=unit_multiplier))
        fit_lines.append(
            "'{file}' using ($1*{unit}):4  title '{title} model' with lines, \\".format(file=fi, title=fi,
                                                                                        unit=unit_multiplier))
    lines.extend(data_lines)
    lines.extend(fit_lines)
    lines.append('')

    if difference:
        settings = '''
unset logscale
set tmargin at screen 0.3
set bmargin at screen 0.15
set xtics
set xlabel "q \\[{unit}\\]"
set ylabel {ylabel}
set yrange [:] # You might want adjust here
# colors used are the same as with "fit". You might want to adjust values of "lc", if you do changes.
plot \\'''.format(unit=out_unit, ylabel='" \{\/Symbol D\}I\/\{\/Symbol s\} \\[-\\]" offset -1,0')

        lines.extend(settings.split('\n'))

        fit_lines = []
        for fi, tp, i in zip(infiles, file_types, range(len(infiles))):
            if tp.units == '1/A':
                unit_multiplier = 10 / out_unit_multiplier
            elif tp.units == '1/nm':
                unit_multiplier = 1 / out_unit_multiplier
            fit_lines.append(
                "'{file}' using ($1*{unit}):(($2-$4)/$3)  notitle lc {color}, \\".format(file=fi, title=fi,
                                                                                unit=unit_multiplier, color=len(infiles)+i+1))
        lines.extend(fit_lines)
        lines.append('')
        lines.append('unset multiplot')
        lines.append('')

    return lines


def process(infiles, output, mode='w', units='1/nm', difference=False, **kwargs):
    """
    Processing worker
    """
    file_types = [get_file_type(fi) for fi in infiles]

    output_lines = []
    for tp in file_types:
        if not isinstance(tp, type(file_types[0])):
            raise TypeError('The input files are not of the same type.')

    output_lines.extend(gnuplot_scattering_common(file_types, units))

    if isinstance(file_types[0], FileTypeDAT):
        output_lines.extend(gnuplot_scattering(infiles, file_types, units))
    elif isinstance(file_types[0], FileTypeFIT):
        output_lines.extend(gnuplot_fit(infiles, file_types, units, difference))

    with open(output, mode) as fiout:
        #fiout.writelines(output_lines)
        for line in output_lines:
            fiout.write(line + '\n') #os.linesep)

    print('''
    Output file {} written. To see the result, run:
    
    gnuplot -p {}'''.format(output, output))


def main(argv=None):  # IGNORE:C0111
    """Command line options."""

    if argv is None:
        argv = sys.argv[1:]

    program_name = os.path.basename(sys.argv[0])
    program_version = "%s" % __version__
    #   program_build_date = str(__date__)
    program_version_message = '%%(prog)s %s \nPython %s' % (program_version, sys.version)
    program_shortdesc = '\t' + program_name + ' -- ' + prog_short_description + '\n'
    program_license = '''%s
USAGE
''' % (prog_long_description + '\n')

    try:
        # Setup argument parser
        parser = argparse.ArgumentParser(description=program_license,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        parser.add_argument('-o', '--output',
                            dest='output',
                            default='gnuplot.plt',
                            help='Name of the output file (default: %(default)s).')

        parser.add_argument('-m', '--mode',
                            metavar='MODE',
                            choices=['w', 'a'],
                            default='w',
                            help='Handling mode for the output file, when file exists. "w" will recreate the file, "a" will append new lines to the existing file (default: %(default)s).')

        parser.add_argument('-u', '--units',
                            dest='units',
                            choices=['A', 'nm'],
                            default='nm',
                            help='Units in the output file. The original file is not modified. (default: %(default)s)')
        parser.add_argument('-d', '--difference',
                            dest='difference',
                            action='store_true',
                            default=False,
                            help='For fit files, create also graph of error-normalized differences between data and fit.'
                            )

        parser.add_argument('-V', '--version',
                            action='version',
                            version=program_version_message)

        parser.add_argument(dest="infiles",
                            help="Files to be plotted.",
                            metavar="FILE",
                            nargs='+')

        # Process arguments
        args = parser.parse_args(argv)

        files = [[file for file in glob.glob(fi)] for fi in args.infiles]
        args.infiles = [fi for fi in itertools.chain.from_iterable(files)]

        args.units = '1/' + args.units

        print(args)

        print('    ======================')
        print('\t' + program_shortdesc.strip())
        print('    ======================')
       # print(atsas_mp.atsas.atsas_mp_license)

        start_t = time.time()
        process(**vars(args))
        el_time = time.time() - start_t
        #    print('Results written to: {}'.format(args.output))
        print('Finished. Time elapsed: {:.1f} s'.format(el_time))
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG or TESTRUN:
            raise (e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2


if __name__ == "__main__":
    sys.exit(main())
