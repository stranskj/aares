"""
Merging multiple files


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""
import copy

import h5py
import numpy

import aares
import logging

__all__ = []
__version__ = aares.__version__
prog_short_description = 'Merges multiple files to one'

import numpy as np
import math
import PIL.Image
import PIL.ImageOps
import freephil as phil
import h5z
import os

import aares.datafiles

phil_core = phil.parse('''
    include scope aares.import_file.phil_core
    
    check_geometry = True
    .help = Check, that experimental geometry for the files is consistent
    .type = bool
    
    output = merged.h5
    .help = Merged file output
    .type = path 
''', process_includes=True)

def merge_files(in_files, out_file = 'merge.h5'):
    '''
    in_files: Headers of files whose data are being merged
    out_file: output file

    '''

    output_h5 = copy.deepcopy(in_files[0])
    first = in_files[0]
    output_h5.path = out_file
#    data = numpy.concatenate([fi.data for fi in in_files])
    time_offset = numpy.concatenate([fi.time_offset for fi in in_files])
 #   output_h5.data = data
    output_h5.time_offset = time_offset
    output_h5.write(out_file)

    with h5py.File(out_file, 'a') as fout:
        data = fout.create_dataset('/entry/data/data',shape=(len(time_offset),*first.frame_size),dtype=first.data.dtype,
                                   compression='gzip'
                                   )
        i = 0
        for fi in in_files:
            logging.info('Writing {}'.format(fi.path))
            no_frames = len(fi.time_offset)
            data[i:i+no_frames] = fi.data
            i += no_frames
    #return output_h5

class JobMerge(aares.Job):
    """
    Run class based on generic AAres run class
    """

    def __process_unhandled__(self):
        aares_files = [fi for fi in self.unhandled if aares.datafiles.is_fls(fi)]

        if len(aares_files) > 0:
            if self.params.to_import.input_file is None:
                self.params.to_import.input_file = []
            self.params.to_import.input_file.extend(aares_files)

            for fi in self.params.to_import.input_file:
                self.unhandled.remove(fi)
        self.params.to_import.search_string.extend(self.unhandled)
        # if len(self.unhandled) > 0:
        #
        #     files = aares.datafiles.get_files(self.unhandled, ['.h5z', '.h5'])
        #     self.params.to_importfiles.extend(files)

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

        data_files = aares.datafiles.DataFilesCarrier(run_phil=self.params.to_import, mainphil=phil_core)
        aares.my_print('Files to be merged:')
        for fi in data_files.files_dict.keys():
            aares.my_print(fi)
        aares.my_print('\n')
        if self.params.check_geometry:
            if len(data_files.file_groups) != 1:
                raise aares.RuntimeErrorUser('Geometries in the files are inconsistent.')

        aares.my_print('Merging files...')
        merge_files(list(data_files.files_dict.values()), self.params.output)
        aares.my_print('Saved the megred file: {}'.format(self.params.output))
#        merged_file.write(self.params.output)


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


def main():
    job = JobMerge()
    return job.job_exit


if __name__ == '__main__':
    main()
