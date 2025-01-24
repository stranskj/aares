"""
Background subtraction


@author:     Jan Stransky

@copyright:  2019-2025 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""
import copy

import numpy
import numpy as np


import aares.datafiles
import aares.export
import aares

import concurrent.futures
import os, logging
import freephil as phil

__all__ = []
__version__ = aares.__version__

from aares import my_print

prog_short_description = 'Performs background subtraction.'

phil_export_str = '''
'''

phil_subtract = phil.parse('''
    include scope aares.common.phil_input_files

    input {
    data_file = None
    .type = path
    .help =  File with the data to be processed. It needs to be explicit file name. Use "aares.import" for more complex file search and handling.
    background_file = None
    .type = path
    .help =  File with the background data to be subtracted. It needs to be explicit file name. Use "aares.import" for more complex file search and handling.
    }  

    include scope aares.common.phil_export

    output {
        directory = 'subtracted'
        .type = path
        .help = Output folder for the processed data
        file = 'subtracted.h5s'
        .type = path
        .help = Output file
        out_files = 'subtracted.fls'
        .type = path
        .help = List of the output files in FLS format.
        export = True
        .type = bool
        .help = Should the reduced data be exported as DAT files?
    }

''' + phil_export_str, process_includes=True)

def subtract_reduced(data, background):
    '''
    Subtracts background intensities from data from two compatible files.
    '''

    if data.intensity.shape != background.intensity.shape:
        raise ValueError('Shape of data and background intensities do not match.')

    output = aares.datafiles.Subtract1D(data, exclude=False)

    output.intensity = data.intensity - background.intensity
    output.intensity_sigma = numpy.sqrt(data.intensity_sigma**2 + background.intensity_sigma**2)

    output.add_process(name='Subtraction', description='Background subtraction')
    output.parents = [data.path, background.path]

    output.update_attributes()

    return output

def subtract_file(data, background, output, export=None):
    '''
    Subtracts background intensities from data from two compatible files.
    '''

    if type(data) == str and os.path.isfile(data):
        data = aares.datafiles.read_file(data)

    if type(background) == str and os.path.isfile(background):
        background = aares.datafiles.read_file(background)

    subtracted = subtract_reduced(data, background)
    subtracted.sample_type = 'sample'
    subtracted.write(output)

    if export is not None:
        path_export = os.path.splitext(output)[0]+'.dat'
        subtracted.export(path_export, export)
        # aares.export.write_atsas(subtracted.q_values, subtracted.intensity, subtracted.intensity_sigma,
        #                          file_name=path_export,
        #                          header=['# {}\n'.format(subtracted.path)])
        my_print('Exported file {} to {}'.format(output, path_export))


def subtract_group(group, files_dict, output_dir, export=None):
    '''
    Subtract data defined within a group.
    '''

    assert os.path.isdir(output_dir)
    files_by_name = group.files_by_name
    output_group = copy.deepcopy(group) #aares.datafiles.FileGroup(group.group_phil)
    output_group.file = []
    for name, fi in files_by_name.items():
        if fi.is_background:
            continue

        work_file = fi.path

        try:
            background =files_by_name[fi.background].path
        except KeyError:
            logging.warning('File does not have associated background: {}, {}'.format(name, fi.path))
            continue

        logging.info('Processing: {}\nSubtracting background ({}) from file {}'.format(name,background, work_file))
        output_name = os.path.join(output_dir,fi.name+'.h5s')
        subtract_file(files_dict[work_file], files_dict[background], output_name, export=export)
        my_print('Background subtracted file written: {}'.format(output_name))

        fiout = aares.datafiles.file_object(path=output_name, name= name)
        output_group.file.append(fiout)

    return output_group

class JobSubtract(aares.Job):
    """
    Run class based on generic saxspoint run class
    """

    def __set_system_phil__(self):
        self.system_phil = phil_subtract

    def __argument_processing__(self):
        pass

    def __help_epilog__(self):
        pass

    def __process_unhandled__(self):
        files1d = []
        for param in self.unhandled:
            if aares.datafiles.is_fls(param):
                self.params.input_files = param
            elif os.path.isfile(param) and aares.datafiles.data1D.Reduced1D.is_type(param):
                files1d.append(param)
            #     if h5z.SaxspointH5.is_type(param):
            #         self.params.input.file_name.append(param)
            #     elif aares.q_transformation.ArrayQ.is_type(param):
            #         self.params.input.q_space = param
            #     elif ReductionBins.is_type(param):
            #         self.params.reduction.file_bin_masks = param
            #     else:
            #         raise aares.RuntimeErrorUser('Unknown type of H5 file: {}'.format(param))
            # elif os.path.splitext(param)[1].lower() == '.png':
            #     self.params.input.mask = param
            else:
                raise aares.RuntimeErrorUser('Unknown input: {}'.format(param))

        if len(files1d) == 2:
            self.params.input.data_file = files1d[0]
            self.params.input.background_file = files1d[1]
        elif len(files1d) != 0:
            raise aares.RuntimeErrorUser('Exactly 2 data files are required, or FLS-file. Provided: {}'.format(len(files1d)))

    def __set_meta__(self):
        """
        Metadata for . See saxspoint.Job for predefined defaults
        """
        super().__set_meta__()
        self._program_short_description = prog_short_description

    def __program_arguments__(self):
        """
        Definition of CLI options.
        """
        pass

    def __worker__(self):
        """
        Function, which is actually run on class execution.
        """

        if self.params.output.export:
            export = self.params.export
        else:
            export = None

        if self.params.input.data_file is None and self.params.input.background_file is None and self.params.input_files is  not None:
            files_in = aares.datafiles.DataFilesCarrier(file_phil=self.params.input_files, mainphil=self.system_phil)
            files_out = copy.deepcopy(files_in)
            files_out.file_groups = []
            aares.create_directory(self.params.output.directory)
            for group in files_in.file_groups:
                output_group = subtract_group(group, files_in.files_dict, self.params.output.directory, export=export)
                files_out.file_groups.append(output_group)

            files_out.write_groups(self.params.output.out_files)
            my_print('Output files written: {}'.format(self.params.output.out_files))

        elif self.params.input.data_file is not None and self.params.input.background_file is not None and self.params.input_files is None:
            my_print('Subtracting background ({}) data from file: {}'.format(self.params.input.background_file, self.params.input.data_file))
            subtract_file(self.params.input.data_file, self.params.input.background_file, self.params.output.file, export=export)
            my_print('Subtracted data written to file: {}'.format(self.params.output.file))
        else:
            raise aares.RuntimeErrorUser('Unsupported input.')


        pass

def main():
    # test()
    job = JobSubtract()
    return job.job_exit


if __name__ == '__main__':
    import sys

    sys.exit(main())