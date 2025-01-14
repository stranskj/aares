"""
Background subtraction


@author:     Jan Stransky

@copyright:  2019-2025 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""
import numpy

import aares
import aares.datafiles
import aares

import concurrent.futures
import os, logging
import freephil as phil

__all__ = []
__version__ = aares.__version__
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
        directory = None
        .type = path
        .help = Output folder for the processed data
    }

''' + phil_export_str, process_includes=True)

def subtract_reduced(data, background):
    '''
    Subtracts background intensities from data from two compatible files.
    '''

    if data.intensity.shape != background.intensity.shape:
        raise ValueError('Shape of data and background intensities do not match.')

    output = aares.datafiles.Subtract1D(data)

    output.intensity = data.intensity - background.intensity
    output.intensity_sigma = numpy.sqrt(data.intensity_sigma**2 + background.intensity_sigma**2)

    output.add_process(name='Subtraction', description='Background subtraction')
    output.parents = [data.path, background.path]

    output.update_attributes()

    return output

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
            elif aares.datafiles.data1D.Reduced1D.is_type(param):
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

        if self.params.input.file_name is not None and len(self.params.input.file_name) > 0:
            files_in = aares.datafiles.DataFilesCarrier(run_phil=self.params, mainphil=self.system_phil)
        elif self.params.input_files is not None:
            files_in = aares.datafiles.DataFilesCarrier(file_phil=self.params.input_files, mainphil=self.system_phil)
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