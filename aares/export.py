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

phil_export_str = '''
'''

phil_export = phil.parse('''
    include scope aares.common.phil_input_files
    
    input {
    file_name = None
    .multiple = True
    .type = path
    .help =  Files with the data to be processed. It needs to be explicit file name. Use "aares.import" for more complex file search and handling.
    }  
    
    include scope aares.common.phil_export
    
    output {
        directory = None
        .type = path
        .help = Output folder for the processed data
    }
    
'''+phil_export_str, process_includes=True)

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


class JobExport(aares.Job):
    """
    Run class based on generic saxpoint run class
    """

    def __set_system_phil__(self):
        self.system_phil = phil_export

    def __argument_processing__(self):
        pass

    def __help_epilog__(self):
        pass

    def __process_unhandled__(self):
        for param in self.unhandled:
            if aares.datafiles.is_fls(param):
                self.params.input_files = param
            elif aares.datafiles.data1D.Reduced1D.is_type(param):
                self.params.input.file_name.append(param)
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
    job = JobExport()
    return job.job_exit


if __name__ == '__main__':
    import sys

    sys.exit(main())