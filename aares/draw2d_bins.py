import os, logging, sys
import freephil as phil

import aares
import aares.integrate
import aares.datafiles
from aares.integrate import ReductionBins

prog_short_description = 'Draw bin masks to PNG.'

phil_draw_bins = phil.parse("""
include scope aares.common.phil_input_files

    input {
    file_name = None
    #.multiple = True
    .type = path
    .help =  File with the data to be processed. It needs to be explicit file name. Use "aares.import" for more complex file search and handling.
    }

    output {
        directory = 'q_images'
        .type = path
        .help = Output folder for the processed data
        prefix = ""
        .type = str
        .help = Prefix to the file names
        clear = True
        .type = bool
        .help = Clear output folder, if exists
    }
""", process_includes=True)

class DrawBinsJob(aares.Job):
    """
    Run class based on generic saxspoint run class
    """

    long_description = 'This tool visualises bin masks used for the data reduction. For each bin in Q-values, PNG file is created. This can be useful for checking which pixels are used around the beamstop.'

    short_description = prog_short_description

    system_phil = phil_draw_bins

    def __set_meta__(self):
        super().__set_meta__()
        self._program_short_description = self.short_description

    def __set_system_phil__(self):
        self.system_phil = phil_draw_bins

    def __help_epilog__(self):
        pass

    def __argument_processing__(self):
        pass

    def __process_unhandled__(self):
        for param in self.unhandled:
            if aares.datafiles.is_fls(param):
                self.params.input_files = param
            elif ReductionBins.is_type(param):
                self.params.input.file_name = param
            else:
                raise aares.RuntimeErrorUser('Unknown file type: {}'.format(param))

    def __worker__(self):

        jobs = []

        if self.params.input.file_name is not None:
            jobs.append((self.params.input.file_name, self.params.output.directory))
        elif self.params.input_files is not None:
            files = aares.datafiles.DataFilesCarrier(file_phil=self.params.input_files, mainphil=aares.integrate.phil_core)

            for group in files.file_groups:
                if ReductionBins.is_type(group.group_phil.reduction.file_bin_masks):
                    jobs.append((group.group_phil.reduction.file_bin_masks, self.params.output.directory +'_' +group.name))
                else:
                    logging.warning('Group {} has no binning file assigned.'.format(group.name))
        else:
            aares.my_print('Nothing to process.')
            sys.exit(0)

        for fi, outdir in jobs:
            bins = ReductionBins(fi)
            bins.draw2d(output_dir=outdir, clear=self.params.output.clear)
            aares.my_print('Images of individual bins are draw in folder: {}'.format(outdir))

def main():
    import sys
    job = DrawBinsJob()
    sys.exit(job.job_exit)

def draw_bins_old():
    # TODO: re do this properly, ideally after lib-bin refactorisation

    import sys
    aares.my_print("AAres draw bins")

    fin = sys.argv[1]

    if not ReductionBins.is_type(fin):
        logging.error('This is not file with Q-bins. Search for suffix ".bins.h5a"')
        sys.exit(1)

    bins = ReductionBins(fin)
    bins.draw2d()

    aares.my_print('Images of individual bins are draw in folder "q_images".')
    aares.my_print('\nFinished.')

    sys.exit(0)

if __name__ == '__main__':
    import sys

    sys.exit(main())