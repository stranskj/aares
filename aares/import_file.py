"""
Importing data files


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import aares
import logging

__all__ = []
__version__ = aares.__version__

import aares.datafiles #import phil_files, is_fls
#from aares.datafiles import DataFilesCarrier
import freephil as phil
import os

prog_short_description = 'Finds and import the data files'

phil_core = phil.parse('''
to_import {
search_string = None
.help = 'How should be searched for the file. If it is a folder, it is searched recursively for all'
        ' data files. Bash-like regex strings are accepted.'
.multiple = True
.type = str

suffix = h5z h5
.help = Suffixes to search for. Only HDF5 files are used by the software.

output = files.fls
.help = PHIL file with description of the imported files and file groups.
.type = path

headers = None
.help = File name, where to store preparsed headers. If None, determined from "output"
.type = path

input_file = None
.multiple = True
.type = path
.help = List of input files in the AAres PHIL format. If the already existing file is given, the headers are read and written again. If multiple files are given, files are merged

name_from = None sample *file_name
.help = How to generate name of the file (e.g. file.name entries)
.type = choice

shorten = True
.help = Strip the common parts of the file name, when generating the name
.type = bool

prefix = None *time file_name
.help = Prefix the file.name enumerated by... It also ensures uniqueness of the naming
.type = choice

force_headers = False
.help = If "input_file" is provided, whether re-reading the headers should be enforced. Otherwise, only the new or modified datafiles are processed.
.type = bool

ignore_merged = True
.help = 'Ignore files with merged frames. Using these diminish some of the AAres features. Moreover,'
        'no special handling fo these is implemented, which can lead to unexpected results.'
.type = bool
.expert_level = 1
}
''')


class JobImport(aares.Job):
    """
    Run class based on generic AAres run class
    """

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

        if not aares.datafiles.is_fls(self.params.to_import.output):
            logging.warning('The file extension should be ".fls". Different extension might cause troubles.')

        if len(self.params.to_import.input_file) > 0:
            aares.my_print('Processing files based on content of file(s):')
            for fi in self.params.to_import.input_file:
                aares.my_print('\t' + fi)
            try:
                file_phils = [phil.parse(file_name=fi) for fi in self.params.to_import.input_file]
                file_scope = aares.datafiles.phil_files.fetch(sources=file_phils)
            except FileNotFoundError as e:
                raise aares.RuntimeErrorUser(e)

            aares.my_print('\nGetting headers extracted earlier...')
            run = aares.datafiles.DataFilesCarrier(file_phil=file_scope, mainphil=self.system_phil)
            if self.params.to_import.force_headers:
                aares.my_print('\nRe-reading the file headers...')
                run.read_headers()
            if self.params.to_import.search_string is not None:
                aares.my_print('Looking for new or modified files...')
                new_files = run.update(self.params.to_import.search_string,
                           suffixes=self.params.to_import.suffix,
                           ignore_merged=self.params.to_import.ignore_merged)
                aares.my_print('Identified {} new or modified files.'.format(len(new_files)))

            if len(run.files_dict) == 0:
                run.read_headers() #Might be done for second time, is HDR-file originally exist...
            if self.params.to_import.headers is None:
                 self.params.to_import.headers = os.path.splitext(self.params.to_import.output)[0]+'.hdr'
            run.header_file = self.params.to_import.headers
        else:
            run = aares.datafiles.DataFilesCarrier(run_phil=self.params.to_import, mainphil=self.system_phil)

        files = aares.datafiles.phil_files.format(run.file_groups)
        #       print(files.as_str(expert_level=0))
        if self.params.to_import.output is not None:
            run.write_groups(self.params.to_import.output)

            aares.my_print('List of imported files was written to: {}'.format(self.params.to_import.output))

        if run.header_file is not None:
            run.write_headers_to_file()
            aares.my_print(
                'Parsed headers of data files are saved to: {}'.format(run.header_file))

        pass

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

    def __process_unhandled__(self):
        '''
        Process unhandled CLI arguments into self.params

        :return:
        '''

        aares_files = [fi for fi in self.unhandled if aares.datafiles.is_fls(fi)]

        if len(aares_files) > 0:
            if self.params.to_import.input_file is None:
                self.params.to_import.input_file = []
            self.params.to_import.input_file.extend(aares_files)

            for fi in self.params.to_import.input_file:
                self.unhandled.remove(fi)
        self.params.to_import.search_string.extend(self.unhandled)

    def __help_epilog__(self):
        '''
        Epilog for the help

        '''
        pass


def main():
    job = JobImport()
    return job.job_exit


if __name__ == '__main__':
    main()
