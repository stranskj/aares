"""
Importing data files


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""
import re

import aares
import logging
import aares.report

__all__ = []
__version__ = aares.__version__

import aares.datafiles #import phil_files, is_fls
#from aares.datafiles import DataFilesCarrier
import freephil as phil
import os
import aares.power

from aares import my_print

prog_short_description = 'Finds and import the data files'

prog_long_description = '''
This job imports the data files in the  AAres environment. Specified path(s) is searched for compatible data files (h5 or h5z). During the import, list of files is summarized in FLS file, which allows to modify some metadata and control some file-specific settings. Data file headers are also analysed and extracted to HDR file. This is done because of the internal layout of H5Z files and it should speed up reprocessing.

Typical usage
-------------

The expected parameters are path(s) to a file tree with `h5` or `h5z` files, or individual files. The path is recursively searched for the files. For example:

```
aares.import data
```

The command prodces two files: 
 * `files.fls`: Text file with list of imported files  
 * `hdr`: extracted file headers
 
The files are also assigned to groups based on the geometry of the experiment. This might play a role for example, if collect the data at different sample to detector distances. In typical cases, there should be only one group.  

Dataset names
-------------

On import, each file is assigned with a unique name. The uniqueness is achieved by prepending the name with a number; typically the files will be ordered by the time of collection. The rest of the name is generated either from the file name or from sample name stored in the h5z file header. This behaviour is controlled using `name=`.

File list update
-----------

If the FLS file already exist, it can be updated with new files. In this mode, files which are new or changed since last imported will be analyzed. The file change is determined using MD5 checksum. This feature is handy, when you want to process the files during the data collection:

```
aares.import data files.fls
```

Full specification of FLS file is available [here](../FLS_file.md)

'''

phil_core = phil.parse('''
to_import {
search_string = None
.help = 'How should be searched for the file. If it is a folder, it is searched recursively for all'
        ' data files. Bash-like regex strings are accepted.'
.multiple = True
.type = str

suffix = 'h5z' 'h5'
.type = strings
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

detect_background = True
.type = bool
.help = 'Enable automatic detection of which samples should be used as buffer/matrix.'

background_detection {
    pattern = "buffer pufr matrix"
    .type = str
    .help = 'A string to search for to flag the sample as background. Multiple space separated can be provided. The search accepts Python Regex syntax (see https://docs.python.org/3/library/re.html#regular-expression-syntax)'
    #.multiple = True
    
    search_in = *name path
    .type = choice
    .help = 'Where should be the string searched for: `path` in the `file.path`; `name` in the `file.name`'
        
}

assign_background = True
.type = bool
.help = 'Enable automatic assignment of which background belongs to which sample.'

background_assignment {
    method = *time
    .type = choice
    .help = 'Method used to assign background. `time` - use the previous file (by  time), which was flagged as buffer.'
    

}

}

include scope aares.power.phil_job_control
''', process_includes=True)



class JobImport(aares.Job):
    """
    Run class based on generic AAres run class
    """

    long_description = prog_long_description

    short_description = prog_short_description

    system_phil = phil_core

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

        nproc = self.params.job_control.nproc

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
            run = aares.datafiles.DataFilesCarrier(file_phil=file_scope, mainphil=self.system_phil, nproc=nproc)
            if self.params.to_import.force_headers:
                aares.my_print('\nRe-reading the file headers...')
                run.read_headers()
            if self.params.to_import.search_string is not None:
                aares.my_print('Looking for new or modified files...')
                new_files = run.update(self.params.to_import.search_string,
                                       run_phil = self.params.to_import,
                           suffixes=self.params.to_import.suffix,
                           ignore_merged=self.params.to_import.ignore_merged)
                aares.report.log_number_of_frames(new_files)
                aares.my_print('Identified {} new or modified files.'.format(len(new_files)))

            if len(run.files_dict) == 0:
                run.read_headers() #Might be done for second time, is HDR-file originally exist...
            if self.params.to_import.headers is None:
                 self.params.to_import.headers = os.path.splitext(self.params.to_import.output)[0]+'.hdr'
            run.header_file = self.params.to_import.headers
        else:
            run = aares.datafiles.DataFilesCarrier(run_phil=self.params.to_import, mainphil=self.system_phil, nproc=nproc)
            aares.report.log_number_of_frames(run.files_dict)

        files = aares.datafiles.phil_files.format(run.file_groups)
        #       print(files.as_str(expert_level=0))

        if self.params.to_import.detect_background:
            if ' ' in self.params.to_import.background_detection.pattern:
                self.params.to_import.background_detection.pattern = self.params.to_import.background_detection.pattern.split(' ')
            my_print('Detecting background files...')
            run.detect_background(self.params.to_import.background_detection.pattern, self.params.to_import.background_detection.search_in)

        if self.params.to_import.assign_background:
            my_print('Assigning background files...')
            run.assign_background(self.params.to_import.background_assignment.method)

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
