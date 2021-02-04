"""
Importing data files


@author:     Jan Stransky

@copyright:  2019, 2020 Institute of Biotechnology, Academy of Sciences of the Czech Republic. All rights reserved.

@license:    GNU GPL v3

@contact:    jan.stransky@ibt.cas.cz
@deffield    updated: Updated
"""

import ares
import logging

__all__ = []
__version__ = ares.__version__
prog_short_description = 'Finds and import the data files'

import h5z
import freephil as phil

phil_files = phil.parse('''
group 
.multiple = True
.help = Group of files with the same geometry header
.expert_level = 0
{
    name = None
    .expert_level = 0
    .help = Name of the group
    .type = str
    
    geometry = None
    .type = path
    .help = File with geometry description. Special PHIL file, or H5 with correct header
    .expert_level = 3
    
    mask = None
    .type = path
    .help = File with the mask to be used.
    .expert_level = 0
    
    file 
    .multiple = True
    .help = A file in the group
    {
        path = None
        .type = path
        .help = Location of the file. This is required parameter of the "file"
        
        name = None
        .type = str
        .help = String which is used as a reference for the file elsewhere. It has to be unique string.
    }
    
}
''')

phil_core = phil.parse('''
search_string = None
.help = How should be searched for the file. If it is a folder, it is searched recursively for all data files. Bash-like regex strings are accepted.
.multiple = True
.type = str

output = files.phil
.help = 

name_from = None sample *file_name
.help = How to generate name of the file
.type = choice

shorten = True
.help = Strip the common parts of the file name
.type = bool

prefix = None *time file_name
.help = Prefix the name enumerated by...
.type = choice

''')

class JobImport(ares.Job):
    """
    Run class based on generic Ares run class
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
    job = JobImport()
    return job.job_exit

if __name__ == '__main__':
    main()