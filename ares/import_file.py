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
import glob
import os
import itertools
import ares.power as pwr

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

suffix = h5z h5
.help = Suffixes to search for. HDF5 files are used by the software.


output = files.phil
.help = PHIL file with description of the imported files and file groups.
.type = path

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

def group_object(files = []):
    '''
    Returns empty Group scope_extract
    :return:
    '''

    pe = phil_files.extract()
    gr = pe.group.pop()
    gr.file.pop()

    gr.file.extend(files)

    return gr

def file_object(path = None, name = None):
    '''
    Returns empty File scope_extract
    :return:
    '''

    pe = phil_files.extract()
    fo = pe.group[0].file.pop()
    fo.path = path
    fo.name = name
    return fo

def search_files(indir, suffix):
    fiout = []
    for root, dirs, files in os.walk(indir):
        for file in files:
            if file.endswith(suffix):
                fiout.append(os.path.join(root, file))
    return fiout

def get_files(inpaths, suffixes):
    if isinstance(suffixes, str):
        suffixes = [suffixes]

    for i,suf in enumerate(suffixes):
        if not(suf[0] == '.'):
            suffixes[i] = '.'+suf

    files = [[file for file in glob.glob(fi)] for fi in inpaths]
    inpaths = [fi for fi in itertools.chain.from_iterable(files)]
    file_list = []
    for fi in inpaths:
        if os.path.isdir(fi):
            for suf in suffixes:
                file_list.extend(search_files(fi, suf))
        else:
            file_list.append(fi)
    return file_list

def files_to_groups(files, headers_to_match = 'entry/instrument/detector'):
    '''
    Split the files in the groups based on common headers.
    :param files: list of files to be distributed
    :return:
    '''
    groups = []
    if isinstance(headers_to_match,str):
        headers_to_match = [headers_to_match]

    if not isinstance(files,dict):
        files = pwr.get_headers_dict(files)


    for fi, hd in files.items():
        file_scope = file_object(path=fi)

        for group in groups:
            if all(hd[match] == files[group.file[0].path][match] for match in headers_to_match):
                group.file.append(file_scope)
                break
        else:
            groups.append(group_object([file_scope]))


    return groups

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

        pass
        files = get_files(self.params.search_string, self.params.suffix)

        saxspoint_geometry_fields =[
            'entry/instrument/detector/depends_on',
            'entry/instrument/detector/description',
            'entry/instrument/detector/detector_number',
            'entry/instrument/detector/distance',
            'entry/instrument/detector/height',
            'entry/instrument/detector/meridional_angle',
            'entry/instrument/detector/sensor_material',
            'entry/instrument/detector/sensor_thickness',
            'entry/instrument/detector/x_pixel_offset',
            'entry/instrument/detector/x_pixel_size',
            'entry/instrument/detector/y_pixel_offset',
            'entry/instrument/detector/y_pixel_size',
            'entry/instrument/detector/x_translation',
            'entry/instrument/monochromator',
        ]

        groups = files_to_groups(files, saxspoint_geometry_fields)
        files_extract = phil_files.extract()
        files_extract.group = groups
        print(phil_files.format(files_extract).as_str(expert_level=0))

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
        self.params.search_string.extend(self.unhandled)

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