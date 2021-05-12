#!/usr/bin/env python3

import aares
# import aares.uv
import argparse
import os
import logging
from aares import my_print
import sys
import pkg_resources
import importlib
import logging

import freephil as phil

# import psutil

__version__ = aares.__version__

prog_short_description = 'Another Angular REduction for SAXS'


phil_core = phil.parse('''
input {
    file_name = None
    .type = path
    .multiple = True
    }


    include scope aares.integrate.phil_core

    include scope aares.q_transformation.phil_core
    
    include scope aares.mask.phil_core

    
''', process_includes=True)

def epilog():
    return_str = 'List of available commands in this package:\n'
    try:

        if 'win' in sys.platform:
            suffix = '.exe'
        else:
            suffix = ''

        for ep in (ept for ept in pkg_resources.iter_entry_points('console_scripts')  # ):
                   if 'aares' in ept.module_name):
            try:
                module = importlib.import_module(ep.module_name)
                return_str += ep.name + suffix + '\n\t' + module.prog_short_description + '\n\n'
            except:
                return_str += 'Cannot find {}\nTry reinstall the package.'.format(ep.module_name)
    except Exception as e:
        logging.exception(e)
        return_str += 'Here is typically list of available commands. Something is wrong.'

    return return_str


# def system_info():
#   memory = dict(psutil.virtual_memory()._asdict())
#  return '''
# Number of CPU cores: {cpu: 3d}
# Total RAM:           {ram_tot: 3.1f} GB
# Available RAM:       {ram_avail: 3.1f} GB
# '''.format(cpu=atsas.get_nproc(),
#          ram_tot=memory['total']/(1024*1024*1024),
#         ram_avail=memory['available']/(1024*1024*1024))

class MainJob(aares.Job):
    """
    Run class based on generic AAres run class
    """

    def __argument_processing__(self):
        pass

    def __process_unhandled__(self):
        pass

    def __set_meta__(self):
        """
        Metadata for . See aares.Job for predefined defaults
        """
        self._program_short_description = prog_short_description

    def __program_arguments__(self):
        """
        Definition of CLI options.
        """
        pass

    def __worker__(self):
        """
        Function, which is actually runned on class execution.
        """
        self._parser.print_help()

    def __set_system_phil__(self):
        '''
        Settings of CLI arguments. self._parser to be used as argparse.ArgumentParser()
        '''
        self._system_phil = phil_core

    def __help_epilog__(self):
        return epilog()

    def __set_argument_parser__(self, parser=None):
        super().__set_argument_parser__(formatter_class=argparse.RawDescriptionHelpFormatter)


def main(argv=None):
    job = MainJob()
    return job.job_exit


if __name__ == "__main__":
    sys.exit(main())
