'''
Common "power" features for Ares
'''

import multiprocessing
import concurrent.futures
import sys

import numpy as np


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(all_args):
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def mp_worker(func, *args, **kwargs):
    '''
    Worker for multiprocessing (concurrent.futures etc.) with KeyboardInterrupt handling

    :param func:
    :param args:
    :param kwargs:
    :return:
    '''
    try:
        result = func(*args, **kwargs)
    except KeyboardInterrupt:
        print('Keyboard interupt recieved, exiting...')
        sys.exit('Keyboard interupt, exited.')
    except ChildProcessError as e:
        print('Child processes shut down.')
        # print(e)
        raise ChildProcessError(e)
    finally:
        pass
    return result

def get_headers(file_list, nproc=None, sort=None):
    """
    Returns list of headers in SaxspointH5 format, sorted by time
    """
    import h5z
    with concurrent.futures.ProcessPoolExecutor(nproc) as ex:
        headers = list(ex.map(mp_worker, [h5z.SaxspointH5]*len(file_list), file_list))

    if sort is not None:
        headers.sort(key=lambda x: x.attrs[sort])
    return headers

def get_headers_dict(file_list, nproc=None, sort=None):
    '''
    Returns dict of file headers, named by the file path

    :param file_list:
    :param nproc:
    :param sort:
    :return:
    '''

    return {fi : hd for fi, hd in zip(file_list, get_headers(file_list, nproc=nproc, sort=None))}
