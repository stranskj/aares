'''
Common "power" features for AAres
'''

import multiprocessing
import concurrent.futures
import sys
import freephil as phil
import logging

import numpy as np

import aares.datafiles
import pickle

phil_job_control = phil.parse('''
job_control {
    nproc = None
    .help = Maximum number of CPUs to be used in total. If None, all available CPUs are used.
    .type = int
    
    jobs = None
    .help = 'Number of jobs to be processed in parallel. One job processes one range of frames, which'
            ' is usually one file. Therefore this setting is useful optimizing hard drive load.'
            ' If None, determined automatically.'
    .type = int
    
    threads = None
    .help = 'Number of CPUs used per one job. If None, determined automatically.'
    .type = int
}
''')


def get_cpu_distribution(job_control):
    """
    Determines splitting to jobs and threads
    :param job_control: job_control keyword from Phil files
    :type job_control: freephil.scope_extract
    :returns: Tuple, max number of threads and max number of jobs
    """

    if job_control.nproc is None:
        nproc = multiprocessing.cpu_count()
    else:
        nproc = job_control.nproc

    if (job_control.jobs is not None) and (job_control.threads is not None):
        jobs = job_control.jobs
        threads = job_control.threads
    elif (job_control.jobs is None) and (job_control.threads is not None):
        threads = job_control.threads
        jobs = max(1,int(nproc / threads))
    elif (job_control.jobs is not None) and (job_control.threads is None):
        jobs = job_control.jobs
        threads = max(1,int(nproc / jobs))
    else:
        jobs = int(nproc / 8)
        if nproc % 8 > 0:
            jobs += 1
        threads = int(nproc / jobs)

    if jobs * threads > nproc:
        logging.warning(
            'Number of simultaneous threads ({}*{}) is bigger than number of available CPUs {}'.format(
                jobs, threads, nproc))
    logging.info('Using {} jobs with {} CPUs per job.'.format(jobs, threads))

    return threads, jobs


def parallel_apply_along_axis(func1d, axis, arr, nproc=None, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, nproc)]

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


def map_mp(func1d, *args, nchunks=None, **kwargs):
    """
    Divide list into chunks, and process them in parallel. Eqivavlent to ''map'' function.
    :param func1d: function to be used
    :param lst: list to be processed
    :param nchunks: number of chunks; all processed in parallel. If None, equals to nmber of CPUs
    :param args, kwargs: arguments for the function, as  with map
    :return: list of results
    """

    try:
        length_iterable = len(args[0])
    except IndexError:
        raise AttributeError('Nothing to iterate over.')

    if nchunks is None:
        nchunks = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=nchunks) as ex:
        res_chunks = ex.map(mp_worker,
                            [func1d] * length_iterable, *args, **kwargs,
                            chunksize=max(1,int(length_iterable / nchunks)))

    return list(res_chunks)

def map_th(func1d, *args, nchunks=None, **kwargs):
    """
    Divide list into chunks, and process them in parallel. Eqivavlent to ''map'' function.
    :param func1d: function to be used
    :param lst: list to be processed
    :param nchunks: number of chunks; all processed in parallel. If None, equals to nmber of CPUs
    :param args, kwargs: arguments for the function, as  with map
    :return: list of results
    """

    try:
        length_iterable = len(args[0])
    except IndexError:
        raise AttributeError('Nothing to iterate over.')

    if nchunks is None:
        nchunks = multiprocessing.cpu_count()

    with concurrent.futures.ThreadPoolExecutor(max_workers=nchunks) as ex:
        res_chunks = ex.map(mp_worker,
                            [func1d] * length_iterable, *args, **kwargs,
                            chunksize=max(1,int(length_iterable / nchunks)))

    return list(res_chunks)


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
    #import h5z
    #with concurrent.futures.ProcessPoolExecutor(nproc) as ex:
    #    headers = list(ex.map(mp_worker, [h5z.SaxspointH5] * len(file_list), file_list))

    headers_dict = get_headers_dict(file_list, nproc=nproc)
    headers = [headers_dict[fi] for fi in file_list ]

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

    import h5z
    from tqdm import tqdm

    with (concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as ex,
          tqdm(total=len(file_list)) as pbar):
        pbar.update(0)
        jobs = {ex.submit(mp_worker, aares.datafiles.read_file, fi) : fi for fi in file_list }
        #headers = list(ex.map(mp_worker, [h5z.SaxspointH5] * len(file_list), file_list))
        headers_out = {}
        for job in concurrent.futures.as_completed(jobs):
            fi = jobs[job]
            try:
                if job.result() is not None:
                    headers_out[fi] = job.result()
            except pickle.PickleError:   # A bit dirty hack for files loaded using class factory
                logging.debug('Processing unpicklable file...')
                headers_out[fi] = aares.datafiles.read_file(fi)
            pbar.update(1)


    return headers_out#{fi: hd for fi, hd in zip(file_list, get_headers(file_list, nproc=nproc, sort=None))}
