import logging

def log_number_of_frames(files_dict, name=None):
    out_str = 'Number of frames in the individual files:\n'
    for key, val in files_dict.items():
        out_str += '{name} : {frames}\n'.format(name= key, frames=val.number_of_frames)
    logging.info(out_str)
