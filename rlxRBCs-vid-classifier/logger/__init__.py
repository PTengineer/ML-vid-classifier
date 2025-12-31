import logging
#import sys
from config import cfg 

def get_logger(name, logfile):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    if log.handlers:
            return log

    file_handler = logging.FileHandler(filename=cfg.log_dir+logfile)
    #console_handler = logging.StreamHandler(sys.stdout)

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt=format_str, datefmt='%Y-%m-%d %H:%M:%S%z')

    file_handler.setFormatter(formatter)

    #console_formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')
    #console_handler.setFormatter(console_formatter)

    log.addHandler(file_handler)
    #log.addHandler(console_handler)

    return log
                                                                                                                    