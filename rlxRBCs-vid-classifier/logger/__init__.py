import logging
from config import cfg 

def get_logger(name, logfile):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    handler = logging.FileHandler(filename=cfg.log_dir+logfile)

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt=format_str, datefmt='%Y-%m-%d %H:%M:%S%z')

    handler.setFormatter(formatter)

    log.addHandler(handler)

    return log
                                                                                                                    