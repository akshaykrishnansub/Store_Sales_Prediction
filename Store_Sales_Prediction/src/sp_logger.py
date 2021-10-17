import logging
from logging.handlers import RotatingFileHandler
from logging import handlers
import sys
import config


def print_loggers(app_name):
    log = logging.getLogger(app_name)
    log.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - [%(name)s] - [%(levelname)s] - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    fh = handlers.RotatingFileHandler(config.LOGFILE, maxBytes=(10000), backupCount=10)
    fh.setFormatter(format)
    log.addHandler(fh)
    return log