import datetime
import logging
import os


def mylog(model_type):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logfile = "./%s_log.txt" % model_type
    fileHandler = logging.FileHandler(logfile, mode='a', encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger
