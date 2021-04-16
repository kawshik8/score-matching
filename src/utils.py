import logging as log
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import logging
import sys
from time import strftime, gmtime
from pdb import set_trace as bp

EPSILON = 1e-8


def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log


def load_model(load_ckpt, model):
    """
    Load a model, for training, evaluation or prediction
    """
    model_state = torch.load(load_ckpt)
    model.load_state_dict(model_state)
    log.info("Load parameters from %s" % load_ckpt)


def save_model(save_ckpt, model):
    """
    Save the parameters of the model to a checkpoint
    """
    torch.save(model.state_dict(), save_ckpt)
    log.info("Save parameters for %s" % save_ckpt)
