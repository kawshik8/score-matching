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
from torchvision.models.inception import inception_v3
from torchvision import transforms
from scipy.stats import entropy
from torch.autograd import Variable
from scipy import linalg

EPSILON = 1e-8

################ ~ 9.35 +/- 0.53 for cifar 10 with 5000 original images and splits = 10
def inception_score(inception_model, images, cuda=True, batch_size=32, resize=True, splits=10, fid_mean=0, fid_covar=0, mnist=False): 
    """Computes the inception score of the generated images image_set
    image_set -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(images)

    #assert batch_size > 0
    #assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size)

    # Load inception model
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    # transform = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    
    def get_pred(x):
        if resize:
            x = up(x)
            if mnist:
                x = torch.cat([x,x,x],dim=1)
        features, logits = inception_model(x)
        return features.detach().data.cpu().numpy(), F.softmax(logits,dim=-1).detach().data.cpu().numpy()

    # Get predictions
    features = np.zeros((N,2048))
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
    # i = 0
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        op = get_pred(batchv)
        features[i*batch_size:i*batch_size + batch_size_i] = op[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = op[1]

    mean = np.mean(features,axis=0)
    covar = np.cov(features,rowvar=False)

    print(np.unique(covar))
    print(np.unique(fid_covar))

    fid = calculate_frechet_distance(mean, covar, fid_mean, fid_covar)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores), fid
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


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

if __name__=='__main__':
    import args
    args = args.process_args()
    import data
    test_data = data.Data(args, 'test')

    r = torch.normal(mean=10,std=1,size=[64,2048]).numpy()
    m = np.mean(r,axis=0)
    c = np.cov(r,rowvar=False)
    print(m.shape,c.shape,(m+1).shape,(c+1).shape)
    print(calculate_frechet_distance(m,c,m+1e-1,c+1e-1))
    loader = torch.utils.data.DataLoader(test_data,5000)
    for i,batch in enumerate(loader):
        if i > 1:
            break
        # print(torch.unique(batch))
        batch = (batch * 2) - 1
        print(inception_score(batch, False, 8, True, 10))
    # print(test_data[0].unsqueeze(0).shape)
    # print(inception_score(test_data[0:10])

