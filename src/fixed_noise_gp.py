import torch
import gpytorch as gp
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ZeroMean
from botorch import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import FixedNoiseGP


def get_gp(X, Y, Yvar):

    '''
    Fixed noise GP: observation noise is known and fixed for each data point
    Used to handle input-dependent noise in MD simulations
    '''
    
    if X.ndim < 2:
        print("Need to specify as matrix of size ntrain by ninputs")

    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)

    input_dim = X.shape[-1]

    outcome_transform = Standardize(m=Y.shape[-1])

    model = FixedNoiseGP(
        train_X=X, 
        train_Y=Y, 
        train_Yvar=Yvar, 
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=input_dim)), 
        mean_module=ZeroMean(batch_shape=torch.Size([])),
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=outcome_transform,
    )
    
    return model


def train_gp(model):
    # model.outcome_transform.eval()
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    mll.train()
    fit_gpytorch_model(mll, num_restarts=10, options={'maxiter': 2000})
    model.eval()
    return model


def predict(model, test_X):
    with torch.no_grad():
        pred = model.posterior(test_X)
        pred_mean = pred.mean.squeeze()
        pred_var = pred.variance.squeeze()
    return pred_mean, pred_var
