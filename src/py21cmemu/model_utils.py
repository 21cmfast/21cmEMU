import torch
from . import sde as sde_lib
from torch.utils.data import Dataset

def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, t, x_cdn=None, cdn=None):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, t, x_cdn=x_cdn, cdn=cdn)
    else:
      model.train()
      return model(x, t, x_cdn=x_cdn, cdn=cdn)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t, x_cdn=None, cdn=None):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        time = t * 999
        score = model_fn(x, time, x_cdn=x_cdn, cdn=cdn)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        time = t * (sde.N - 1)
        score = model_fn(x, time, x_cdn=x_cdn, cdn=cdn)
        std = sde.sqrt_one_minus_alphas_cumprod.to(x.device)[time.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t, x_cdn=None, cdn=None):
      if continuous:
        time = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        time = sde.T - t
        time *= sde.N - 1
        time = torch.round(time).long()

      score = model_fn(x, time, x_cdn=x_cdn, cdn=cdn)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

class PS_Dataset(Dataset):
    def __init__(self, labels, cond):
        self.labels = labels
        self.cond=cond
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        cond = self.cond[idx]
        return label, cond