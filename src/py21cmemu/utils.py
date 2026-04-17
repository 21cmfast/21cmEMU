import torch
import numpy as np
from pathlib import Path
import logging

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.cpu().gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def transform(x, scale, bias):
    d = torch.log10(x)
    unit = (d - bias) / scale
    y =  unit * 2 - 1 # scale to [-1, 1]
    return y

def reverse_transform(y, scale, bias):
    unit = (y + 1) / 2
    d = unit * scale + bias
    return 10**d

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def restore_checkpoint(ckpt_dir, state, device):
  ckpt_path = Path(ckpt_dir)
  if not ckpt_path.exists():
    ckpt_path.mkdir(exist_ok = True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)

