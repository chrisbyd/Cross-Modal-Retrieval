import math
import json
from typing import NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class Uniformer(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    