from abc import ABC, abstractmethod
import equinox as eqx
from dataclasses import dataclass
from .self_adaptive import SAHparams, SelfAdaptive
#from ._energynet import EnergyHparams, EnergyNet
from typing import Optional

class AbstractOperatorNet(eqx.Module, ABC):
	"""
	An abstract base class for all Operator Networks. 
	This defines the common interface for all OperatorNetwork types.
	"""
	self_adaptive: Optional[SelfAdaptive]
	is_self_adaptive: bool
	period: float
	
	# add z_score_data 
	u_std: float
	u_mean: float
	x_std: float
	x_mean: float
	t_std: float
	t_mean: float

	@abstractmethod
	def __init__(self, hparams):
		#checks whether the hparams for self-adaptive weights are set
		self.is_self_adaptive = hparams.λ_learning_rate is not None
		if self.is_self_adaptive:
			self.self_adaptive = SelfAdaptive(hparams)
		else:
			self.self_adaptive = None
   
		self.u_std = hparams.u_std
		self.u_mean = hparams.u_mean
		self.x_std = hparams.x_std
		self.x_mean = hparams.x_mean
		self.t_std = hparams.t_std
		self.t_mean = hparams.t_mean
  
		self.period = 20./self.x_std

	@abstractmethod
	def __call__(self, a, x, t):
		"""Computes the prediction of the network"""
		pass

	@abstractmethod
	def predict_whole_grid(self, a, x, t):
		"""Not all operator nets take array input for x and t.
  		This function predicts over the whole grid"""
		pass

	@abstractmethod
	def predict_whole_grid_batch(self, a, x, t):
		pass

	def encode_u(self, u):
		"""Encodes the output of the network"""
		return (u - self.u_mean) / self.u_std

	def decode_u(self, u):
		return u * self.u_std + self.u_mean

	def encode_x(self, x):
		return (x - self.x_mean) / self.x_std

	def decode_x(self, x):
		return x * self.x_std + self.x_mean

	def encode_t(self, t):
		return (t - self.t_mean) / self.t_std

	def decode_t(self, t):
		return t * self.t_std + self.t_mean
	
@dataclass(kw_only=True, frozen=True)
class AbstractHparams(SAHparams):#, EnergyHparams):
	"""Specifies the hyperparameters of an abstract network.
	These are parameters that are common to all networks.
	"""
	seed: int = 0 # seed for reproducibility
	batch_size: int = 16 # batch size
	learning_rate: float = 1e-3 # learning rate for the "regular" network parameters
	optimizer: str = "adam" # optimizer to use for the "regular" network parameters
 
	#z-score data
	u_std: float = 1.0
	u_mean: float = 0.0
	x_std: float = 1.0
	x_mean: float = 0.0
	t_std: float = 1.0
	t_mean: float = 0.0