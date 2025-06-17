# made by josh to work with model
# test_network also made by josh, right now is just decimal output from download

import torch
import os
import microns
from microns.build import network
from fnn.model.feedforwards import InputDense
from fnn.model.recurrents import CvtLstm
from fnn.model.cores import FeedforwardRecurrent
from fnn.model.monitors import Plane
from fnn.model.pixels import StaticPower, SigmoidPower
from fnn.model.retinas import Angular
from fnn.model.perspectives import MlpMonitorRetina
from fnn.model.modulations import MlpLstm
from fnn.model.positions import Gaussian
from fnn.model.bounds import Tanh
from fnn.model.features import Vanilla
from fnn.model.readouts import PositionFeature
from fnn.model.reductions import Mean
from fnn.model.units import Poisson
from fnn.model.networks import Visual



class model(torch.nn.Module):
    def __init__(self, units = 3):
        super().__init__()
        self.feedforward = InputDense(
            input_spatial=6,
            input_stride=2,
            block_channels=[32, 64, 128],
            block_groups=[1, 2, 4],
            block_layers=[2, 2, 2],
            block_temporals=[3, 3, 3],
            block_spatials=[3, 3, 3],
            block_pools=[2, 2, 1],
            out_channels=128,
            nonlinear="gelu",
        )
        self.recurrent = CvtLstm(
            in_channels=256,
            out_channels=128,
            hidden_channels=256,
            common_channels=512,
            groups=8,
            spatial=3,
        )
        self.core = FeedforwardRecurrent(
            feedforward=self.feedforward,
            recurrent=self.recurrent,
        )
        self.perspective = MlpMonitorRetina(
            mlp_features=16,
            mlp_layers=3,
            mlp_nonlinear="gelu",
            height=128,
            width=192,
            monitor=Plane(),
            monitor_pixel=StaticPower(power=1.7),
            retina=Angular(degrees=75),
            retina_pixel=SigmoidPower(),
        )
        self.modulation = MlpLstm(
            mlp_features=16,
            mlp_layers=1,
            mlp_nonlinear="gelu",
            lstm_features=16,
        )
        self.readout = PositionFeature(
            position=Gaussian(),
            bound=Tanh(),
            feature=Vanilla(),
        )
        self.network = Visual(
            core=self.core,
            perspective=self.perspective,
            modulation=self.modulation,
            readout=self.readout,
            reduce=Mean(),
            unit=Poisson(),
        )
        network.__init__(
            stimuli=1,
            perspectives=2,
            modulations=2,
            streams=4,
            units=units,
        )

path_to_pt = os.path.join(os.getcwd(), "data", "microns", "params_4_7.pt")
my_model = model()
my_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "microns"), map_location="cpu")) # use cpu if no cuda
