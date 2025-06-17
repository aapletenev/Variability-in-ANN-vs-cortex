# save model script from josh, still in works

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
import torch

class network_class():
    """
    Parameters
    ----------
    units : int
        number of units (josh defaulted to three)

    Returns
    -------
    fnn.model.networks.Visual
        visual neural network
    """
    def __init__(self):
        super(network_class, self).__init__()
        # core component
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
        # core component, transforms input from perspective+modulation to produce feature representations
        self.recurrent = CvtLstm(
            in_channels=256,
            out_channels=128,
            hidden_channels=256,
            common_channels=512,
            groups=8,
            spatial=3,
        )
        # 3rd layer of ann
        self.core = FeedforwardRecurrent(
            feedforward=self.feedforward,
            recurrent=self.recurrent,
        )
        # 1st layer of ann, infers perspective from retina
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
        # 2nd layer, transforms behavioral var. to produce dynamic state of mouse
        self.modulation = MlpLstm(
            mlp_features=16,
            mlp_layers=1,
            mlp_nonlinear="gelu",
            lstm_features=16,
        )
        # 4th layer of ann, maps core's outputs onto activity of indivudual neurons 
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
        network_class._init(
            stimuli=1,
            perspectives=2,
            modulations=2,
            streams=4,
            units=self.units,
        )

model = network_class()

# Initialize optimizer
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

""""
# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    """