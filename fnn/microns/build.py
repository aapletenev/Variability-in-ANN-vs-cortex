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


def network(units):
    """
    Parameters
    ----------
    units : int
        number of units

    Returns
    -------
    fnn.model.networks.Visual
        visual neural network
    """
    # core component
    feedforward = InputDense(
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
    recurrent = CvtLstm(
        in_channels=256,
        out_channels=128,
        hidden_channels=256,
        common_channels=512,
        groups=8,
        spatial=3,
    )
    # 3rd layer of ann
    core = FeedforwardRecurrent(
        feedforward=feedforward,
        recurrent=recurrent,
    )
    # 1st layer of ann, infers perspective from retina
    perspective = MlpMonitorRetina(
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
    modulation = MlpLstm(
        mlp_features=16,
        mlp_layers=1,
        mlp_nonlinear="gelu",
        lstm_features=16,
    )
    # 4th layer of ann, maps core's outputs onto activity of indivudual neurons 
    readout = PositionFeature(
        position=Gaussian(),
        bound=Tanh(),
        feature=Vanilla(),
    )


    network = Visual(
        core=core,
        perspective=perspective,
        modulation=modulation,
        readout=readout,
        reduce=Mean(),
        unit=Poisson(),
    )
    network._init(
        stimuli=1,
        perspectives=2,
        modulations=2,
        streams=4,
        units=units,
    )
    return network
