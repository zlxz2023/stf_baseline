from .GTS.supervisor import GTSSupervisor
from .GWNet.supervisor import GraphWaveNetSupervisor
from .AGCRN.supervisor import AGCRNSupervisor
from .STAEformer.supervisor import STAEformerSupervisor
from .STID.supervisor import STIDSupervisor
from .STWave.supervisor import STWaveSupervisor


supervisor_dict = {
    "GTS": GTSSupervisor,
    "GWNet": GraphWaveNetSupervisor,
    "AGCRN": AGCRNSupervisor,
    "STAEformer": STAEformerSupervisor,
    "STID": STIDSupervisor,
    "STWave": STWaveSupervisor,
}