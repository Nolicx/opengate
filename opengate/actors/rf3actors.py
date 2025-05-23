
import numpy as np
import threading
import numpy as np
import opengate_core as g4

from ..exception import fatal
from .base import ActorBase

from ..base import process_cls

from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType

from radiation_simulation.analysis.calculations import bresenham_batch_trajectories, generate_voxel_grid, scale_positions_to_voxel_grid
from radiation_simulation.visualization.three_d import plot_3d_heatmap
# crf = CartesianRadiationField(vec3(1,1,1), vec3(10,10,10))
# channel = crf.add_channel("test")
# channel.add_layer("test", "keV", DType.FLOAT32)

class RF3Actor(ActorBase, g4.GateRF3Actor):

    user_info_defaults = {
        "batch_size": (
            2e5,
            {
                "doc": "FIXME",
            },
        ),
        "output": (
            None,
            {
                "doc": "FIXME",
            },
        ),
    }

    def __init__(self, *args, **kwargs) -> None:
        ActorBase.__init__(self, *args, **kwargs)
        self.__initcpp__()
        
        self._total_hits = 0
        self._total_energy = 0
        self.voxel_size = 5
        self.world_limits = np.array([[-500, 500], [-500, 500], [-500, 500]])
        self.voxel_grid = generate_voxel_grid(self.world_limits, self.voxel_size)
        self.lock = None
        

    def __initcpp__(self)-> None:
        g4.GateRF3Actor.__init__(self, self.user_info)
        self.AddActions(
            {
                "SteppingAction",
                "BeginOfRunActionMasterThread",
                "EndOfRunActionMasterThread",
                "BeginOfRunAction",
                "EndOfRunAction",
            }
        )
        
    def __getstate__(self)-> dict:
        # needed to not pickle objects that cannot be pickled (g4, cuda, lock, etc).
        return_dict = super().__getstate__()
        return return_dict

    def initialize(self):
        # call the initialize() method from the super class (python-side)
        ActorBase.initialize(self)
        self.lock = threading.Lock()

        # initialize C++ side
        self.InitializeUserInfo(self.user_info)
        self.InitializeCpp()
        self.SetCallbackFunction(self.apply)

    def apply(self, actor):
        # we need a lock when the CallbackFunction is called from the C++ side
        if self.simulation.use_multithread:
            with self.lock:
                self.process_data(actor)
        else:
            self.process_data(actor)

    def process_data(self, actor):

        # get values from cpp side
        energy = np.array(actor.GetEnergy())
        pre_pos_x = np.array(actor.GetPrePositionX())
        pre_pos_y = np.array(actor.GetPrePositionY())
        pre_pos_z = np.array(actor.GetPrePositionZ())
        post_pos_x = np.array(actor.GetPostPositionX())
        post_pos_y = np.array(actor.GetPostPositionY())
        post_pos_z = np.array(actor.GetPostPositionZ())
        
        pre_positions = np.array([pre_pos_x, pre_pos_y, pre_pos_z]).T  # Shape (50, 3)
        post_positions = np.array([post_pos_x, post_pos_y, post_pos_z]).T  # Shape (50, 3)
        positions = np.array([pre_positions, post_positions])  # Shape (2, 50, 3)        
        
        voxelized_positions = scale_positions_to_voxel_grid(
            self.world_limits,
            positions,
            np.array(self.voxel_grid.shape),
        )
        
        grid_positions, trajectory_ids = bresenham_batch_trajectories(voxelized_positions, progress=False)
        energy_flat = energy[trajectory_ids]
        np.add.at(self.voxel_grid, tuple(grid_positions), energy_flat)
        
        # num_hits = np.array(actor.GetCurrentNumberOfHits())

        # self._total_hits += num_hits
        # self._total_energy += np.sum(energy)
        
        # do nothing if no hits
        if energy.size == 0:
            return

    def EndOfRunActionMasterThread(self, run_index):
        return 0    # Wenn kein 0, dann kackt alles ab

    def EndSimulationAction(self):
        with self.lock:
            print(self.voxel_grid.max())
            print(self.voxel_grid)
            plot_3d_heatmap(self.voxel_grid, self.world_limits, (self.voxel_size, self.voxel_size, self.voxel_size))
        
        g4.GateRF3Actor.EndSimulationAction(self)
        ActorBase.EndSimulationAction(self)
        del self.lock   #Cannot be pickled
        del self.voxel_grid

process_cls(RF3Actor)