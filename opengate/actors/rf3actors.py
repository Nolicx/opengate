
import numpy as np
import threading
import numpy as np
import opengate_core as g4
import gc

from ..exception import fatal
from .base import ActorBase
from .digitizers import DigitizerBase

from ..base import process_cls

from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType

from radiation_simulation.analysis.calculations import bresenham_batch_trajectories, generate_voxel_grid, scale_positions_to_voxel_grid
from radiation_simulation.visualization.three_d import plot_3d_heatmap
CRF = CartesianRadiationField(vec3(1,1,1), vec3(0.005,0.005,0.005))
CHANNEL = CRF.add_channel("test")
CHANNEL.add_layer("energies", "keV", DType.FLOAT64)
CHANNEL.add_layer("hits", "counts", DType.UINT64)
CHANNEL.add_layer("mean", "keV", DType.FLOAT64)
CHANNEL.add_layer("std", "keV", DType.FLOAT64)

class RF3Actor(DigitizerBase, g4.GateRF3Actor):

    user_info_defaults = {
        "batch_size": (
            2e5,
            {
                "doc": "FIXME",
            },
        ),
        # "output": (
        #     None,
        #     {
        #         "doc": "FIXME",
        #     },
        # ),
    }

    def __init__(self, *args, **kwargs) -> None:
        DigitizerBase.__init__(self, *args, **kwargs)
        
        # self._total_hits = 0
        self._total_energy = 0
        self.voxel_size = 5
        self.world_limits = np.array([[-500, 500], [-500, 500], [-500, 500]])
        self.energy_grid = CHANNEL.get_layer_as_ndarray("energies")
        self.hits_grid = CHANNEL.get_layer_as_ndarray("hits")
        self.mean_grid = CHANNEL.get_layer_as_ndarray("mean")
        self.std_grid = CHANNEL.get_layer_as_ndarray("std")
        # self.lock = None
        
        self.__initcpp__()
        

    def __initcpp__(self)-> None:
        g4.GateRF3Actor.__init__(self, self.user_info)
        # self.AddActions(
        #     {
        #         "SteppingAction",
        #         "BeginOfRunActionMasterThread",
        #         "EndOfRunActionMasterThread",
        #         "BeginOfRunAction",
        #         "EndOfRunAction",
        #     }
        # )

    def initialize(self):
        # call the initialize() method from the super class (python-side)
        DigitizerBase.initialize(self)

        # initialize C++ side
        self.InitializeUserInfo(self.user_info)
        self.InitializeCpp()
        self.SetCallbackFunction(self.apply)
        # self.lock = threading.Lock()

    # def __getstate__(self)-> dict:
    #     # needed to not pickle objects that cannot be pickled (g4, cuda, lock, etc).
    #     return_dict = super().__getstate__()
    #     return return_dict

    #TODO: One lock seems to be enough
    def apply(self, actor):
        # we need a lock when the CallbackFunction is called from the C++ side
        # if self.simulation.use_multithread:
        #     with self.lock:
        #         print(f"Acquired lock in Python apply, thread ID: {threading.get_ident()}")
        #         self.process_data(actor)
        #         print(f"Released lock in Python apply, thread ID: {threading.get_ident()}")
        # else:
        self.process_data(actor)

    def process_data(self, actor):
        # get values from cpp side
        energy = np.array(actor.GetEnergy())
        print(f"Energy size: {energy.size}, thread ID: {threading.get_ident()}")
        if energy.size == 0:
            return  # do nothing if no hits
        
        pre_positions = np.zeros((energy.size, 3), dtype=np.float64)
        post_positions = np.zeros((energy.size, 3), dtype=np.float64)
        pre_positions[:, 0] = np.array(actor.GetPrePositionX(), copy = True)
        pre_positions[:, 1] = np.array(actor.GetPrePositionY(), copy = True)
        pre_positions[:, 2] = np.array(actor.GetPrePositionZ(), copy = True)
        post_positions[:, 0] = np.array(actor.GetPostPositionX(), copy = True)
        post_positions[:, 1] = np.array(actor.GetPostPositionY(), copy = True)
        post_positions[:, 2] = np.array(actor.GetPostPositionZ(), copy = True)
        positions = np.array([pre_positions, post_positions])   # Shape: (2, num_hits, 3)   
        
        voxelized_positions = scale_positions_to_voxel_grid(
            self.world_limits,
            positions,
            np.array(self.energy_grid.shape),
        )
        print("Voxelized positions shape:", voxelized_positions.shape, f"thread ID: {threading.get_ident()}")  # Should be (2, 50, 3)
        grid_positions, trajectory_ids = bresenham_batch_trajectories(voxelized_positions, progress=False)
        print("Grid positions shape:", grid_positions.shape, f"thread ID: {threading.get_ident()}")  # Should be (N, 3)
        energy_flat = energy[trajectory_ids]
        print(f"Energies flat: {energy_flat.shape}, thread ID: {threading.get_ident()}")  # Should be (N,)
        np.add.at(self.energy_grid, tuple(grid_positions), energy_flat)
        # np.add.at(self.hits_grid, tuple(grid_positions), 1)
        
        # for voxel_idx in grid_positions:
        #         i, j, k = voxel_idx
        #         idx = (i, j, k)
        #         # Alle Energiewerte für dieses Voxel sammeln
        #         for e in energy_flat:
        #             # Welfords Methode
        #             old_n = self.self.hits_grid[idx]
        #             old_mean = self.mean_grid[idx]
        #             old_M2 = self.std_grid[idx] ** 2 * (old_n - 1) if old_n > 1 else 0
        #             new_n = old_n + 1
        #             delta = e - old_mean
        #             new_mean = old_mean + delta / new_n
        #             new_M2 = old_M2 + delta * (e - new_mean)
        #             # Aktualisiere Grids
        #             self.self.hits_grid[idx] = new_n
        #             self.mean_grid[idx] = new_mean
        #             self.std_grid[idx] = np.sqrt(new_M2 / (new_n - 1)) if new_n > 1 else 0
        
        # # num_hits = np.array(actor.GetCurrentNumberOfHits())

        # self._total_hits += num_hits
        self._total_energy += np.sum(energy)
        print(f"Total energy so far: {self._total_energy}, thread ID: {threading.get_ident()}")


    def EndOfRunActionMasterThread(self, run_index):
        return 0    # Wenn kein 0, dann kackt alles ab

    def StartSimulationAction(self):
        DigitizerBase.StartSimulationAction(self)
        g4.GateRF3Actor.StartSimulationAction(self)

    def EndSimulationAction(self):
        # with self.lock:
        print(self.energy_grid.max())
        print(self.energy_grid)
            # plot_3d_heatmap(self.energy_grid, self.world_limits, (self.voxel_size, self.voxel_size, self.voxel_size))
        # print(f"Total hits: {self._total_hits}")
        print(f"Total energy: {self._total_energy}")
        g4.GateRF3Actor.EndSimulationAction(self)
        DigitizerBase.EndSimulationAction(self)
        
        # del self.lock   #Cannot be pickled
        # del self.energy_grid

process_cls(RF3Actor)