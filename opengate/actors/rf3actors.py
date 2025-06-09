
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
from radiation_simulation.visualization.two_d import plot_voxel_histograms_2d

MAX_ENERGY = 125
NUM_BINS = 25   # 5 keV steps
BIN_WIDTH = MAX_ENERGY / NUM_BINS

CRF = CartesianRadiationField(vec3(1,1,1), vec3(0.005,0.005,0.005))
CHANNEL = CRF.add_channel("test")
CHANNEL.add_layer("energies", "keV", DType.FLOAT64)
CHANNEL.add_layer("hits", "counts", DType.UINT64)
# CHANNEL.add_layer("mean", "keV", DType.FLOAT64)
# CHANNEL.add_layer("std", "keV", DType.FLOAT64)
CHANNEL.add_histogram_layer("histograms", NUM_BINS, BIN_WIDTH, "keV")

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
        
        self._total_energy = 0
        self.voxel_size = 5
        self.world_limits = np.array([[-500, 500], [-500, 500], [-500, 500]])
        self.energy_grid = CHANNEL.get_layer_as_ndarray("energies")
        self.hits_grid = CHANNEL.get_layer_as_ndarray("hits")
        # self.mean_grid = CHANNEL.get_layer_as_ndarray("mean")
        # self.std_grid = CHANNEL.get_layer_as_ndarray("std")
        self.histogram_grid = CHANNEL.get_layer_as_ndarray("histograms")
        self.histogram_means = np.zeros_like(self.histogram_grid, dtype=np.float64)
        self.histogram_counts = np.zeros_like(self.energy_grid, dtype=np.int64)
        self.histogram_m2 = np.zeros_like(self.histogram_grid, dtype=np.float64) #Maximaum possible error
        self.eps_rel = np.ones_like(self.energy_grid, dtype=np.float64)
        self.variance_means = np.zeros_like(self.histogram_grid, dtype=np.float64)
        
        self.lock = None
        
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
        self.lock = threading.Lock()

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
        energies = np.array(actor.GetEnergy())   #MeV
        if energies.size == 0:
            return  # do nothing if no hits
    
        # Backup old values
        old_hits_grid = self.hits_grid.copy()  # Für new_hits_mask
        # old_histogram_grid = self.histogram_grid.copy()
        # old_histogram_means = self.histogram_means.copy()
        
        # Get values from sim
        pre_positions = np.zeros((energies.size, 3), dtype=np.float64)    #TODO: als numpy array zurückgeben?
        post_positions = np.zeros((energies.size, 3), dtype=np.float64)
        pre_positions[:, 0] = np.array(actor.GetPrePositionX())
        pre_positions[:, 1] = np.array(actor.GetPrePositionY())
        pre_positions[:, 2] = np.array(actor.GetPrePositionZ())
        post_positions[:, 0] = np.array(actor.GetPostPositionX())
        post_positions[:, 1] = np.array(actor.GetPostPositionY())
        post_positions[:, 2] = np.array(actor.GetPostPositionZ())
        positions = np.array([pre_positions, post_positions])   # Shape: (2, num_hits, 3)   
        voxelized_positions = scale_positions_to_voxel_grid(
            self.world_limits,
            positions,
            np.array(self.energy_grid.shape),
        )
        grid_indices, trajectory_ids = bresenham_batch_trajectories(voxelized_positions, progress=False)
        energies_flat = energies[trajectory_ids]
        # Update energies and hits
        np.add.at(self.energy_grid, tuple(grid_indices), energies_flat)
        np.add.at(self.hits_grid, tuple(grid_indices), 1)
          
        bin_indices = np.clip((energies_flat / BIN_WIDTH), 0, NUM_BINS - 1).astype(
            np.uint16
        )
         # 1. FILL HISTOGRAMS WITH HITS
        grid_idx_4d = np.vstack((grid_indices, bin_indices[None, :]))
        np.add.at(self.histogram_grid, tuple(grid_idx_4d), 1)
        
        # 2. CALCULATE NEW MEAN HISTOGRAMS USING HITS_GRID
        new_hits_mask_3d = (self.hits_grid > old_hits_grid) #& self.hits_grid > 0   # Entspricht Anzahl von unique positions
        # new_hits_mask_4d = np.repeat(new_hits_mask_3d[:, :, :, np.newaxis], NUM_BINS, axis=-1)  # 25x mal so viele Einträge, passt
        # print(self.hits_grid.sum(), self.histogram_grid.sum()) # Gleich viele Einträge
        self.histogram_means[new_hits_mask_3d, :] = self.histogram_grid[new_hits_mask_3d, :] / self.hits_grid[new_hits_mask_3d, np.newaxis]
        # print(self.histogram_grid[new_hits_mask_3d, :], self.hits_grid[new_hits_mask_3d, np.newaxis])
        # (
        #         self.histogram_grid[new_hits_mask_4d] / 
        #         self.hits_grid[new_hits_mask_4d][..., np.newaxis][new_hits_mask_4d]
        #     )
        # max_hits_idx = np.unravel_index(np.argmax(self.hits_grid), self.hits_grid.shape)
        # max_hits = self.hits_grid[max_hits_idx]
        # print(f"Voxel mit den meisten Treffern: Index {max_hits_idx}, Treffer: {max_hits}")
        # print(f"Max histogram grid: {self.histogram_grid[max_hits_idx]}")
        # print(f"Max hits grid: {self.hits_grid[max_hits_idx]}")
        # print(f"Max histogram mean: {self.histogram_means[max_hits_idx]}")
        # print(self.histogram_means[max_hits_idx] * self.hits_grid[max_hits_idx])
                
        # Identify bins that were updated
        self.histogram_counts[new_hits_mask_3d] += 1 #TODO: Könnte auch 3D sein, um Speicher zu sparen
        # print(self.histogram_counts.sum(),   # = 25 * histogram_means.sum()
        #       self.hits_grid.sum(),         # Mehrere Hits
        #       self.histogram_grid.sum(),    # Mehrere Hits
        #       self.histogram_means[new_hits_mask_3d, :].sum(),
        #       (self.hits_grid != 0).sum(),  # = histogram_means.sum()
        #       self.histogram_means[new_hits_mask_3d, :].mean()) # = 1/25

        # 3. CALCULATE DELTA
        delta = self.histogram_means[new_hits_mask_3d, :] - self.variance_means[new_hits_mask_3d, :]
        self.variance_means[new_hits_mask_3d, :] += delta / self.histogram_counts[new_hits_mask_3d, np.newaxis]
        delta2 = self.histogram_means[new_hits_mask_3d, :] - self.variance_means[new_hits_mask_3d, :]
        self.histogram_m2[new_hits_mask_3d, :] += delta * delta2
        # print(delta, delta2)
        print(delta.shape, delta2.shape, self.histogram_means.shape)
        # print(f"Max histogram means: {self.histogram_means.max()}")
        print(f"Max histogram m2: {self.histogram_m2.max()}")

        # 4. CALCULATE RELATIVE ERROR
        update_eps_rel_mask = self.histogram_counts > 2  # Only, where histograms were updated 2 or more times
        valid_eps_rel_mask = new_hits_mask_3d & update_eps_rel_mask
        print(new_hits_mask_3d.sum(), valid_eps_rel_mask.sum())
        
        self.eps_rel[valid_eps_rel_mask] = np.sum((self.histogram_m2[valid_eps_rel_mask, :] / self.histogram_counts[valid_eps_rel_mask, np.newaxis]), axis=1)
        self.eps_rel[valid_eps_rel_mask] = (self.eps_rel[valid_eps_rel_mask] / NUM_BINS) * 4
        print(self.eps_rel.min(), self.eps_rel.max(), self.eps_rel.mean())
        # print(self.eps_rel)
        # print("debug", np.sum(self.histogram_m2[valid_eps_rel_mask, :] / self.histogram_counts[valid_eps_rel_mask, np.newaxis], axis=1))
        sorted_eps_rel = np.sort(self.eps_rel, axis=None)
        print(sorted_eps_rel[::20000])
        # print(sorted_eps_rel)
        print("Eps_rel min and max: ", sorted_eps_rel[0], sorted_eps_rel[-1])
        print("Eps_rel 75% Quantil", sorted_eps_rel[int(sorted_eps_rel.size * 0.75)])  # 0.95: 95% Quantil

        self._total_energy += np.sum(energies)
        print(f"Total energy so far: {self._total_energy}, thread ID: {threading.get_ident()}")


    def EndOfRunActionMasterThread(self, run_index):
        return 0    # Wenn kein 0, dann kackt alles ab

    def StartSimulationAction(self):
        DigitizerBase.StartSimulationAction(self)
        g4.GateRF3Actor.StartSimulationAction(self)

    def EndSimulationAction(self):
        with self.lock:
            # print(f"Max energy in grid: {self.energy_grid.max()}")
            # print(f"Max mean in grid: {self.mean_grid.max()}")
            # print(f"Max std in grid: {self.std_grid.max()}")
            # print(np.sum(self.std_grid < 50) / self.std_grid.size)
            # plot_voxel_histograms_2d(self.histogram_grid)
            # plot_3d_heatmap(self.eps_rel, self.world_limits, (self.voxel_size, self.voxel_size, self.voxel_size))
            print(f"Total energy: {self._total_energy}")
            g4.GateRF3Actor.EndSimulationAction(self)
            DigitizerBase.EndSimulationAction(self)
        
        del self.lock   #Cannot be pickled
        # del self.energy_grid

# # 1. Backup old values
# old_hits_grid = self.hits_grid.copy()
# old_mean_grid = self.mean_grid.copy()

# # 2. Accumulate hits and energies
# np.add.at(self.energy_grid, tuple(grid_indices), energies_flat)
# np.add.at(self.hits_grid, tuple(grid_indices), 1)

# # 3. Calculate new Mean
# new_hits_mask = self.hits_grid != old_hits_grid
# # new_hits_mask = new_hits_mask & (self.hits_grid != 0)
# self.mean_grid[new_hits_mask] = self.energy_grid[new_hits_mask] / self.hits_grid[new_hits_mask]

# print("Max values after accumulation:")
# print(self.energy_grid.max())
# print(self.hits_grid.max())
# print(self.mean_grid.max())

# # 4. Welford method for standard deviation
# # delta = x_n - mean_{n-1}
# delta = np.zeros_like(self.energy_grid, dtype=np.float64)
# delta_values = energies_flat - old_mean_grid[tuple(grid_indices)]
# np.add.at(delta, tuple(grid_indices), delta_values)

# print("Max values after delta calculation:")
# print(delta_values.max())
# print(delta.max())

# # delta2 = x_n - mean_n 
# delta2 = np.zeros_like(self.energy_grid, dtype=np.float64)
# delta2_values = energies_flat - self.mean_grid[tuple(grid_indices)]
# np.add.at(delta2, tuple(grid_indices), delta2_values)

# print("Max values after delta2 calculation:")
# print(delta2_values.max())
# print(delta2.max())

# print(self.std_grid.max())
# # Update M2: M2_n = M2_{n-1} + delta * delta2
# self.std_grid[new_hits_mask] += delta[new_hits_mask] * delta2[new_hits_mask]
# print(self.std_grid.max())
# print(np.isnan(self.std_grid).sum())

# # Calculate standard deviation
# std_mask = new_hits_mask & (self.hits_grid > 1)  # Avoid division by zero
# self.std_grid[std_mask] = np.sqrt(np.maximum(self.std_grid[std_mask], 0) / self.hits_grid[std_mask]) #Negative $ M2-Werte sind Artefakte
# print(np.isnan(self.std_grid).sum())

process_cls(RF3Actor)