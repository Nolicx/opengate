
import numpy as np
import threading
import numpy as np
import opengate_core as g4
import gc
from line_profiler import profile

from ..exception import fatal
from .base import ActorBase
from .digitizers import DigitizerBase

from ..base import process_cls

from RadFiled3D.RadFiled3D import CartesianRadiationField, vec3, DType

from radiation_simulation.analysis.calculations import bresenham_batch_trajectories, generate_voxel_grid, scale_positions_to_voxel_grid
from radiation_simulation.visualization.three_d import plot_3d_heatmap
from radiation_simulation.visualization.two_d import plot_voxel_histograms_2d

CRF = CartesianRadiationField(vec3(1,1,1), vec3(0.005,0.005,0.005))
CHANNEL = CRF.add_channel("test")
CHANNEL.add_layer("energies", "keV", DType.FLOAT64)
CHANNEL.add_layer("hits", "counts", DType.UINT64)


class RF3Actor(DigitizerBase, g4.GateRF3Actor):

    user_info_defaults = {
        "batch_size": (
            100.000,
            {
                "doc": "FIXME",
            },
        ),
        "threshold": (
            0.100,
            {
                "doc": "FIXME",
            },
        ),
        "percentile": (
            0.950,
            {
                "doc": "FIXME",
            },
        ),
        "max_energy": (
            125,
            {
                "doc": "FIXME",
            },
        ),
        "num_bins": (
            5,
            {
                "doc": "FIXME",
            },
        ),
    }

    def __init__(self, *args, **kwargs) -> None:
        DigitizerBase.__init__(self, *args, **kwargs)
        
        self.voxel_size = 5
        self.world_limits = np.array([[-500, 500], [-500, 500], [-500, 500]])
        self.energy_grid = CHANNEL.get_layer_as_ndarray("energies")
        self.hits_grid = CHANNEL.get_layer_as_ndarray("hits")
        
        self.max_energy = self.user_info["max_energy"] / 1000 # Convert to MeV
        self.num_bins = self.user_info["num_bins"]
        self.bin_width = self.max_energy / self.num_bins
        
        CHANNEL.add_histogram_layer("histograms", self.num_bins, self.bin_width, "MeV")
        self.histogram_grid = CHANNEL.get_layer_as_ndarray("histograms")
        self.histogram_means = np.zeros_like(self.histogram_grid, dtype=np.float16) #TODO: Schneller?
        self.histogram_counts = np.zeros_like(self.energy_grid, dtype=np.int64)
        self.histogram_m2 = np.zeros_like(self.histogram_grid, dtype=np.float16) 
        self.eps_rel = np.ones_like(self.energy_grid, dtype=np.float16)    # Maximum possible error     
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
        self.SetCallbackFunction(self.process_data)
        self.lock = threading.Lock()

    # def __getstate__(self)-> dict:
    #     # needed to not pickle objects that cannot be pickled (g4, cuda, lock, etc).
    #     return_dict = super().__getstate__()
    #     return return_dict

    @profile
    def process_data(self, actor):
        # get values from cpp side
        energies = actor.GetEnergy() #MeV
        if energies.size == 0:
            return  # do nothing if no hits
    
        # Backup old values
        with self.lock:
            old_hits_grid = self.hits_grid.copy()  # For creating new_hits_mask
        
        # Get values from sim
        pre_positions = actor.GetPrePosition()
        post_positions = actor.GetPostPosition()
        positions = np.array([pre_positions, post_positions])   # Shape: (2, num_hits, 3)   
        voxelized_positions = scale_positions_to_voxel_grid(
            self.world_limits,
            positions,
            np.array(self.energy_grid.shape),
        )
        grid_indices, trajectory_ids = bresenham_batch_trajectories(voxelized_positions, progress=False)
        # grid_indices = tuple(grid_indices)  
        energies_flat = energies[trajectory_ids]
        bin_indices = np.clip((energies_flat / self.bin_width), 0, self.num_bins - 1).astype(
            np.uint16
        )
        grid_idx_4d = np.vstack((grid_indices, bin_indices[None, :]))
        # grid_idx_4d = tuple(grid_idx_4d)
        
        # with self.lock:
        print("Acquired lock with thread:", threading.get_ident())
        # 1. FILL HISTOGRAMS , ENERGIES AND HITS WITH HITS 
        np.add.at(self.energy_grid, grid_indices, energies_flat)
        np.add.at(self.hits_grid, grid_indices, 1)
        np.add.at(self.histogram_grid, grid_idx_4d, 1)
    
        # 2. CALCULATE NEW MEAN HISTOGRAMS USING HITS_GRID
        new_hits_mask_3d = (self.hits_grid != old_hits_grid) & self.hits_grid > 0   # Entspricht Anzahl von unique positions
        self.histogram_means[new_hits_mask_3d, :] = self.histogram_grid[new_hits_mask_3d, :] / self.hits_grid[new_hits_mask_3d, np.newaxis]

        # 3. CALCULATE DELTA
        delta = self.histogram_means[new_hits_mask_3d, :] - self.variance_means[new_hits_mask_3d, :]
        self.variance_means[new_hits_mask_3d, :] += delta / self.histogram_counts[new_hits_mask_3d, np.newaxis]
        # delta2 = self.histogram_means[new_hits_mask_3d, :] - self.variance_means[new_hits_mask_3d, :]
        self.histogram_m2[new_hits_mask_3d, :] += delta * (self.histogram_means[new_hits_mask_3d, :] - self.variance_means[new_hits_mask_3d, :])
        
        # 4. CALCULATE RELATIVE ERROR
        update_eps_rel_mask = self.histogram_counts > 2  # Only, where histograms were updated 2 or more times
        valid_eps_rel_mask = new_hits_mask_3d & update_eps_rel_mask
        
        self.eps_rel[valid_eps_rel_mask] = np.sum(
            (self.histogram_m2[valid_eps_rel_mask, :] / self.histogram_counts[valid_eps_rel_mask, np.newaxis]), axis=1
            ) * (4 / self.num_bins) 
        # self.eps_rel[valid_eps_rel_mask] = (self.eps_rel[valid_eps_rel_mask] / self.num_bins) * 4

        percentile = self.user_info["percentile"]
        percentile_idx = int(self.eps_rel.size * percentile)
        sorted_eps_rel = np.partition(self.eps_rel.ravel(), percentile_idx)
        print(sorted_eps_rel[::20000])
        print(f"Eps_rel {percentile * 100}% Quantil {sorted_eps_rel[percentile_idx]}")
        
        threshhold = self.user_info["threshold"]
        print(f"Threshold cleared: {sorted_eps_rel[percentile_idx] <= threshhold}")
        if sorted_eps_rel[percentile_idx] <= threshhold:
            g4.GateRF3Actor.StopSimulation(self)
        print("Releasing lock with thread:", threading.get_ident())


            # max_hits_idx = np.unravel_index(np.argmax(self.hits_grid), self.hits_grid.shape)
            # max_hits = self.hits_grid[max_hits_idx]
            # print(f"Voxel mit den meisten Treffern: Index {max_hits_idx}, Treffer: {max_hits}")
            # print(f"Max histogram grid: {self.histogram_grid[max_hits_idx]}")
            # print(f"Max hits grid: {self.hits_grid[max_hits_idx]}")
            # print(f"Max histogram mean: {self.histogram_means[max_hits_idx]}")
            # print("Means * hits: ", self.histogram_means[max_hits_idx] * self.hits_grid[max_hits_idx])
                    
            # # Identify bins that were updated
            # self.histogram_counts[new_hits_mask_3d] += 1 
            # # print(self.histogram_counts.sum(),   # = 25 * histogram_means.sum()
            # #       self.hits_grid.sum(),         # Mehrere Hits
            # #       self.histogram_grid.sum(),    # Mehrere Hits
            # #       self.histogram_means[new_hits_mask_3d, :].sum(),
            # #       (self.hits_grid != 0).sum(),  # = histogram_means.sum()
            # #       self.histogram_means[new_hits_mask_3d, :].mean()) # = 1/25
            # print(f"Max histogram counts: {self.histogram_counts[max_hits_idx]}")

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
            
            print(f"Generated Photons: {g4.GateRF3Actor.GetNumberOfAbsorbedEvents(self)}")
            print(f"Registered Hits: {g4.GateRF3Actor.GetNumberOfHits(self)}")
            g4.GateRF3Actor.EndSimulationAction(self)
            DigitizerBase.EndSimulationAction(self)
            # plot_3d_heatmap(self.energy_grid, self.world_limits, (self.voxel_size, self.voxel_size, self.voxel_size))
        
            indices_per_axis = np.linspace(25, 175, 3, dtype=int)
            x_indices, y_indices, z_indices = np.meshgrid(indices_per_axis, indices_per_axis, indices_per_axis, indexing='ij')

            indices_array = np.column_stack([x_indices.ravel(), y_indices.ravel(), z_indices.ravel()])
            for idx in indices_array:
                plot_voxel_histograms_2d(self.histogram_grid, "tests/test_data/rf3actor_test_histograms", idx)
        
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