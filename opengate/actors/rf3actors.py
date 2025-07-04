import threading

import numpy as np
import opengate_core as g4
from RadFiled3D.RadFiled3D import CartesianRadiationField, DType, vec3

from simdos.analysis.utils import store_rf3_file
from simdos.calculations import (
    dda_batch_raycast,
)
from simdos.calculations.update_voxel_grid import (
    evaluate_relative_errors,
    update_grids_numba,
)
from simdos.visualization.two_d import plot_evaluation_results


from ..base import process_cls
from .digitizers import DigitizerBase


class RF3ActorV2(DigitizerBase, g4.GateRF3ActorV2):  # type: ignore
    user_info_defaults = {
        "events_eval_size": (
            60_000,
            {
                "doc": "Number of total hits collected before evaluating the relative error.",
            },
        ),
        "rel_error_threshold": (
            0.050,
            {
                "doc": "Relative error threshold per voxel to stop the simulation.",
            },
        ),
        "rel_error_percentile": (
            0.950,
            {
                "doc": "Percentile of voxels to clear rel_error_threshold to stop the simulation.",
            },
        ),
        "max_energy": (
            125,
            {
                "doc": "Max energy produced by the source in keV.",
            },
        ),
        "num_bins": (
            25,
            {
                "doc": "Number of energy bins of the voxel histograms.",
            },
        ),
        "update_histograms_threshold": (
            10,
            {
                "doc": "Number of hits per voxel to update the histograms.",
            },
        ),
        "voxel_size": (
            5,
            {
                "doc": "Size of a single voxel side in mm. The voxel grid is cubic.",
            },
        ),
        "channel_name": (
            "voxel_world_actor",
            {
                "doc": "Name of the used rf3 channel.",
            },
        ),
        "output_filename": (
            "rf3_actor.rf3",
            {
                "doc": "Name of the output RF3 file.",
            },
        ),
        "tracer_type": (
            "Linetracing",
            {
                "doc": "Linetracing, Sampling, Bresenham or DDA.",
            },
        ),
    }

    # TODO: Die defaults sind fucky, es wird nichts zugewiesen
    def __init__(self, *args, **kwargs) -> None:
        DigitizerBase.__init__(self, *args, **kwargs)

        self.world_size = None
        self.world_limits = None

        self.eval_num_photons = []
        self.eval_eps_rel_cleared_percentage = []

        self.__initcpp__()

    def __initcpp__(self) -> None:
        g4.GateRF3ActorV2.__init__(self, self.user_info)  # type: ignore

    def initialize(self) -> None:
        # call the initialize() method from the super class (python-side)
        DigitizerBase.initialize(self)

        self.world_size = np.array(
            self.simulation.volume_manager.world_volume.size
        )  # Convert to m

        self.eval_num_photons = []
        self.eval_eps_rel_cleared_percentage = []

        # initialize C++ side
        self.user_info["output_path"] = self.simulation.output_dir
        self.user_info["world_size"] = self.world_size
        self.InitializeUserInfo(self.user_info)
        self.InitializeCpp()
        self.SetCallbackFunction(self.process_data)

    def process_data(self, actor) -> None:
        pass

    def __getstate__(self) -> dict:
        # needed to not pickle objects that cannot be pickled (g4, cuda, lock, etc).
        return_dict = super().__getstate__()

        # Standard values, other stuff needs too much RAM
        standard_entries = [
            "_simulation",
            "number_of_warnings",
            "_temporary_warning_cache",
            "user_info",
            "actor_engine",
            "user_output",
            "interfaces_to_user_output",
            "mother_attached_to",
        ]
        for key in list(return_dict.keys()):
            if key not in standard_entries:
                del return_dict[key]

        return return_dict

    def EndOfRunActionMasterThread(self, run_index) -> None:
        return 0  # type: ignore # Wenn kein 0, dann kackt alles ab

    def StartSimulationAction(self) -> None:
        DigitizerBase.StartSimulationAction(self)
        g4.GateRF3ActorV2.StartSimulationAction(self)  # type: ignore

    def EndSimulationAction(self) -> None:
        print(
            f"Generated Photons: {g4.GateRF3ActorV2.GetNumberOfAbsorbedEvents(self)}"  # type: ignore
        )
        # print(f"Registered Hits: {g4.GateRF3ActorV2.GetNumberOfHits(self)}")  # type: ignore
        # DigitizerBase.EndSimulationAction(self)

        plot_evaluation_results(
            self.eval_num_photons,
            self.eval_eps_rel_cleared_percentage,
            output_path=self.simulation.output_dir,
        )

        g4.GateRF3ActorV2.EndSimulationAction(self)  # type: ignore


class RF3Actor(DigitizerBase, g4.GateRF3Actor):  # type: ignore
    user_info_defaults = {
        "hits_batch_size": (
            5_000,
            {
                "doc": "Number of hits collected by a thread before processing them.",
            },
        ),
        "hits_eval_size": (
            60_000,
            {
                "doc": "Number of total hits collected before evaluating the relative error.",
            },
        ),
        "rel_error_threshold": (
            0.050,
            {
                "doc": "Relative error threshold per voxel to stop the simulation.",
            },
        ),
        "rel_error_percentile": (
            0.950,
            {
                "doc": "Percentile of voxels to clear rel_error_threshold to stop the simulation.",
            },
        ),
        "max_energy": (
            125,
            {
                "doc": "Max energy produced by the source in keV.",
            },
        ),
        "num_bins": (
            25,
            {
                "doc": "Number of energy bins of the voxel histograms.",
            },
        ),
        "update_histograms_threshold": (
            10,
            {
                "doc": "Number of hits per voxel to update the histograms.",
            },
        ),
        "voxel_size": (
            5,
            {
                "doc": "Size of a single voxel side in mm. The voxel grid is cubic.",
            },
        ),
        "channel_name": (
            "voxel_world_actor",
            {
                "doc": "Name of the used rf3 channel.",
            },
        ),
        "output_filename": (
            "rf3_actor.rf3",
            {
                "doc": "Name of the output RF3 file.",
            },
        ),
        # "trace_mode": (
        #     "round_safe",
        #     {
        #         "doc": "FIXME",
        #     },
        # ),
    }

    # TODO: Die defaults sind fucky, es wird nichts zugewiesen
    def __init__(self, *args, **kwargs) -> None:
        DigitizerBase.__init__(self, *args, **kwargs)

        self.world_size = None
        self.world_limits = None
        self.voxel_size = None

        self.crf = None

        self.energy_grid = None

        self.max_energy = None
        self.num_bins = None
        self.bin_width = None

        self.histogram_grid = None
        self.histogram_means = None
        self.histogram_hits_grid = None
        self.histogram_update_counts = None
        self.histogram_variances = None
        self.histogram_variances_means = None
        self.update_histogram_threshold = None

        self.hits_batch_size = None
        self.hits_eval_size = None
        self.num_callbacks_to_eval = None
        self.current_num_callbacks = None

        self.eps_rel = None
        self.rel_error_percentile = None
        self.rel_error_threshold = None

        self.eval_num_photons = []
        self.eval_eps_rel_cleared_percentage = []

        self.calc_lock = threading.Lock()
        self.eval_lock = threading.Lock()

        self.__initcpp__()

    def __initcpp__(self) -> None:
        g4.GateRF3Actor.__init__(self, self.user_info)  # type: ignore

    def initialize(self) -> None:
        # call the initialize() method from the super class (python-side)
        DigitizerBase.initialize(self)

        self.world_size = np.array(
            self.simulation.volume_manager.world_volume.size
        )  # Convert to m
        self.world_limits = np.stack(
            [-self.world_size // 2, self.world_size // 2], axis=1
        )  # type: ignore
        self.voxel_size = self.user_info["voxel_size"] / 1000  # Convert to m

        self.crf = CartesianRadiationField(
            vec3(
                self.world_size[0] / 1000,
                self.world_size[1] / 1000,
                self.world_size[2] / 1000,
            ),
            vec3(self.voxel_size, self.voxel_size, self.voxel_size),
        )
        channel = self.crf.add_channel(self.user_info["channel_name"])

        channel.add_layer("energies", "MeV", DType.FLOAT64)
        channel.add_layer("hits", "count", DType.UINT64)
        self.energy_grid = channel.get_layer_as_ndarray("energies")

        self.max_energy = self.user_info["max_energy"] / 1000  # Convert to MeV
        self.num_bins = self.user_info["num_bins"]
        self.bin_width = self.max_energy / self.num_bins

        channel.add_histogram_layer("histograms", self.num_bins, self.bin_width, "MeV")
        self.histogram_grid = channel.get_layer_as_ndarray("histograms")
        self.histogram_means = np.zeros_like(self.histogram_grid, dtype=np.float32)
        self.histogram_hits_grid = np.zeros_like(self.energy_grid, dtype=np.int64)
        self.histogram_update_counts = np.zeros_like(self.energy_grid, dtype=np.int64)
        self.histogram_variances = np.zeros_like(self.histogram_grid, dtype=np.float32)
        self.histogram_variances_means = np.zeros_like(
            self.histogram_grid, dtype=np.float64
        )
        self.update_histograms_threshold = self.user_info["update_histograms_threshold"]

        self.hits_batch_size = self.user_info["hits_batch_size"]
        self.hits_eval_size = self.user_info["hits_eval_size"]
        self.num_callbacks_to_eval = self.hits_eval_size // self.hits_batch_size
        self.current_num_callbacks = 0
        self.rel_error_percentile = self.user_info["rel_error_percentile"]
        self.rel_error_threshold = self.user_info["rel_error_threshold"]

        channel.add_layer("eps_rel", "percent", DType.FLOAT32)
        self.eps_rel = channel.get_layer_as_ndarray("eps_rel")
        self.eps_rel.fill(1)  # Maximum possible relative error

        self.eval_num_photons = []
        self.eval_eps_rel_cleared_percentage = []

        self.calc_lock = threading.Lock()
        self.eval_lock = threading.Lock()

        # initialize C++ side
        self.InitializeUserInfo(self.user_info)
        self.InitializeCpp()
        self.SetCallbackFunction(self.process_data)

        del channel

    def __getstate__(self) -> dict:
        # needed to not pickle objects that cannot be pickled (g4, cuda, lock, etc).
        return_dict = super().__getstate__()

        # Standard values, other stuff needs too much RAM
        test = [
            "_simulation",
            "number_of_warnings",
            "_temporary_warning_cache",
            "user_info",
            "actor_engine",
            "user_output",
            "interfaces_to_user_output",
            "mother_attached_to",
        ]
        for key in list(return_dict.keys()):
            if key not in test:
                del return_dict[key]

        return return_dict

    def process_data(self, actor) -> None:
        # get values from cpp side
        energies = actor.GetEnergy()  # MeV
        if energies.size == 0:
            return  # do nothing if no hits

        # Get values from sim
        pre_positions = actor.GetPrePosition()
        post_positions = actor.GetPostPosition()

        grid_indices, trajectory_ids = dda_batch_raycast(
            start_positions=pre_positions,
            end_positions=post_positions,
            voxel_size=self.voxel_size * 1000,  # mm to m # type: ignore
            world_limits=self.world_limits,
            grid_shape=np.array(self.energy_grid.shape),  # type: ignore
        )

        energies_flat = np.array(
            energies[trajectory_ids], dtype=np.float32
        )  # Shape: (num_hits,)
        bin_indices = np.clip(
            (energies_flat / self.bin_width),  # type: ignore
            0,
            self.num_bins - 1,  # type: ignore
        ).astype(np.uint16)

        with self.calc_lock:  # type: ignore
            update_grids_numba(
                np.ascontiguousarray(self.energy_grid),
                np.ascontiguousarray(self.histogram_hits_grid),
                np.ascontiguousarray(self.histogram_grid),
                np.ascontiguousarray(self.histogram_means),
                np.ascontiguousarray(self.histogram_update_counts),
                np.ascontiguousarray(energies_flat),
                np.ascontiguousarray(self.histogram_variances_means),
                np.ascontiguousarray(self.histogram_variances),
                np.ascontiguousarray(grid_indices),
                np.ascontiguousarray(bin_indices),
                update_histograms_threshold=self.update_histograms_threshold,  # type: ignore
            )

            self.current_num_callbacks += 1  # type: ignore

        with self.eval_lock:  # type: ignore
            if self.current_num_callbacks >= self.num_callbacks_to_eval:  # type: ignore
                print(
                    f"Evaluating with {g4.GateRF3Actor.GetNumberOfAbsorbedEvents(self)} Photons."  # type: ignore
                )
                self.current_num_callbacks = 0

                evaluate_relative_errors(
                    np.ascontiguousarray(self.histogram_variances),
                    np.ascontiguousarray(self.histogram_update_counts),
                    np.ascontiguousarray(self.eps_rel),
                )

                rel_error_percentile_idx = int(
                    self.eps_rel.size * self.rel_error_percentile  # type: ignore
                )
                sorted_eps_rel = np.sort(self.eps_rel, axis=None)  # type: ignore

                quantile_value = sorted_eps_rel[rel_error_percentile_idx]
                print(
                    f"Eps_rel {self.rel_error_percentile * 100}% Quantil {quantile_value}"  # type: ignore
                )
                print(
                    f"Percentage of voxels with eps_rel < {quantile_value}: {np.sum(self.eps_rel < quantile_value) / self.eps_rel.size * 100:.2f}%"  # type: ignore
                )

                self.eval_num_photons.append(
                    g4.GateRF3Actor.GetNumberOfAbsorbedEvents(self)  # type: ignore
                )
                self.eval_eps_rel_cleared_percentage.append(
                    np.sum(self.eps_rel < quantile_value) / self.eps_rel.size * 100  # type: ignore
                )

                if quantile_value <= self.rel_error_threshold:
                    g4.GateRF3Actor.StopSimulation(self)  # type: ignore
                    print("Threshold cleared, stopping simulation.")
                else:
                    print("Threshold not cleared, continuing simulation.")

        del (
            energies,
            pre_positions,
            post_positions,
            grid_indices,
            trajectory_ids,
            energies_flat,
            bin_indices,
        )

    def EndOfRunActionMasterThread(self, run_index) -> None:
        return 0  # type: ignore # Wenn kein 0, dann kackt alles ab

    def StartSimulationAction(self) -> None:
        DigitizerBase.StartSimulationAction(self)
        g4.GateRF3Actor.StartSimulationAction(self)  # type: ignore

    def EndSimulationAction(self) -> None:
        # with self.eval_lock:
        print(
            f"Generated Photons: {g4.GateRF3Actor.GetNumberOfAbsorbedEvents(self)}"  # type: ignore
        )
        print(f"Registered Hits: {g4.GateRF3Actor.GetNumberOfHits(self)}")  # type: ignore
        # DigitizerBase.EndSimulationAction(self)

        # plot_3d_heatmap(self.energy_grid, self.world_limits, (self.voxel_size, self.voxel_size, self.voxel_size))

        # indices_per_axis = np.linspace(25, 175, 3, dtype=int)
        # x_indices, y_indices, z_indices = np.meshgrid(indices_per_axis, indices_per_axis, indices_per_axis, indexing='ij')

        # indices_array = np.column_stack([x_indices.ravel(), y_indices.ravel(), z_indices.ravel()])
        # for idx in indices_array:
        #     plot_voxel_histograms_2d(self.histogram_grid, "tests/test_data/rf3actor_test_histograms", idx)

        plot_evaluation_results(
            self.eval_num_photons,
            self.eval_eps_rel_cleared_percentage,
            output_path=self.simulation.output_dir,
        )

        store_rf3_file(
            self.crf, self.simulation.output_dir / self.user_info["output_filename"]
        )  # type: ignore

        # TODO: hits grid mit histogram_grid berechnen
        del self.calc_lock  # Cannot be pickled
        del self.eval_lock  # Cannot be pickled
        del self.crf

        g4.GateRF3Actor.EndSimulationAction(self)  # type: ignore


process_cls(RF3Actor)
process_cls(RF3ActorV2)
