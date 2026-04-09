import matplotlib
matplotlib.use("Agg")

import opengate_core as g4

from ..base import process_cls
from .base import ActorBase
from .digitizers import DigitizerBase


class AIDosActor(ActorBase, g4.GateAIDosActor):  # type: ignore
    user_info_defaults = {
        "world_size":                  ([1000, 1000, 1000], {"doc": "mm"}),
        "voxel_size":                  (5.0,   {"doc": "mm"}),
        "max_energy":                  (125.0, {"doc": "keV"}),
        "num_bins":                    (25,    {"doc": "Energy histogram bins"}),
        "update_histograms_threshold": (10,    {"doc": "Hits per voxel before variance update"}),
        "events_eval_size":            (60000, {"doc": "Events between convergence checks"}),
        "rel_error_threshold":         (0.05,  {"doc": "Convergence threshold"}),
        "rel_error_percentile":        (0.95,  {"doc": "Percentile of voxels to satisfy threshold"}),
        "convergence_region_mode":     ("all", {"doc": "all | no_enclosure | no_objects | no_enclosure_no_objects"}),
        "tracer_type":                 ("DDA", {"doc": "DDA only"}),
        "scoring_quantities":          (["transported_energy", "ambient_dose"],
                                        {"doc": "transported_energy, energy_fluence, air_kerma, ambient_dose"}),
        "adeph_path":                  ("",    {"doc": "Path to adeph.dat (ambient dose conversion factors)"}),
        "mu_en_air_path":              ("",    {"doc": "Path to mu_en_air.dat (NIST air mass-energy coefficients)"}),
        "channel_name_general":        ("general",  {"doc": "RF3 channel name"}),
        "channel_name_beam":           ("beam",     {"doc": "RF3 channel name"}),
        "channel_name_room":           ("room",     {"doc": "RF3 channel name"}),
        "channel_name_object":         ("object",   {"doc": "RF3 channel name"}),
        "output_path":                 (".",        {"doc": "Output directory (overridden by simulation.output_dir)"}),
        "output_filename":             ("radiation_field.rf3", {"doc": "Output filename"}),
    }

    def __init__(self, *args, **kwargs) -> None:
        ActorBase.__init__(self, *args, **kwargs)
        self.__initcpp__()

    def __initcpp__(self) -> None:
        g4.GateAIDosActor.__init__(self, self.user_info)

    def initialize(self) -> None:
        ActorBase.initialize(self)
        self.user_info["output_path"] = str(self.simulation.output_dir)
        self.user_info["world_size"] = list(self.simulation.volume_manager.world_volume.size)
        self.InitializeUserInfo(self.user_info)
        self.InitializeCpp()

    def StartSimulationAction(self) -> None:
        g4.GateAIDosActor.StartSimulationAction(self)

    def EndSimulationAction(self) -> None:
        print(f"[AIDosActor] Simulated primaries: {g4.GateAIDosActor.GetNumberOfAbsorbedEvents(self)}")
        g4.GateAIDosActor.EndSimulationAction(self)

    def EndOfRunActionMasterThread(self, run_index) -> None:
        return 0  # type: ignore

    def __getstate__(self):
        return_dict = super().__getstate__()
        standard_entries = [
            "_simulation", "number_of_warnings", "_temporary_warning_cache",
            "user_info", "actor_engine", "user_output",
            "interfaces_to_user_output", "mother_attached_to",
        ]
        for key in list(return_dict.keys()):
            if key not in standard_entries:
                del return_dict[key]
        return return_dict


process_cls(AIDosActor)
