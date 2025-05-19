from box import Box
import numpy as np
import itk
import threading

import opengate_core as g4
from ..utility import g4_units, LazyModuleLoader
from ..exception import fatal
from .base import ActorBase
from .actoroutput import ActorOutputSingleImage, ActorOutputRoot
from ..base import process_cls

class RF3Actor(ActorBase, g4.GateRF3Actor):

    user_info_defaults = {
        "batch_size": (
            50.000,
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
        
        self.output = self.user_info.output if self.user_info.output else {"total_hits": 0, "total_energy": 0}

        self._total_hits = 0
        self._total_energy = 0
        self.lock = None

    def __initcpp__(self)-> None:
        g4.GateRF3Actor.__init__(self, self.user_info)
        self.AddActions(
            {
                "BeginOfRunActionMasterThread",
                "EndOfRunActionMasterThread",
                "BeginOfRunAction",
                "EndOfRunAction",
                "SteppingAction",
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
        num_hits = np.array(actor.GetCurrentNumberOfHits())
        if self.output is not None:
            self.output["total_hits"] += num_hits
            self.output["total_energy"] += np.sum(energy)
        
        # do nothing if no hits
        if energy.size == 0:
            return

    def EndOfRunActionMasterThread(self, run_index):
        return 

    def EndSimulationAction(self):
        if self.output:
            print("Total hits: ", self.output["total_hits"])
            print("Total energy: ", self.output["total_energy"])
        
        g4.GateRF3Actor.EndSimulationAction(self)
        ActorBase.EndSimulationAction(self)
        del self.lock   #Cannot be pickled

process_cls(RF3Actor)