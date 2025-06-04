/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateRF3Actor.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyGateRF3Actor : public GateRF3Actor {
public:
    using GateRF3Actor::GateRF3Actor;

//   void BeginOfRunActionMasterThread(int run_id) override {
//     PYBIND11_OVERLOAD(void, GateRF3Actor, BeginOfRunActionMasterThread, run_id);
//   }

//   int EndOfRunActionMasterThread(int run_id) override {
//     PYBIND11_OVERLOAD(int, GateRF3Actor, EndOfRunActionMasterThread, run_id);
//   }

};

void init_GateRF3Actor(py::module &m) {
    py::class_<GateRF3Actor, PyGateRF3Actor,
                std::unique_ptr<GateRF3Actor, py::nodelete>, GateVActor>(
        m, "GateRF3Actor")
    // py::class_<GateRF3Actor, GateVActor>(m, "GateRF3Actor")
        .def(py::init<py::dict &>(), py::keep_alive<1, 2>())
        .def("InitializeUserInfo", &GateRF3Actor::InitializeUserInfo)
        .def("BeginOfRunAction", &GateRF3Actor::BeginOfRunAction)
        .def("EndOfRunAction", &GateRF3Actor::EndOfRunAction)
        .def("SetCallbackFunction", &GateRF3Actor::SetCallbackFunction)
        // .def("GetCurrentNumberOfHits", &GateRF3Actor::GetCurrentNumberOfHits)
        // .def("GetCurrentRunId", &GateRF3Actor::GetCurrentRunId)
        .def("GetEnergy", &GateRF3Actor::GetEnergy)
        .def("GetPrePositionX", &GateRF3Actor::GetPrePositionX)
        .def("GetPrePositionY", &GateRF3Actor::GetPrePositionY)
        .def("GetPrePositionZ", &GateRF3Actor::GetPrePositionZ)
        .def("GetPostPositionX", &GateRF3Actor::GetPostPositionX)
        .def("GetPostPositionY", &GateRF3Actor::GetPostPositionY)
        .def("GetPostPositionZ", &GateRF3Actor::GetPostPositionZ);
        // .def("GetDirectionX", &GateRF3Actor::GetDirectionX)
        // .def("GetDirectionY", &GateRF3Actor::GetDirectionY)
        // .def("GetDirectionZ", &GateRF3Actor::GetDirectionZ)
}
