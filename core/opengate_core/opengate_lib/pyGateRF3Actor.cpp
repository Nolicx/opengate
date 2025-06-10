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
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_GateRF3Actor(py::module &m) {
    // py::class_<GateRF3Actor, PyGateRF3Actor,
    //             std::unique_ptr<GateRF3Actor, py::nodelete>, GateVActor>(
    //     m, "GateRF3Actor")
    py::class_<GateRF3Actor, GateVActor>(m, "GateRF3Actor")
        .def(py::init<py::dict &>(), py::keep_alive<1, 2>())
        .def("InitializeUserInfo", &GateRF3Actor::InitializeUserInfo)
        .def("BeginOfRunAction", &GateRF3Actor::BeginOfRunAction)
        .def("EndOfRunAction", &GateRF3Actor::EndOfRunAction)
        .def("SetCallbackFunction", &GateRF3Actor::SetCallbackFunction)
        // .def("GetCurrentNumberOfHits", &GateRF3Actor::GetCurrentNumberOfHits)
        // .def("GetCurrentRunId", &GateRF3Actor::GetCurrentRunId)
        .def("GetEnergy", [](GateRF3Actor &self) { return py::array(py::cast(self.GetEnergy())); })
        .def("GetPrePositionX", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPrePositionX())); })
        .def("GetPrePositionY", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPrePositionY())); })
        .def("GetPrePositionZ", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPrePositionZ())); })
        .def("GetPrePosition", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPrePosition())); })
        .def("GetPostPositionX", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPostPositionX())); })
        .def("GetPostPositionY", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPostPositionY())); })
        .def("GetPostPositionZ", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPostPositionZ())); })
        .def("GetPostPosition", [](GateRF3Actor &self) { return py::array(py::cast(self.GetPostPosition())); })
        .def("StopSimulation", &GateRF3Actor::StopSimulation)
        .def("GetNumberOfAbsorbedEvents", &GateRF3Actor::GetNumberOfAbsorbedEvents)
        .def("GetNumberOfHits", &GateRF3Actor::GetNumberOfHits);
        // .def("GetDirectionX", &GateRF3Actor::GetDirectionX)
        // .def("GetDirectionY", &GateRF3Actor::GetDirectionY)
        // .def("GetDirectionZ", &GateRF3Actor::GetDirectionZ)
}


// class PyGateRF3Actor : public GateRF3Actor {
// public:
//     using GateRF3Actor::GateRF3Actor;

// //   void BeginOfRunActionMasterThread(int run_id) override {
// //     PYBIND11_OVERLOAD(void, GateRF3Actor, BeginOfRunActionMasterThread, run_id);
// //   }

// //   int EndOfRunActionMasterThread(int run_id) override {
// //     PYBIND11_OVERLOAD(int, GateRF3Actor, EndOfRunActionMasterThread, run_id);
// //   }
// };