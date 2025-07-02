/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateRF3ActorV2.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_GateRF3ActorV2(py::module &m) {
    py::class_<GateRF3ActorV2, GateVActor>(m, "GateRF3ActorV2")
        .def(py::init<py::dict &>(), py::keep_alive<1, 2>())
        .def("InitializeUserInfo", &GateRF3ActorV2::InitializeUserInfo)
        .def("StartSimulationAction", &GateRF3ActorV2::StartSimulationAction)
        .def("BeginOfRunAction", &GateRF3ActorV2::BeginOfRunAction)
        .def("EndSimulationAction", &GateRF3ActorV2::EndSimulationAction)
        .def("EndOfRunAction", &GateRF3ActorV2::EndOfRunAction)
        
        .def("SetCallbackFunction", &GateRF3ActorV2::SetCallbackFunction)
        .def("StopSimulation", &GateRF3ActorV2::StopSimulation)
        .def("GetNumberOfAbsorbedEvents", &GateRF3ActorV2::GetNumberOfAbsorbedEvents);
}
