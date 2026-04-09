/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateAIDosActor.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_GateAIDosActor(py::module &m) {
    py::class_<GateAIDosActor, GateVActor>(m, "GateAIDosActor")
        .def(py::init<py::dict &>(), py::keep_alive<1, 2>())
        .def("InitializeUserInfo", &GateAIDosActor::InitializeUserInfo)
        .def("StartSimulationAction", &GateAIDosActor::StartSimulationAction)
        .def("BeginOfRunAction", &GateAIDosActor::BeginOfRunAction)
        .def("EndSimulationAction", &GateAIDosActor::EndSimulationAction)
        .def("EndOfRunAction", &GateAIDosActor::EndOfRunAction)

        // .def("SetCallbackFunction", &GateAIDosActor::SetCallbackFunction)
        .def("StopSimulation", &GateAIDosActor::StopSimulation)
        .def("GetNumberOfAbsorbedEvents", &GateAIDosActor::GetNumberOfAbsorbedEvents);
}
