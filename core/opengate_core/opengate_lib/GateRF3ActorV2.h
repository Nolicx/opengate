/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef RF3ActorV2_h
#define RF3ActorV2_h

#include "GateHelpers.h"
#include "GateVActor.h"
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <shared_mutex>
#include <string>

#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include <RadFiled3D/RadiationField.hpp>
#include <RadFiled3D/GridTracer.hpp>

namespace py = pybind11;

class GateRF3ActorV2 : public GateVActor {

public:
    // Callback function
    using CallbackFunctionType = std::function<void(GateRF3ActorV2 *)>;

    explicit GateRF3ActorV2(py::dict &user_info);

    ~GateRF3ActorV2() override;
    
    void InitializeUserInfo(py::dict &user_info) override;
    void InitializeCpp() override;
    void StartSimulationAction() override;
    void BeginOfEventAction(const G4Event *event) override;  
    void BeginOfRunAction(const G4Run * /*run*/) override;  
    // void PreUserTrackingAction(const G4Track *track) override;
    void SteppingAction(G4Step *) override;     //Called when step in attached volume
    void EndOfEventAction(const G4Event *event) override;
    void EndOfRunAction(const G4Run *run) override;
    // void EndOfSimulationWorkerAction(const G4Run *run) override;
    void EndSimulationAction() override;
    void SetCallbackFunction(CallbackFunctionType &f);  // Set the user "apply" function (python)
    // int GetCurrentNumberOfHits() const;
    // int GetCurrentRunId() const;

    void BeginOfRunActionMasterThread(int run_id) override;  // Called at simulation start (master thread only)
    int EndOfRunActionMasterThread(int run_id) override;  // Called at simulation end (master thread only)

    int GetNumberOfAbsorbedEvents() const { return fNumberOfAbsorbedEvents; }
    int GetNumberOfHits() const { return fNumberOfHits; }
    bool runTerminationFlag;
    void StopSimulation();

    std::shared_ptr<RadFiled3D::CartesianRadiationField> crf;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> channel;
    std::shared_ptr<RadFiled3D::GridTracer> tracer;
    std::shared_ptr<std::vector<std::shared_mutex>> mutexes;
    glm::vec3 half_field_dim;

protected:
    CallbackFunctionType fCallbackFunction;

    int fNumberOfHits;
    int fNumberOfAbsorbedEvents;
    bool evaluationFlag;
    // bool runTerminationFlag;

    std::vector<double> worldSize;
    int eventsEvalSize;
    float relErrorThreshold;
    float relErrorPercentile;
    float maxEnergy;
    int numBins;
    float binWidth;
    int updateHistogramsThreshold;
    float voxelSize;
    std::string channelName;
    std::string outputPath;
    std::string outputFileName;
    std::string tracerType;

    int sNumThreads;
};

#endif // RadFiled3DActor_h