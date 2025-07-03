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
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <string>
#include <functional>

#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include <RadFiled3D/RadiationField.hpp>
#include <RadFiled3D/GridTracer.hpp>

namespace py = pybind11;

class GateRF3ActorV2 : public GateVActor {

public:
    // Callback function type
    using CallbackFunctionType = std::function<void(GateRF3ActorV2 *)>;

    explicit GateRF3ActorV2(py::dict &user_info);
    ~GateRF3ActorV2() override;
    
    // Initialization methods
    void InitializeUserInfo(py::dict &user_info) override;
    void InitializeCpp() override;

    // Simulation action methods
    void StartSimulationAction() override;
    void BeginOfEventAction(const G4Event *event) override;  
    void BeginOfRunAction(const G4Run * /*run*/) override;  
    // void PreUserTrackingAction(const G4Track *track) override;
    void SteppingAction(G4Step *) override;     //Called when step in attached volume
    void EndOfEventAction(const G4Event *event) override;
    void EndOfRunAction(const G4Run *run) override;
    // void EndOfSimulationWorkerAction(const G4Run *run) override;
    void EndSimulationAction() override;
    // int GetCurrentNumberOfHits() const;
    // int GetCurrentRunId() const;

    // Master thread methods
    void BeginOfRunActionMasterThread(int run_id) override;  // Called at simulation start (master thread only)
    int EndOfRunActionMasterThread(int run_id) override;  // Called at simulation end (master thread only)

    // Callback and control methods
    void SetCallbackFunction(CallbackFunctionType &f);  // Set the user "apply" function (python)
    void StopSimulation();

    // Thread-safe getter methods
    size_t GetNumberOfAbsorbedEvents() const { return numberOfAbsorbedEvents.load(); }
    size_t GetNumberOfHits() const { return numberOfHits.load(); }
    bool IsRunTerminated() const { return runTerminationFlag.load(); }

private:
    // Constants
    static constexpr float VARIANCE_SCALING_FACTOR = 4.0f;
    static constexpr int MIN_UPDATE_COUNTS = 2;
    static constexpr float DEFAULT_ERROR_VALUE = 1.0f;

    // Public data members (for compatibility)
    std::shared_ptr<RadFiled3D::CartesianRadiationField> crf;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> channel;
    std::shared_ptr<RadFiled3D::GridTracer> tracer;
    glm::vec3 half_field_dim;

    // Thread synchronisation
    mutable std::mutex evalMutex;
    std::shared_ptr<std::vector<std::shared_mutex>> mutexes;
    int sNumThreads;

    // Configuration parameters
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

    // Callback function
    CallbackFunctionType fCallbackFunction;

    // int fNumberOfHits;
    // int fNumberOfAbsorbedEvents;
    // bool evaluationFlag;
    // bool runTerminationFlag;
    // Thread-safe counters and flags
    std::atomic<size_t> numberOfAbsorbedEvents{0};
    std::atomic<size_t> numberOfHits{0};
    std::atomic<bool> evaluationFlag{false};
    std::atomic<bool> runTerminationFlag{false};
};

#endif // RadFiled3DActor_h