/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef RF3ActorV2_h
#define RF3ActorV2_h

#include "G4Cache.hh"
#include "GateVActor.h"

#include <pybind11/stl.h>
#include <glm/glm.hpp>

// #include <shared_mutex>
#include <mutex>
#include <atomic>
#include <string>
#include <functional>
#include <unordered_map>

#include <RadFiled3D/RadiationField.hpp>
#include <RadFiled3D/GridTracer.hpp>

namespace py = pybind11;

/// Static region label used to classify voxel energy deposition.
enum class VoxelRegion : int {
  WORLD  = 0,
  OBJECT = 1,
  ENCLOSURE = 2,
  // CARM   = 2,
  // später gern erweitern: PATIENT, TABLE, SHIELD, ...
};

enum class ConvergenceRegionMode : int {
  ALL = 0,
  NO_ENCLOSURE = 1,
  NO_OBJECTS = 2,
  NO_ENCLOSURE_NO_OBJECTS = 3,
};

/// Actor that writes energy, spectra and statistical uncertainty into a RadFiled3D voxel grid and performs adaptive stopping.
class GateRF3ActorV2 : public GateVActor {

public:
    explicit GateRF3ActorV2(py::dict &user_info);
    ~GateRF3ActorV2() override;
    
    // Initialization
    void InitializeUserInfo(py::dict &user_info) override;
    void InitializeCpp() override;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> CreateEnergyChannel(const std::string& channel_name);

    // Simulation hooks
    void StartSimulationAction() override;
    void BeginOfEventAction(const G4Event *event) override;  
    void BeginOfRunAction(const G4Run * /*run*/) override;  
    void SteppingAction(G4Step *) override;     //Called when step in attached volume
    void EndOfEventAction(const G4Event *event) override;
    void EndOfRunAction(const G4Run *run) override;
    void EndSimulationAction() override;

    // Master thread only 
    void BeginOfRunActionMasterThread(int run_id) override;  // Called at simulation start (master thread only)
    int EndOfRunActionMasterThread(int run_id) override;  // Called at simulation end (master thread only)

    // Control simulation termination
    void StopSimulation();

    // Information Getters
    size_t GetNumberOfAbsorbedEvents() const { return numberOfAbsorbedEvents.load(); }
    size_t GetNumberOfHits() const { return numberOfHits.load(); }
    bool IsRunTerminated() const { return runTerminationFlag.load(); }

  private:
    /// Assign WORLD/OBJECT labels to each voxel via G4 navigator.
    void InitializeVoxelRegions();

    /// Accumulate a single voxel hit: energy, histogram bin, and BEAM/ROOM/OBJECT category based on `scattered`.
    void AccumulateVoxelHit(size_t voxelIndex, float energy, bool scattered, size_t binIndex);
    
    /// Periodic check of statistical error to decide early stopping.
    void MaybeEvaluateAndStop();
    bool ShouldIncludeRegion(VoxelRegion region) const;

    // Standard values constants controlling the uncertainty estimator.
    static constexpr float VARIANCE_SCALING_FACTOR = 4.0f;
    static constexpr int MIN_UPDATE_COUNTS = 2;
    static constexpr float DEFAULT_ERROR_VALUE = 1.0f;

    // RadFiled3D field and access
    std::shared_ptr<RadFiled3D::CartesianRadiationField> crf;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> generalChannel;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> beamChannel;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> roomChannel;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> objectChannel;
    std::shared_ptr<RadFiled3D::GridTracer> tracer;

    // Synchronization: per-voxel locks + evaluation lock
    mutable std::mutex evalMutex;
    std::shared_ptr<std::vector<std::mutex>> mutexes;

    // Configuration parameters (from Python)
    std::vector<double> worldSize;
    int eventsEvalSize;
    float relErrorThreshold;
    float relErrorPercentile;
    float maxEnergy;
    int numBins;
    float binWidth;
    int updateHistogramsThreshold;
    float voxelSize;
    std::string generalChannelName;
    std::string beamChannelName;
    std::string roomChannelName;
    std::string objectChannelName;

    ConvergenceRegionMode convergenceRegionMode = ConvergenceRegionMode::ALL;
    std::string convergenceRegionModeString;

    std::string outputPath;
    std::string outputFileName;
    std::string tracerType;

    glm::ivec3 voxelCounts;
    glm::vec3 voxelDims;
    glm::vec3 halfFieldDims;
    size_t numVoxels;

    // Thread-safe counters
    std::atomic<size_t> numberOfAbsorbedEvents{0};
    std::atomic<size_t> numberOfHits{0};
    std::atomic<bool> evaluationFlag{false};
    std::atomic<bool> runTerminationFlag{false};

    /// Thread-local state: trackID -> has scattered at least once.
    struct threadLocalT {
      std::unordered_map<G4int, bool> hasScattered;
    };
    G4Cache<threadLocalT> fThreadLocalData;
};

#endif // RF3ActorV2_h
