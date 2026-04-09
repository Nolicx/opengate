/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateAIDosActor_h
#define GateAIDosActor_h

#include "G4Cache.hh"
#include "GateVActor.h"

#include <pybind11/stl.h>
#include <glm/glm.hpp>

#include <mutex>
#include <atomic>
#include <string>
#include <unordered_set>
#include <vector>
#include <set>

#include <RadFiled3D/RadiationField.hpp>
#include <RadFiled3D/GridTracer.hpp>

namespace py = pybind11;

/// Static region label used to classify voxel energy deposition.
enum class VoxelRegion : int {
  WORLD  = 0, // TODO: rename to ROOM for clarity?
  OBJECT = 1,
  ENCLOSURE = 2,
};

enum class ConvergenceRegionMode : int {
  ALL = 0,
  NO_ENCLOSURE = 1,
  NO_OBJECTS = 2,
  NO_ENCLOSURE_NO_OBJECTS = 3,
};

struct LookupTable {
    std::vector<double> energiesMeV;
    std::vector<double> values;
    double Interpolate(double eMeV) const;
};

/// Actor that scores configurable radiation field quantities into a RadFiled3D voxel grid
/// and performs adaptive stopping based on statistical convergence.
class GateAIDosActor : public GateVActor {

public:
    explicit GateAIDosActor(py::dict &user_info);
    ~GateAIDosActor() override;

    // Initialization
    void InitializeUserInfo(py::dict &user_info) override;
    void InitializeCpp() override;

    // Simulation hooks
    void StartSimulationAction() override;
    void BeginOfRunAction(const G4Run *run) override;
    void BeginOfEventAction(const G4Event *event) override;
    void SteppingAction(G4Step *step) override;
    void EndOfEventAction(const G4Event *event) override;
    void EndOfRunAction(const G4Run *run) override;
    void EndSimulationAction() override;

    // Master thread only
    void BeginOfRunActionMasterThread(int run_id) override;
    int EndOfRunActionMasterThread(int run_id) override;

    // Control & status
    void StopSimulation();
    size_t GetNumberOfAbsorbedEvents() const { return numberOfAbsorbedEvents.load(); } //needed?
    size_t GetNumberOfHits() const { return numberOfHits.load(); }//needed?
    bool IsRunTerminated() const { return runTerminationFlag.load(); }

private:
    // --- Initialization helpers ---
    void InitializeVoxelRegions();
    void LoadLookupTable(const std::string& path, LookupTable& table, double energyScaleFactor = 1e3);
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> CreateEnergyChannel(const std::string& name);
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> CreateGeneralChannel();

    // --- Stepping & accumulation ---
    void AccumulateVoxelHits(size_t voxelIndex, float segmentLengthCm, float energyMeV,
                             bool scattered, size_t binIndex);
    void FinalizeQuantities(RadFiled3D::VoxelGridBuffer* ch);
    void FinalizeGeneralChannel();

    // --- Convergence check ---
    void MaybeEvaluateAndStop();
    bool ShouldIncludeRegion(VoxelRegion region) const;

    static constexpr float VARIANCE_SCALING_FACTOR = 4.0f;
    static constexpr int   MIN_UPDATE_COUNTS       = 2;
    static constexpr float DEFAULT_ERROR_VALUE     = 1.0f;

    // --- RadFiled3D ---
    std::shared_ptr<RadFiled3D::CartesianRadiationField> crf;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> generalChannel;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> beamChannel;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> roomChannel;
    std::shared_ptr<RadFiled3D::VoxelGridBuffer> objectChannel;
    std::shared_ptr<RadFiled3D::GridTracer>      tracer;

    // --- Synchronization ---
    mutable std::mutex                      evalMutex;
    std::shared_ptr<std::vector<std::mutex>> mutexes;

    // --- Configuration (from Python) ---
    std::vector<double> worldSize;
    int    eventsEvalSize;
    float  relErrorThreshold;
    float  relErrorPercentile;
    int    numBins;
    int    updateHistogramsThreshold;
    std::string generalChannelName;
    std::string beamChannelName;
    std::string roomChannelName;
    std::string objectChannelName;
    std::string outputPath;
    std::string outputFileName;
    std::string tracerType;
    ConvergenceRegionMode convergenceRegionMode = ConvergenceRegionMode::ALL;
    std::string           convergenceRegionModeString;
    std::set<std::string> scoringQuantities;
    float       voxelSize;
    std::string adephPath;
    std::string muEnAirPath;

    // --- Lookup tables ---
    LookupTable adephTable;
    LookupTable muEnAirTable;

    // --- Derived quantities (set in InitializeCpp) ---
    glm::ivec3 voxelCounts;
    glm::vec3  voxelDims;
    glm::vec3  halfFieldDims;
    size_t     numVoxels;
    float      voxelVolumeCm3;
    float      binWidth;
    bool       needsStepLengths = false;

    // --- Runtime counters ---
    std::atomic<size_t> numberOfAbsorbedEvents{0};
    std::atomic<size_t> numberOfHits{0};
    std::atomic<bool>   evaluationFlag{false};
    std::atomic<bool>   runTerminationFlag{false};

    struct threadLocalT {
        std::unordered_set<G4int> hasScattered;
        std::vector<RadFiled3D::VoxelHit> voxelHits;
    };
    G4Cache<threadLocalT> fThreadLocalData;
};

#endif // AIDosActor_h
