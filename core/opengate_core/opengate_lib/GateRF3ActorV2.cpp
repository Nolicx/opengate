/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   ------------------------------------ -------------- */

#include "GateRF3ActorV2.h"
#include "GateHelpersDict.h"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4ThreeVector.hh"
#include "G4VProcess.hh"
#include "GateSourceManager.h"
#include "G4EmProcessSubType.hh"
#include "G4Gamma.hh"

#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include "tqdm/tqdm.h"

// Hollow STL Files will be marked as room inside of them
// Object and Wall Voxels should not be needed to be checked for simulation stop
// Need to separate humans from objects?
// Need to separate enclosure from objects?

namespace {
VoxelRegion ClassifyPoint(G4Navigator *navigator, const G4ThreeVector &pos) {
  G4VPhysicalVolume *vol = navigator->LocateGlobalPointAndSetup(pos);
  if (!vol) {
    return VoxelRegion::WORLD;
  }
  const auto &name = vol->GetName();
  if (name == "world") {
    return VoxelRegion::WORLD;
  }
  if (name == "room" || name == "floor" || name == "ceiling" || name.rfind("wall_", 0) == 0) {
    return VoxelRegion::ENCLOSURE;
  }
  return VoxelRegion::OBJECT;
}
}  // namespace

GateRF3ActorV2::GateRF3ActorV2(py::dict &user_info): GateVActor(user_info, true) {
  // Register which actor hooks are used.
  fActions.insert("StartSimulationAction");
  fActions.insert("BeginOfRunAction");
  fActions.insert("BeginOfEventAction");
  // fActions.insert("PreUserTrackingAction");
  fActions.insert("SteppingAction");
  fActions.insert("EndOfRunAction");
  fActions.insert("EndOfEventAction");
  // fActions.insert("EndOfSimulationWorkerAction");
  fActions.insert("EndSimulationAction");
  fActions.insert("BeginOfRunActionMasterThread");
  fActions.insert("EndOfRunActionMasterThread");

  // Initialize atomic counters/flags.
  this->numberOfHits.store(0);
  this->numberOfAbsorbedEvents.store(0);
  this->evaluationFlag.store(false);
  this->runTerminationFlag.store(false);
}

GateRF3ActorV2::~GateRF3ActorV2() {
  // for debug
}

void GateRF3ActorV2::InitializeUserInfo(py::dict &user_info) {
  // Let GateVActor handle common configuration first.
  G4cout << "[RF3Actor] InitializeUserInfo()" << G4endl;
  GateVActor::InitializeUserInfo(user_info);

  // Read RF3-specific parameters from Python.
  this->worldSize = DictGetVecDouble(user_info, "world_size");
  this->eventsEvalSize = DictGetInt(user_info, "events_eval_size");
  this->relErrorThreshold = DictGetDouble(user_info, "rel_error_threshold");
  this->relErrorPercentile = DictGetDouble(user_info, "rel_error_percentile");
  this->maxEnergy = DictGetDouble(user_info, "max_energy") / 1000; //MeV
  this->numBins = DictGetInt(user_info, "num_bins");
  this->binWidth = this->maxEnergy / this->numBins;
  this->updateHistogramsThreshold = DictGetInt(user_info, "update_histograms_threshold");
  this->voxelSize = DictGetDouble(user_info, "voxel_size") / 1000; // Meter

  this->convergenceRegionMode = ConvergenceRegionMode::ALL;
  this->convergenceRegionModeString = "all";
  if (user_info.contains("convergence_region_mode")) {
    this->convergenceRegionModeString = DictGetStr(user_info, "convergence_region_mode");
    if (this->convergenceRegionModeString == "all") {
      this->convergenceRegionMode = ConvergenceRegionMode::ALL;
    } else if (this->convergenceRegionModeString == "no_enclosure") {
      this->convergenceRegionMode = ConvergenceRegionMode::NO_ENCLOSURE;
    } else if (this->convergenceRegionModeString == "no_objects") {
      this->convergenceRegionMode = ConvergenceRegionMode::NO_OBJECTS;
    } else if (this->convergenceRegionModeString == "no_enclosure_no_objects") {
      this->convergenceRegionMode = ConvergenceRegionMode::NO_ENCLOSURE_NO_OBJECTS;
    } else {
      G4cout << "[RF3Actor] Unknown convergence_region_mode '" << this->convergenceRegionModeString
             << "', defaulting to 'all'." << G4endl;
      this->convergenceRegionMode = ConvergenceRegionMode::ALL;
      this->convergenceRegionModeString = "all";
    }
  }

  G4cout << "[RF3Actor] Convergence region mode : " << this->convergenceRegionModeString << G4endl;

  this->generalChannelName = "general"; //DictGetStr(user_info, "channel_name");
  this->beamChannelName = "beam";
  this->roomChannelName = "room";
  this->objectChannelName = "object";

  this->outputPath = DictGetStr(user_info, "output_path");
  this->outputFileName = DictGetStr(user_info, "output_filename");
  this->tracerType = DictGetStr(user_info, "tracer_type");
}

void GateRF3ActorV2::InitializeCpp() {
  G4cout << "[RF3Actor] InitializeCpp()" << G4endl;
  // GateVActor::InitializeCpp(); // Not needed here; this actor owns its RF3 grid.

  // Reset counters for a fresh run.
  this->numberOfAbsorbedEvents.store(0);
  this->evaluationFlag.store(false);
  this->numberOfHits.store(0);

  // Create Cartesian RF3 field with world size and voxel size.
  this->crf = std::make_shared<RadFiled3D::CartesianRadiationField>(
    glm::vec3(this->worldSize[0] / 1000,
              this->worldSize[1] / 1000,
              this->worldSize[2] / 1000),
    glm::vec3(this->voxelSize));

  // --- Logging: Grid & Histogram Setup ---
  this->voxelCounts = this->crf->get_voxel_counts();
  this->voxelDims = this->crf->get_voxel_dimensions();
  this->numVoxels = this->voxelCounts.x * this->voxelCounts.y * this->voxelCounts.z;
  // One mutex per voxel for thread-safe accumulation.
  this->mutexes = std::make_shared<std::vector<std::mutex>>(this->numVoxels);

  // Precompute half field dimensions (mm) for coordinate transforms.
  this->halfFieldDims = glm::vec3(
						  static_cast<float>(this->voxelCounts.x * this->voxelDims.x * 1000) / 2.f,
              static_cast<float>(this->voxelCounts.y * this->voxelDims.y * 1000) / 2.f,
						  static_cast<float>(this->voxelCounts.z * this->voxelDims.z * 1000) / 2.f
					  );

  this->generalChannel = this->CreateEnergyChannel(this->generalChannelName);
  // Per-voxel histograms and variance tracking.
  this->generalChannel->add_layer<int>("update_counts", 0, "counts");     // histogram_update_grid
  this->generalChannel->add_custom_layer<RadFiled3D::HistogramVoxel>("histogram_variances_means", RadFiled3D::HistogramVoxel(this->numBins, this->binWidth, nullptr), 0.f, "variances_means");
  this->generalChannel->add_custom_layer<RadFiled3D::HistogramVoxel>("histogram_variances", RadFiled3D::HistogramVoxel(this->numBins, this->binWidth, nullptr), 0.f, "variances");
  this->generalChannel->add_layer<float>("eps_rel", 1.f, "percent");
  // Voxel region labels (WORLD / OBJECT / ENCLOSURE).
  this->generalChannel->add_layer<int>("voxel_region", static_cast<int>(VoxelRegion::WORLD), "label");
  
  // Select tracer implementation.
  if (this->tracerType == "Linetracing") {
  this->tracer = std::make_shared<RadFiled3D::LinetracingGridTracer>(*this->generalChannel);
  } else if (this->tracerType == "Sampling") {
    this->tracer = std::make_shared<RadFiled3D::SamplingGridTracer>(*this->generalChannel);
  } else if (this->tracerType == "Bresenham") {
    this->tracer = std::make_shared<RadFiled3D::BresenhamGridTracer>(*this->generalChannel);
  } else if (this->tracerType == "DDA") {
    this->tracer = std::make_shared<RadFiled3D::DDAGridTracer>(*this->generalChannel);
  }
  // FIXME 
  // if (this->tracerType != "DDA") {
  //   G4cout << "[RF3Actor] Forcing tracer to DDA for segment-length scoring." << G4endl;
  // }
  // this->tracer = std::make_shared<RadFiled3D::DDAGridTracer>(*this->generalChannel);

  this->beamChannel = this->CreateEnergyChannel(this->beamChannelName);
  this->roomChannel = this->CreateEnergyChannel(this->roomChannelName);
  this->objectChannel = this->CreateEnergyChannel(this->objectChannelName);
  // Track-length estimator for energy fluence is currently disabled.

  G4cout << "[RF3Actor] World size (m): " << worldSize[0] / 1000.0 << " x " << worldSize[1] / 1000.0 << " x " << worldSize[2] / 1000.0 << G4endl;
  G4cout << "[RF3Actor] Voxel counts : " << this->voxelCounts.x << " x " << this->voxelCounts.y << " x " << this->voxelCounts.z << " = " << this->voxelCounts.x * this->voxelCounts.y * this->voxelCounts.z << " voxels" << G4endl;
  G4cout << "[RF3Actor] Voxel size (m) : " << this->voxelDims.x << " x " << this->voxelDims.y << " x " << this->voxelDims.z << G4endl;
  G4cout << "[RF3Actor] Energy bins : " << numBins << ", bin width (MeV) = " << binWidth << G4endl;
  G4cout << "[RF3Actor] Tracer type : " << tracerType << G4endl;
  G4cout << "[RF3Actor] relError p-quantile : " << relErrorPercentile * 100.0 << " %" << G4endl;
  G4cout << "[RF3Actor] relError threshold : " << relErrorThreshold << " %" << G4endl;

  // Precompute voxel -> region mapping using Geant4 geometry.
  this->InitializeVoxelRegions();
}

std::shared_ptr<RadFiled3D::VoxelGridBuffer> GateRF3ActorV2::CreateEnergyChannel(const std::string& name) {
  crf->add_channel(name);
  auto channel = crf->get_channel(name);
  channel->add_layer<float>("energies", 0.f, "MeV");
  channel->add_layer<int>("hits", 0, "counts");
  channel->add_custom_layer<RadFiled3D::HistogramVoxel>(
      "histograms",
      RadFiled3D::HistogramVoxel(numBins, binWidth, nullptr),
      0.f, "MeV");
  // channel->add_layer<float>("energy_fluence", 0.f, "MeV*m");
  return channel;
}

void GateRF3ActorV2::InitializeVoxelRegions() {
  G4cout << "[RF3Actor] InitializeVoxelRegions()" << G4endl;
  
  auto *transportMgr = G4TransportationManager::GetTransportationManager();
  G4Navigator *navigator = transportMgr->GetNavigatorForTracking();

  size_t worldCount  = 0;
  size_t enclosureCount = 0;
  size_t objectCount = 0;

  G4cout << "[RF3Actor] Classifying voxel regions (x slices: "
         << this->voxelCounts.x << ", total voxels: " << this->numVoxels << ")"
         << G4endl;

  // For each voxel, sample center + corners; assign WORLD only if all samples are WORLD.
  for (int ix : tqdm::range(0, this->voxelCounts.x)) {
    for (int iy = 0; iy < this->voxelCounts.y; ++iy) {
      for (int iz = 0; iz < this->voxelCounts.z; ++iz) {
        const double x_mm = (ix + 0.5) * this->voxelDims.x * 1000.0 - this->halfFieldDims.x;
        const double y_mm = (iy + 0.5) * this->voxelDims.y * 1000.0 - this->halfFieldDims.y;
        const double z_mm = (iz + 0.5) * this->voxelDims.z * 1000.0 - this->halfFieldDims.z;
        const double dx = 0.5 * this->voxelDims.x * 1000.0;
        const double dy = 0.5 * this->voxelDims.y * 1000.0;
        const double dz = 0.5 * this->voxelDims.z * 1000.0;

        size_t worldHits = 0;
        size_t enclosureHits = 0;
        size_t objectHits = 0;

        const G4ThreeVector center(x_mm, y_mm, z_mm);
        VoxelRegion centerRegion = ClassifyPoint(navigator, center);
        if (centerRegion == VoxelRegion::WORLD) {
          ++worldHits;
        } else if (centerRegion == VoxelRegion::ENCLOSURE) {
          ++enclosureHits;
        } else {
          ++objectHits;
        }

        const double xs[2] = {x_mm - dx, x_mm + dx};
        const double ys[2] = {y_mm - dy, y_mm + dy};
        const double zs[2] = {z_mm - dz, z_mm + dz};

        for (double sx : xs) {
          for (double sy : ys) {
            for (double sz : zs) {
              const G4ThreeVector pos(sx, sy, sz);
              VoxelRegion sampleRegion = ClassifyPoint(navigator, pos);
              if (sampleRegion == VoxelRegion::WORLD) {
                ++worldHits;
              } else if (sampleRegion == VoxelRegion::ENCLOSURE) {
                ++enclosureHits;
              } else {
                ++objectHits;
              }
            }
          }
        }

        const size_t total_hits = worldHits + enclosureHits + objectHits;
        VoxelRegion region = VoxelRegion::OBJECT;
        if (total_hits > 0 && (worldHits == total_hits)) { // all samples WORLD
          region = VoxelRegion::WORLD;
          ++worldCount;
        } else if (enclosureHits > objectHits) {
          region = VoxelRegion::ENCLOSURE;
          ++enclosureCount;
        } else {
          ++objectCount;
        }

        auto &labelVoxel = this->generalChannel->get_voxel<RadFiled3D::ScalarVoxel<int>>("voxel_region", ix, iy, iz);
        labelVoxel = static_cast<int>(region);
      }
    }
  }

  G4cout << "[RF3Actor] WORLD voxels : " << worldCount  << G4endl;
  G4cout << "[RF3Actor] ENCLOSURE voxels : " << enclosureCount << G4endl;
  G4cout << "[RF3Actor] OBJECT voxels : " << objectCount << G4endl;
}

void GateRF3ActorV2::StartSimulationAction() {
}

void GateRF3ActorV2::BeginOfRunActionMasterThread(int run_id) {
  // Reset termination for this run.
  this->runTerminationFlag.store(false);
}

void GateRF3ActorV2::BeginOfRunAction(const G4Run *run) {
}

void GateRF3ActorV2::EndOfRunAction(const G4Run * /*run*/) {
}

int GateRF3ActorV2::EndOfRunActionMasterThread(int run_id) {
  return 0;
}

void GateRF3ActorV2::BeginOfEventAction(const G4Event *event) {
  // Reset track-level scattering flags for this event.
  auto &tls = fThreadLocalData.Get();
  tls.hasScattered.clear();

  // Count primary events and trigger evaluation periodically.
  size_t currentEvents = numberOfAbsorbedEvents.fetch_add(1) + 1;
  if (currentEvents % this->eventsEvalSize == 0) {
      evaluationFlag.store(true);
  }
}

// void GateRF3ActorV2::PreUserTrackingAction(const G4Track *track) {}

void GateRF3ActorV2::SteppingAction(G4Step *step) {
  G4Track *track = step->GetTrack();
  if ((track->GetKineticEnergy() <= 0.0)
      || (track->GetDefinition() != G4Gamma::Definition())) {
    return;
  }

  auto &tls = fThreadLocalData.Get();
  auto &hasScattered = tls.hasScattered;

  G4int trackID = track->GetTrackID();
  bool scattered = false;
  auto it = hasScattered.find(trackID);
  if (it != hasScattered.end()) {
    scattered = it->second;
  }

  // Mark as scattered when the photon leaves the primary (uncollided) beam.
  // Secondaries are never primary beam by definition.
  if (track->GetCurrentStepNumber() == 1) {
    if (track->GetParentID() > 0 || track->GetCreatorProcess() != nullptr) {
      scattered = true;
      hasScattered[trackID] = true;
    }
  }

  // Update scatter flag if this step is caused by a gamma EM interaction.
  auto *pre = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();
  auto *scatterProcess = post->GetProcessDefinedStep();
  if (scatterProcess) {
    if (scatterProcess->GetProcessType() == fElectromagnetic) {
      const auto st = scatterProcess->GetProcessSubType();
      // "Uncollided -> collided" for gammas: any EM interaction.
      if (st == fComptonScattering ||
          st == fRayleigh ||
          st == fPhotoElectricEffect ||
          st == fGammaConversion ||
          st == fGammaGeneralProcess) {
        scattered = true;
        hasScattered[trackID] = true;
      }
    }
  }

  // Map step segment into voxel indices using the chosen tracer.
  auto prePos = pre->GetPosition();
  auto postPos = post->GetPosition();
  auto kineticEnergy = pre->GetKineticEnergy();

  std::vector<size_t> voxelIndices = this->tracer->trace(
				(glm::vec3(prePos[0], prePos[1], prePos[2]) + this->halfFieldDims) / glm::vec3(1000),
	      (glm::vec3(postPos[0], postPos[1], postPos[2]) + this->halfFieldDims) / glm::vec3(1000)
      );
  // auto voxelHits = static_cast<RadFiled3D::DDAGridTracer*>(this->tracer.get())->trace_with_lengths(
  //   (glm::vec3(prePos[0], prePos[1], prePos[2]) + this->halfFieldDims) / glm::vec3(1000),
  //   (glm::vec3(postPos[0], postPos[1], postPos[2]) + this->halfFieldDims) / glm::vec3(1000)
  // );

  // Used for all voxels
  size_t binIndex = static_cast<size_t>(kineticEnergy / this->binWidth);
  if (binIndex >= this->numBins) {
    binIndex = this->numBins - 1;
  }

  for (const auto& voxelIndex : voxelIndices) {
    this->AccumulateVoxelHit(voxelIndex, kineticEnergy, scattered, binIndex);
  }
}

void GateRF3ActorV2::AccumulateVoxelHit(size_t voxelIndex, float kineticEnergy, bool scattered, size_t binIndex){
  // voxel_region is static after InitializeVoxelRegions(); safe to read without lock.
  auto &regionVoxel = this->generalChannel
                          ->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("voxel_region", voxelIndex)
                          .get_data();
  VoxelRegion region = static_cast<VoxelRegion>(regionVoxel);

  this->numberOfHits.fetch_add(1);

  std::lock_guard<std::mutex> lock((*this->mutexes)[voxelIndex]);
  auto& generalHits = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("hits", voxelIndex).get_data();
  auto& generalVoxelEnergy = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies", voxelIndex);
  auto& generalVoxelHist = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histograms", voxelIndex);      

  // Basic tallies.

  generalHits += 1;
  generalVoxelEnergy += kineticEnergy;

  auto* generalHistData = &generalVoxelHist.get_data();
  generalHistData[binIndex] += 1.f;

  // Classification logic:
  // - scattered == false: primary BEAM contribution
  // - scattered == true: ROOM or OBJECT depending on voxel region
  if (!scattered) {
    // Never scattered -> BEAM, independent of region.
    auto &energyBeam = this->beamChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies",voxelIndex).get_data();
    energyBeam += kineticEnergy;
    auto& beamHist = this->beamChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histograms", voxelIndex);      
    auto* beamHistData = &beamHist.get_data();
    beamHistData[binIndex] += 1.f;
    // auto &sumEL = this->beamChannel
    //                   ->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energy_fluence", voxelIndex)
    //                   .get_data();
    // sumEL += kineticEnergy * segmentLength;
  } else {
    // Already scattered -> assign to OBJECT or ROOM.
    if (region == VoxelRegion::OBJECT || region == VoxelRegion::ENCLOSURE) {
      auto &energyObject = this->objectChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies",voxelIndex).get_data();
      energyObject += kineticEnergy;
      auto& objectHist = this->objectChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histograms", voxelIndex);      
      auto* objectHistData = &objectHist.get_data();
      objectHistData[binIndex] += 1.f;
      // auto &sumEL = this->objectChannel
      //                   ->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energy_fluence", voxelIndex)
      //                   .get_data();
      // sumEL += kineticEnergy * segmentLength;
    } else {
      // Everything not marked OBJECT or ENCLOSURE is treated as ROOM (= WORLD).
      auto &energyRoom = this->roomChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies",voxelIndex).get_data();
      energyRoom += kineticEnergy;
      auto& roomHist = this->roomChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histograms", voxelIndex);      
      auto* roomHistData = &roomHist.get_data();
      roomHistData[binIndex] += 1.f;
      // auto &sumEL = this->roomChannel
      //                   ->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energy_fluence", voxelIndex)
      //                   .get_data();
      // sumEL += kineticEnergy * segmentLength;
    }
  }

  // Update running estimates of histogram variance every N generalHits.
  if (generalHits % this->updateHistogramsThreshold == 0)
  {
    auto& updateCounts = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", voxelIndex).get_data();
    auto& variancesVoxel = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histogram_variances", voxelIndex);
    auto& variancesMeansVoxel = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histogram_variances_means", voxelIndex);
    
    auto* variancesData = &variancesVoxel.get_data();
    auto* variancesMeansData = &variancesMeansVoxel.get_data();

    updateCounts += 1;

    // Welford-like online update per histogram bin.
    for (size_t j = 0; j < this->numBins; j++) {
      float generalHistMean = generalHistData[j] / generalHits;
      float deltaJ = generalHistMean - variancesMeansData[j];
      variancesMeansData[j] += deltaJ / updateCounts;

      float deltaJ2 = generalHistMean - variancesMeansData[j];
      variancesData[j] += deltaJ * deltaJ2; 
    }
  }

}

bool GateRF3ActorV2::ShouldIncludeRegion(VoxelRegion region) const {
  switch (this->convergenceRegionMode) {
    case ConvergenceRegionMode::ALL:
      return true;
    case ConvergenceRegionMode::NO_ENCLOSURE:
      return region != VoxelRegion::ENCLOSURE;
    case ConvergenceRegionMode::NO_OBJECTS:
      return region != VoxelRegion::OBJECT;
    case ConvergenceRegionMode::NO_ENCLOSURE_NO_OBJECTS:
      return (region != VoxelRegion::ENCLOSURE) && (region != VoxelRegion::OBJECT);
    default:
      return true;
  }
}

void GateRF3ActorV2::MaybeEvaluateAndStop() {
  // Only one thread should evaluate at a time.
  if (this->evalMutex.try_lock())
  {
    if (this->evaluationFlag.load())
    {
      // Reset flag so other events do not re-enter immediately.
      this->evaluationFlag.store(false);
      std::vector<float> errors;
      errors.reserve(this->numVoxels);

      // Snapshot per-voxel stats under lock; compute eps_rel without holding locks.
      std::vector<float> epsRelSnapshot(this->numVoxels, this->DEFAULT_ERROR_VALUE);
      std::vector<bool> includeVoxel(this->numVoxels, false);
      for (size_t i = 0; i < this->numVoxels; i++)
      {
        int updateCounts = 0;
        std::vector<float> variancesSnapshot;

        {
          // Short lock to copy the data needed for evaluation.
          std::lock_guard<std::mutex> lock((*this->mutexes)[i]);
          auto &regionVoxel = this->generalChannel
                                  ->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("voxel_region", i)
                                  .get_data();
          VoxelRegion region = static_cast<VoxelRegion>(regionVoxel);
          if (!this->ShouldIncludeRegion(region)) {
            continue;
          }
          includeVoxel[i] = true;

          updateCounts = this->generalChannel
                             ->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", i)
                             .get_data();
          if (updateCounts > this->MIN_UPDATE_COUNTS) {
            auto &variances = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histogram_variances", i);
            auto *variancesData = &variances.get_data();
            variancesSnapshot.assign(variancesData, variancesData + this->numBins);
          }
        }

        float epsRelValue = this->DEFAULT_ERROR_VALUE;
        if (updateCounts > this->MIN_UPDATE_COUNTS) {
          float sumM2OverCounts = 0.f;
          for (size_t j = 0; j < this->numBins; j++) {
            sumM2OverCounts += variancesSnapshot[j] / updateCounts;
          }
          epsRelValue = sumM2OverCounts * (this->VARIANCE_SCALING_FACTOR / this->numBins);
        }
        epsRelSnapshot[i] = epsRelValue;
        errors.push_back(epsRelValue);
      }

      if (errors.empty()) {
        G4cout << "[RF3Actor] No voxels selected for convergence check; skipping." << G4endl;
        this->evalMutex.unlock();
        return;
      }

      // Write eps_rel back to the grid with short per-voxel locks.
      for (size_t i = 0; i < this->numVoxels; i++) {
        if (!includeVoxel[i]) {
          continue;
        }
        std::lock_guard<std::mutex> lock((*this->mutexes)[i]);
        auto &epsRel = this->generalChannel->get_voxel_flat<float>("eps_rel", i);
        epsRel = epsRelSnapshot[i];
      }

      // Evaluate chosen percentile of epsRel.
      std::sort(errors.begin(), errors.end());
      size_t percentileIndex = static_cast<size_t>(errors.size() * this->relErrorPercentile);
      if (percentileIndex >= errors.size()) {
        percentileIndex = errors.size() - 1;
      }
      float quantileValue = errors[percentileIndex];

      size_t numBelow = 0;
      for (const auto& err : errors) {
        if (err < quantileValue) {
          ++numBelow;
        }
      }

      float cleared_percentage = (static_cast<float>(numBelow) / errors.size()) * 100.f;
      size_t currentAbsorbedEvents = numberOfAbsorbedEvents.load();

      // ----- Neue, verständliche Logs -----
      G4cout << "\n[RF3Actor] ===== Statistical convergence check =====" << G4endl;
      G4cout << "[RF3Actor] Primary photons simulated : " << currentAbsorbedEvents << " (events)" << G4endl;
      G4cout << "[RF3Actor] Number of voxels (in check) : " << errors.size() << G4endl;
      G4cout << "[RF3Actor] Target eps_rel quantile : " << (relErrorPercentile * 100.0f) << " %" << G4endl;
      G4cout << "[RF3Actor] Target eps_rel threshold : " << relErrorThreshold << " %" << G4endl;
      G4cout << "[RF3Actor] Measured eps_rel at quantile : " << quantileValue << G4endl;
      G4cout << "[RF3Actor] Voxels below threshold (eps_rel < " << relErrorThreshold << "): " << numBelow << " / " << errors.size() << " (" << cleared_percentage << " %)" << G4endl;

      bool stop = (quantileValue <= this->relErrorThreshold);
      if (stop) {
        G4cout << "[RF3Actor] Convergence reached: quantile(" << relErrorPercentile * 100.0f << "%) = " << quantileValue << " <= " << relErrorThreshold << "  -> requesting stop." << G4endl;
        this->StopSimulation();
      } else {
        G4cout << "[RF3Actor] Convergence NOT reached: quantile(" << relErrorPercentile * 100.0f << "%) = " << quantileValue << " > " << relErrorThreshold << "  -> continue simulation." << G4endl;
      }
    }
    this->evalMutex.unlock();
  } 
}

void GateRF3ActorV2::EndOfEventAction(const G4Event *event) {
  // If a stop was requested, ask the source manager to terminate the run.
   // Optional: periodically check if statistical criterion is met.
  if (this->evaluationFlag.load())
  {
    this->MaybeEvaluateAndStop();
  }
  if (runTerminationFlag.load()){
    fSourceManager->SetRunTerminationFlag(true);
  }
}

void GateRF3ActorV2::StopSimulation() {
  // Only set a flag here; actual termination is requested in EndOfEventAction.
  runTerminationFlag.store(true);
}

// void GateRF3ActorV2::EndOfSimulationWorkerAction(const G4Run * /*unused*/) {
// }

void GateRF3ActorV2::EndSimulationAction() {
  const std::string geometryInfo = "convergence_region_mode=" + this->convergenceRegionModeString +
                                   ", rel_error_threshold=" + std::to_string(this->relErrorThreshold) +
                                   ", rel_error_percentile=" + std::to_string(this->relErrorPercentile);

  // Minimal metadata for now; can be extended with real simulation info.
  std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
		RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
			static_cast<int>(this->numberOfAbsorbedEvents.load()),  // Number of primary particles
			geometryInfo, // Geometry Infos
			"", // Physics List
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
				glm::vec3(0.f, 0.f, 0.f), // Translation
				glm::vec3(0.f),           // Rotation
				0.f,                      // Voltage?
				""                        // Tube ID
			)
		),
		RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
			"AiDOS", // Software Name
			"", // Version
			"", // Repo
			""  // Commit
		)
	);

  // this->crf->remove_channel("general");
  this->generalChannel->remove_layer("energies");
  this->generalChannel->remove_layer("histograms");
  // this->generalChannel->remove_layer("voxel_region"); // Maybe useful later
  this->generalChannel->remove_layer("update_counts");
  this->generalChannel->remove_layer("eps_rel");
  this->generalChannel->remove_layer("histogram_variances");
  this->generalChannel->remove_layer("histogram_variances_means");
  // this->generalChannel->remove_layer("energy_fluence");
  
  this->generalChannel->remove_layer("hits");
  this->beamChannel->remove_layer("hits");
  this->roomChannel->remove_layer("hits");
  this->objectChannel->remove_layer("hits");

  // Persist the field to disk.
  RadFiled3D::Storage::FieldStore::store(this->crf,
    metadata,
    this->outputPath + "/" + this->outputFileName,
    RadFiled3D::Storage::StoreVersion::V1);
}
