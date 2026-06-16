/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateAIDosActor.h"
#include "GateHelpersDict.h"
#include "GateSourceManager.h"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4ThreeVector.hh"
#include "G4VProcess.hh"
#include "G4EmProcessSubType.hh"
#include "G4Gamma.hh"

#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include "tqdm/tqdm.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

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

// ---------------------------------------------------------------------------
// LookupTable
// ---------------------------------------------------------------------------

double LookupTable::Interpolate(double eMeV) const {
    if (this->energiesMeV.empty()) return 0.0;
    if (eMeV <= energiesMeV.front()) return values.front();
    if (eMeV >= energiesMeV.back()) return values.back();

    auto it = std::lower_bound(energiesMeV.begin(), energiesMeV.end(), eMeV);
    size_t i = std::distance(energiesMeV.begin(), it);

    // log-log interpolation
    double e0 = energiesMeV[i-1], e1 = energiesMeV[i];
    double v0 = values[i-1], v1 = values[i];
    double t = std::log(eMeV / e0) / std::log(e1 / e0);
    return std::exp(std::log(v0) + t * std::log(v1 / v0));
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

GateAIDosActor::GateAIDosActor(py::dict &user_info): GateVActor(user_info, true) {
  fActions.insert("StartSimulationAction");
  fActions.insert("BeginOfRunAction");
  fActions.insert("BeginOfEventAction");
  fActions.insert("SteppingAction");
  fActions.insert("EndOfRunAction");
  fActions.insert("EndOfEventAction");
  fActions.insert("EndSimulationAction");
  fActions.insert("BeginOfRunActionMasterThread");
  fActions.insert("EndOfRunActionMasterThread");

  this->numberOfHits.store(0);
  this->numberOfAbsorbedEvents.store(0);
  this->evaluationFlag.store(false);
  this->runTerminationFlag.store(false);
}

GateAIDosActor::~GateAIDosActor() {}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void GateAIDosActor::InitializeUserInfo(py::dict &user_info) {
  G4cout << "[AIDosActor] InitializeUserInfo()" << G4endl;
  GateVActor::InitializeUserInfo(user_info);

  this->worldSize = DictGetVecDouble(user_info, "world_size");
  this->eventsEvalSize = DictGetInt(user_info, "events_eval_size");
  this->relErrorThreshold = DictGetDouble(user_info, "rel_error_threshold");
  this->relErrorPercentile = DictGetDouble(user_info, "rel_error_percentile");

  float maxEnergy = DictGetDouble(user_info, "max_energy") / 1000.f; // keV -> MeV
  this->numBins = DictGetInt(user_info, "num_bins");
  this->binWidth = maxEnergy / this->numBins;
  this->updateHistogramsThreshold = DictGetInt(user_info, "update_histograms_threshold");
  this->voxelSize = DictGetDouble(user_info, "voxel_size") / 1000.f; // mm -> m

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
      G4cout << "[AIDosActor] Unknown convergence_region_mode '" << this->convergenceRegionModeString
             << "', defaulting to 'all'." << G4endl;
      this->convergenceRegionMode = ConvergenceRegionMode::ALL;
      this->convergenceRegionModeString = "all";
    }
  }
  G4cout << "[AIDosActor] Convergence region mode : " << this->convergenceRegionModeString << G4endl;

  this->generalChannelName = DictGetStr(user_info, "channel_name_general");
  this->beamChannelName    = DictGetStr(user_info, "channel_name_beam");
  this->roomChannelName    = DictGetStr(user_info, "channel_name_room");
  this->objectChannelName  = DictGetStr(user_info, "channel_name_object");

  this->outputPath     = DictGetStr(user_info, "output_path");
  this->outputFileName = DictGetStr(user_info, "output_filename");
  this->tracerType     = DictGetStr(user_info, "tracer_type");

  this->scoringQuantities.clear();
  auto quantities = DictGetVecStr(user_info, "scoring_quantities");
  for (const auto& q : quantities)
      this->scoringQuantities.insert(q);

  this->needsStepLengths = this->scoringQuantities.count("energy_fluence") ||
                            this->scoringQuantities.count("air_kerma") ||
                            this->scoringQuantities.count("ambient_dose");

  if (this->scoringQuantities.count("ambient_dose"))
      this->adephPath = DictGetStr(user_info, "adeph_path");
  if (this->scoringQuantities.count("air_kerma"))
      this->muEnAirPath = DictGetStr(user_info, "mu_en_air_path");
}

void GateAIDosActor::InitializeCpp() {
  G4cout << "[AIDosActor] InitializeCpp()" << G4endl;

  this->numberOfAbsorbedEvents.store(0);
  this->evaluationFlag.store(false);
  this->numberOfHits.store(0);

  this->crf = std::make_shared<RadFiled3D::CartesianRadiationField>(
    glm::vec3(this->worldSize[0] / 1000,
              this->worldSize[1] / 1000,
              this->worldSize[2] / 1000),
    glm::vec3(this->voxelSize));

  this->voxelCounts = this->crf->get_voxel_counts();
  this->voxelDims   = this->crf->get_voxel_dimensions();

  float dimCm_x = this->voxelDims.x * 100.f;
  float dimCm_y = this->voxelDims.y * 100.f;
  float dimCm_z = this->voxelDims.z * 100.f;
  this->voxelVolumeCm3 = dimCm_x * dimCm_y * dimCm_z;

  this->LoadLookupTable(this->adephPath, this->adephTable);
  this->LoadLookupTable(this->muEnAirPath, this->muEnAirTable, 1e-3);

  this->numVoxels = this->voxelCounts.x * this->voxelCounts.y * this->voxelCounts.z;
  this->mutexes = std::make_shared<std::vector<std::mutex>>(this->numVoxels);

  this->halfFieldDims = glm::vec3(
      static_cast<float>(this->voxelCounts.x * this->voxelDims.x * 1000) / 2.f,
      static_cast<float>(this->voxelCounts.y * this->voxelDims.y * 1000) / 2.f,
      static_cast<float>(this->voxelCounts.z * this->voxelDims.z * 1000) / 2.f);

  this->generalChannel = this->CreateGeneralChannel();
  this->tracer = std::make_shared<RadFiled3D::DDAGridTracer>(*this->generalChannel);

  this->beamChannel   = this->CreateEnergyChannel(this->beamChannelName);
  this->roomChannel   = this->CreateEnergyChannel(this->roomChannelName);
  this->objectChannel = this->CreateEnergyChannel(this->objectChannelName);

  G4cout << "[AIDosActor] World size (m)  : " << worldSize[0]/1000.0 << " x " << worldSize[1]/1000.0 << " x " << worldSize[2]/1000.0 << G4endl;
  G4cout << "[AIDosActor] Voxel counts    : " << this->voxelCounts.x << " x " << this->voxelCounts.y << " x " << this->voxelCounts.z << " = " << this->numVoxels << " voxels" << G4endl;
  G4cout << "[AIDosActor] Voxel size (m)  : " << this->voxelDims.x << " x " << this->voxelDims.y << " x " << this->voxelDims.z << G4endl;
  G4cout << "[AIDosActor] Energy bins     : " << numBins << ", bin width (MeV) = " << binWidth << G4endl;
  G4cout << "[AIDosActor] Tracer type     : " << tracerType << G4endl;
  G4cout << "[AIDosActor] relError p-quantile : " << relErrorPercentile * 100.0f << " %" << G4endl;
  G4cout << "[AIDosActor] relError threshold  : " << relErrorThreshold << " %" << G4endl;

  this->InitializeVoxelRegions();
}

void GateAIDosActor::LoadLookupTable(const std::string& path, LookupTable& table, double energyScaleFactor) {
    if (path.empty() || path.find_first_not_of(" \t\n\r") == std::string::npos)
        return;
    G4cout << "[AIDosActor] Loading lookup table from " << path << G4endl;
    std::ifstream f(path);
    if (!f.is_open()) {
        G4Exception("GateAIDosActor", "LoadLookupTable", FatalException,
                    ("Cannot open lookup table: " + path).c_str());
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        // col 1: energy (unit depends on energyScaleFactor: GeV for adeph.dat, keV for mu_en_air)
        // col 2: conversion coefficient (h*(10) [Sv·cm²] or μ_en/ρ [cm²/g])
        // col 3: ignored (σ for adeph.dat, μ/ρ for mu_en_air)
        double energy, val, unused;
        ss >> energy >> val >> unused;
        table.energiesMeV.push_back(energy * energyScaleFactor);
        table.values.push_back(val);
    }
    G4cout << "[AIDosActor] Loaded " << table.energiesMeV.size()
           << " entries from " << path << G4endl;
}

std::shared_ptr<RadFiled3D::VoxelGridBuffer> GateAIDosActor::CreateEnergyChannel(const std::string& name) {
  crf->add_channel(name);
  auto channel = crf->get_channel(name);

  channel->add_layer<float>("transported_energy", 0.f, "MeV");

  if (this->needsStepLengths)
      channel->add_custom_layer<RadFiled3D::HistogramVoxel<float>>(
          "step_lengths", RadFiled3D::HistogramVoxel<float>(numBins, binWidth, nullptr), 0.f, "cm");

  if (this->scoringQuantities.count("energy_fluence"))
      channel->add_layer<float>("energy_fluence", 0.f, "MeV/cm2");
  if (this->scoringQuantities.count("air_kerma"))
      channel->add_layer<float>("air_kerma", 0.f, "Gy");
  if (this->scoringQuantities.count("ambient_dose"))
      channel->add_layer<float>("ambient_dose", 0.f, "Sv");

  return channel;
}

std::shared_ptr<RadFiled3D::VoxelGridBuffer> GateAIDosActor::CreateGeneralChannel() {
  auto channel = this->CreateEnergyChannel(this->generalChannelName);

  // Convergence tracking (generalChannel only)
  channel->add_layer<int>("hits", 0, "counts");
  channel->add_custom_layer<RadFiled3D::HistogramVoxel<float>>(
      "histograms", RadFiled3D::HistogramVoxel<float>(numBins, binWidth, nullptr), 0.f, "MeV");
  channel->add_layer<int>("update_counts", 0, "counts");
  channel->add_custom_layer<RadFiled3D::HistogramVoxel<float>>(
      "histogram_variances_means", RadFiled3D::HistogramVoxel<float>(numBins, binWidth, nullptr), 0.f, "variances_means");
  channel->add_custom_layer<RadFiled3D::HistogramVoxel<float>>(
      "histogram_variances", RadFiled3D::HistogramVoxel<float>(numBins, binWidth, nullptr), 0.f, "variances");
  channel->add_layer<float>("eps_rel", 1.f, "percent");

  // Voxel region labels — written once in InitializeVoxelRegions, kept in output
  channel->add_layer<int>("voxel_region", static_cast<int>(VoxelRegion::WORLD), "label");

  return channel;
}

void GateAIDosActor::FinalizeGeneralChannel() {
  this->FinalizeQuantities(this->generalChannel.get());

  this->generalChannel->remove_layer("hits");
  this->generalChannel->remove_layer("histograms");
  this->generalChannel->remove_layer("update_counts");
  this->generalChannel->remove_layer("eps_rel");
  this->generalChannel->remove_layer("histogram_variances");
  this->generalChannel->remove_layer("histogram_variances_means");
}

void GateAIDosActor::InitializeVoxelRegions() {
  G4cout << "[AIDosActor] InitializeVoxelRegions()" << G4endl;

  auto *transportMgr = G4TransportationManager::GetTransportationManager();
  G4Navigator *navigator = transportMgr->GetNavigatorForTracking();

  size_t worldCount = 0, enclosureCount = 0, objectCount = 0;

  G4cout << "[AIDosActor] Classifying voxel regions (x slices: "
         << this->voxelCounts.x << ", total voxels: " << this->numVoxels << ")" << G4endl;

  for (int ix : tqdm::range(0, this->voxelCounts.x)) {
    for (int iy = 0; iy < this->voxelCounts.y; ++iy) {
      for (int iz = 0; iz < this->voxelCounts.z; ++iz) {
        const double x_mm = (ix + 0.5) * this->voxelDims.x * 1000.0 - this->halfFieldDims.x;
        const double y_mm = (iy + 0.5) * this->voxelDims.y * 1000.0 - this->halfFieldDims.y;
        const double z_mm = (iz + 0.5) * this->voxelDims.z * 1000.0 - this->halfFieldDims.z;
        const double dx = 0.5 * this->voxelDims.x * 1000.0;
        const double dy = 0.5 * this->voxelDims.y * 1000.0;
        const double dz = 0.5 * this->voxelDims.z * 1000.0;

        size_t worldHits = 0, enclosureHits = 0, objectHits = 0;

        const G4ThreeVector center(x_mm, y_mm, z_mm);
        VoxelRegion centerRegion = ClassifyPoint(navigator, center);
        if      (centerRegion == VoxelRegion::WORLD)     ++worldHits;
        else if (centerRegion == VoxelRegion::ENCLOSURE) ++enclosureHits;
        else                                             ++objectHits;

        const double xs[2] = {x_mm - dx, x_mm + dx};
        const double ys[2] = {y_mm - dy, y_mm + dy};
        const double zs[2] = {z_mm - dz, z_mm + dz};

        for (double sx : xs) {
          for (double sy : ys) {
            for (double sz : zs) {
              VoxelRegion r = ClassifyPoint(navigator, G4ThreeVector(sx, sy, sz));
              if      (r == VoxelRegion::WORLD)     ++worldHits;
              else if (r == VoxelRegion::ENCLOSURE) ++enclosureHits;
              else                                  ++objectHits;
            }
          }
        }

        const size_t total = worldHits + enclosureHits + objectHits;
        VoxelRegion region = VoxelRegion::OBJECT;
        if (total > 0 && worldHits == total) {
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

  G4cout << "[AIDosActor] WORLD voxels     : " << worldCount     << G4endl;
  G4cout << "[AIDosActor] ENCLOSURE voxels : " << enclosureCount << G4endl;
  G4cout << "[AIDosActor] OBJECT voxels    : " << objectCount    << G4endl;
}

// ---------------------------------------------------------------------------
// Simulation hooks
// ---------------------------------------------------------------------------

void GateAIDosActor::StartSimulationAction() {}

void GateAIDosActor::BeginOfRunActionMasterThread(int /*run_id*/) {
  this->runTerminationFlag.store(false);
}

void GateAIDosActor::BeginOfRunAction(const G4Run * /*run*/) {}

void GateAIDosActor::BeginOfEventAction(const G4Event * /*event*/) {
  auto &tls = fThreadLocalData.Get();
  std::unordered_set<G4int>().swap(tls.hasScattered);

  size_t currentEvents = numberOfAbsorbedEvents.fetch_add(1) + 1;
  if (currentEvents % this->eventsEvalSize == 0) {
      evaluationFlag.store(true, std::memory_order_relaxed);
  }
}

void GateAIDosActor::SteppingAction(G4Step *step) {
  G4Track *track = step->GetTrack();
  if ((track->GetKineticEnergy() <= 0.0)
      || (track->GetDefinition() != G4Gamma::Definition())) {
    return;
  }

  auto &tls = fThreadLocalData.Get();
  auto &hasScattered = tls.hasScattered;

  G4int trackID = track->GetTrackID();
  bool scattered = (hasScattered.find(trackID) != hasScattered.end());

  if (track->GetCurrentStepNumber() == 1) {
    if (track->GetParentID() > 0 || track->GetCreatorProcess() != nullptr) {
      scattered = true;
      hasScattered.insert(trackID);
    }
  }

  auto *pre  = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();
  auto *scatterProcess = post->GetProcessDefinedStep();
  if (scatterProcess) {
    if (scatterProcess->GetProcessType() == fElectromagnetic) {
      const auto st = scatterProcess->GetProcessSubType();
      if (st == fComptonScattering ||
          st == fRayleigh ||
          st == fPhotoElectricEffect ||
          st == fGammaConversion ||
          st == fGammaGeneralProcess) {
        scattered = true;
        hasScattered.insert(trackID);
      }
    }
  }

  auto prePos    = pre->GetPosition();
  auto postPos   = post->GetPosition();
  auto energyMeV = pre->GetKineticEnergy();

  glm::vec3 prePosVec  = (glm::vec3(prePos[0],  prePos[1],  prePos[2])  + this->halfFieldDims) / glm::vec3(1000);
  glm::vec3 postPosVec = (glm::vec3(postPos[0], postPos[1], postPos[2]) + this->halfFieldDims) / glm::vec3(1000);
  tls.voxelHits = static_cast<RadFiled3D::DDAGridTracer*>(this->tracer.get())->trace_with_lengths(prePosVec, postPosVec);

  size_t binIndex = static_cast<size_t>(energyMeV / this->binWidth);
  if (binIndex >= static_cast<size_t>(this->numBins))
    binIndex = this->numBins - 1;

  for (const RadFiled3D::VoxelHit& voxelHit : tls.voxelHits) {
    float lengthCm = voxelHit.length * 100.f;
    this->AccumulateVoxelHits(voxelHit.index, lengthCm, energyMeV, scattered, binIndex);
  }
}

void GateAIDosActor::EndOfEventAction(const G4Event * /*event*/) {
  if (this->evaluationFlag.load()) {
    this->MaybeEvaluateAndStop();
  }
  if (runTerminationFlag.load()) {
    fSourceManager->SetRunTerminationFlag(true);
  }
}

void GateAIDosActor::EndOfRunAction(const G4Run * /*run*/) {}

int GateAIDosActor::EndOfRunActionMasterThread(int /*run_id*/) {
  return 0;
}

void GateAIDosActor::EndSimulationAction() {
  const std::string geometryInfo =
      "convergence_region_mode=" + this->convergenceRegionModeString +
      ", rel_error_threshold="   + std::to_string(this->relErrorThreshold) +
      ", rel_error_percentile="  + std::to_string(this->relErrorPercentile);

  std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata =
      std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
          RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
              static_cast<int>(this->numberOfAbsorbedEvents.load()),
              geometryInfo,
              "",
              RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
                  glm::vec3(0.f), glm::vec3(0.f), 0.f, "")),
          RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
              "AIDos", "", "", ""));

  this->FinalizeGeneralChannel();
  this->FinalizeQuantities(this->beamChannel.get());
  this->FinalizeQuantities(this->roomChannel.get());
  this->FinalizeQuantities(this->objectChannel.get());

  RadFiled3D::Storage::FieldStore::store(
      this->crf,
      metadata,
      this->outputPath + "/" + this->outputFileName,
      RadFiled3D::Storage::StoreVersion::V1);
}

// ---------------------------------------------------------------------------
// Internal logic
// ---------------------------------------------------------------------------

void GateAIDosActor::AccumulateVoxelHits(size_t voxelIndex, float segmentLengthCm, float energyMeV, bool scattered, size_t binIndex) {
  auto &regionVoxel = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("voxel_region", voxelIndex).get_data();
  VoxelRegion region = static_cast<VoxelRegion>(regionVoxel);

  this->numberOfHits.fetch_add(1);

  std::lock_guard<std::mutex> lock((*this->mutexes)[voxelIndex]);

  auto& generalHits = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("hits", voxelIndex).get_data();
  auto& generalTransportedEnergy = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("transported_energy", voxelIndex);
  auto& generalVoxelHist = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("histograms", voxelIndex);

  generalHits += 1;
  generalTransportedEnergy += energyMeV;

  auto* generalHistData = &generalVoxelHist.get_data();
  generalHistData[binIndex] += 1.f;

  auto targetChannel = this->beamChannel;
  if (scattered) {
    targetChannel = (region == VoxelRegion::OBJECT || region == VoxelRegion::ENCLOSURE)
                    ? this->objectChannel
                    : this->roomChannel;
  }

  targetChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("transported_energy", voxelIndex).get_data() += energyMeV;

  if (this->needsStepLengths) {
      auto* sl = &targetChannel->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("step_lengths", voxelIndex).get_data();
      sl[binIndex] += segmentLengthCm;
      auto* slGeneral = &this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("step_lengths", voxelIndex).get_data();
      slGeneral[binIndex] += segmentLengthCm;
  }

  if (generalHits % this->updateHistogramsThreshold == 0) {
    auto& updateCounts       = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", voxelIndex).get_data();
    auto& variancesVoxel     = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("histogram_variances", voxelIndex);
    auto& variancesMeansVoxel = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("histogram_variances_means", voxelIndex);

    auto* variancesData      = &variancesVoxel.get_data();
    auto* variancesMeansData = &variancesMeansVoxel.get_data();

    updateCounts += 1;

    for (size_t j = 0; j < static_cast<size_t>(this->numBins); j++) {
      float generalHistMean = generalHistData[j] / generalHits;
      float deltaJ  = generalHistMean - variancesMeansData[j];
      variancesMeansData[j] += deltaJ / updateCounts;
      float deltaJ2 = generalHistMean - variancesMeansData[j];
      variancesData[j] += deltaJ * deltaJ2;
    }
  }
}

void GateAIDosActor::FinalizeQuantities(RadFiled3D::VoxelGridBuffer* ch) {
    if (needsStepLengths) {
        const bool doFluence  = scoringQuantities.count("energy_fluence") > 0;
        const bool doAmbient  = scoringQuantities.count("ambient_dose")   > 0;
        const bool doKerma    = scoringQuantities.count("air_kerma")      > 0;
        const float rhoAir    = 1.293e-3f;
        const float MEV_TO_GY = 1.602e-13f * 1e6f / 1.293f;
        const float norm      = this->voxelVolumeCm3 * static_cast<float>(this->numberOfAbsorbedEvents.load());

        for (size_t i = 0; i < numVoxels; i++) {
            auto* sl = &ch->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("step_lengths", i).get_data();
            float sumEjLj = 0.f, sumH10Lj = 0.f, sumKermaLj = 0.f;
            for (size_t j = 0; j < static_cast<size_t>(numBins); j++) {
                float lj = sl[j];
                if (lj == 0.f) continue;
                float Ej = (j + 0.5f) * binWidth;
                if (doFluence) sumEjLj   += Ej * lj;
                if (doAmbient) sumH10Lj  += static_cast<float>(adephTable.Interpolate(Ej)) * lj;
                if (doKerma)   sumKermaLj += static_cast<float>(muEnAirTable.Interpolate(Ej)) * rhoAir * Ej * lj;
            }
            if (doFluence) ch->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energy_fluence", i).get_data() = sumEjLj / norm;
            if (doAmbient) ch->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("ambient_dose",   i).get_data() = sumH10Lj / norm;
            if (doKerma)   ch->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("air_kerma",      i).get_data() = (sumKermaLj / norm) * MEV_TO_GY;
        }
        ch->remove_layer("step_lengths");
    }
}

void GateAIDosActor::StopSimulation() {
  runTerminationFlag.store(true, std::memory_order_relaxed);
}

bool GateAIDosActor::ShouldIncludeRegion(VoxelRegion region) const {
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

void GateAIDosActor::MaybeEvaluateAndStop() {
  std::unique_lock<std::mutex> evalLock(this->evalMutex, std::try_to_lock);
  if (!evalLock.owns_lock()) return;

  if (!this->evaluationFlag.load()) return;
  this->evaluationFlag.store(false);

  std::vector<float> errors;
  errors.reserve(this->numVoxels);

  std::vector<float> epsRelSnapshot(this->numVoxels, this->DEFAULT_ERROR_VALUE);
  std::vector<bool>  includeVoxel(this->numVoxels, false);

  // for (size_t i = 0; i < this->numVoxels; i++) {
  //   int updateCounts = 0;
  //   std::vector<float> variancesSnapshot;

  //   {
  //     std::lock_guard<std::mutex> lock((*this->mutexes)[i]);
  //     auto &regionVoxel = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("voxel_region", i).get_data();
  //     VoxelRegion region = static_cast<VoxelRegion>(regionVoxel);
  //     if (!this->ShouldIncludeRegion(region)) continue;
  //     includeVoxel[i] = true;

  //     updateCounts = this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", i).get_data();
  //     if (updateCounts > this->MIN_UPDATE_COUNTS) {
  //       auto &variances    = this->generalChannel->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("histogram_variances", i);
  //       auto *variancesData = &variances.get_data();
  //       variancesSnapshot.assign(variancesData, variancesData + this->numBins);
  //     }
  //   }

  //   float epsRelValue = this->DEFAULT_ERROR_VALUE;
  //   if (updateCounts > this->MIN_UPDATE_COUNTS) {
  //     float sumM2OverCounts = 0.f;
  //     for (size_t j = 0; j < static_cast<size_t>(this->numBins); j++)
  //       sumM2OverCounts += variancesSnapshot[j] / updateCounts;
  //     epsRelValue = sumM2OverCounts * (this->VARIANCE_SCALING_FACTOR / this->numBins);
  //   }
  //   epsRelSnapshot[i] = epsRelValue;
  //   errors.push_back(epsRelValue);
  // }

  // if (errors.empty()) {
  //   G4cout << "[AIDosActor] No voxels selected for convergence check; skipping." << G4endl;
  //   return;
  // }

  // for (size_t i = 0; i < this->numVoxels; i++) {
  //   if (!includeVoxel[i]) continue;
  //   std::lock_guard<std::mutex> lock((*this->mutexes)[i]);
  //   auto &epsRel = this->generalChannel->get_voxel_flat<float>("eps_rel", i);
  //   epsRel = epsRelSnapshot[i];
  // }

  for (size_t i = 0; i < this->numVoxels; i++) {
    // voxel_region ist nach InitializeVoxelRegions read-only — kein Lock nötig
    VoxelRegion region = static_cast<VoxelRegion>(
        this->generalChannel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("voxel_region", i).get_data());
    if (!this->ShouldIncludeRegion(region)) continue;
    includeVoxel[i] = true;

    // Approximativer Snapshot — leicht veraltete Werte sind für Konvergenzcheck akzeptabel
    int updateCounts = this->generalChannel
        ->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", i).get_data();

    float epsRelValue = this->DEFAULT_ERROR_VALUE;
    if (updateCounts > this->MIN_UPDATE_COUNTS) {
      auto* variancesData = &this->generalChannel
          ->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>("histogram_variances", i).get_data();
      float sumM2OverCounts = 0.f;
      for (size_t j = 0; j < static_cast<size_t>(this->numBins); j++)
        sumM2OverCounts += variancesData[j] / updateCounts;
      epsRelValue = sumM2OverCounts * (this->VARIANCE_SCALING_FACTOR / this->numBins);
    }
    epsRelSnapshot[i] = epsRelValue;
    errors.push_back(epsRelValue);
  }

  if (errors.empty()) {
    G4cout << "[AIDosActor] No voxels selected for convergence check; skipping." << G4endl;
    return;
  }

  // eps_rel direkt schreiben — Worker-Threads lesen diesen Layer nicht
  for (size_t i = 0; i < this->numVoxels; i++) {
    if (!includeVoxel[i]) continue;
    this->generalChannel->get_voxel_flat<float>("eps_rel", i) = epsRelSnapshot[i];
  }

  std::sort(errors.begin(), errors.end());
  size_t percentileIndex = static_cast<size_t>(errors.size() * this->relErrorPercentile);
  if (percentileIndex >= errors.size())
    percentileIndex = errors.size() - 1;
  float quantileValue = errors[percentileIndex];

  size_t numBelow = 0;
  for (const auto& err : errors)
    if (err < quantileValue) ++numBelow;

  float cleared_percentage = (static_cast<float>(numBelow) / errors.size()) * 100.f;
  size_t currentAbsorbedEvents = numberOfAbsorbedEvents.load();

  G4cout << "\n[AIDosActor] ===== Statistical convergence check =====" << G4endl;
  G4cout << "[AIDosActor] Primary photons simulated      : " << currentAbsorbedEvents << G4endl;
  G4cout << "[AIDosActor] Voxels in check                : " << errors.size() << G4endl;
  G4cout << "[AIDosActor] Target eps_rel quantile        : " << (relErrorPercentile * 100.0f) << " %" << G4endl;
  G4cout << "[AIDosActor] Target eps_rel threshold       : " << relErrorThreshold << G4endl;
  G4cout << "[AIDosActor] Measured eps_rel at quantile   : " << quantileValue << G4endl;
  G4cout << "[AIDosActor] Voxels below threshold         : " << numBelow << " / " << errors.size() << " (" << cleared_percentage << " %)" << G4endl;

  bool stop = (quantileValue <= this->relErrorThreshold);
  if (stop) {
    G4cout << "[AIDosActor] Convergence reached -> requesting stop." << G4endl;
    this->StopSimulation();
  } else {
    G4cout << "[AIDosActor] Convergence NOT reached -> continue simulation." << G4endl;
  }
}
