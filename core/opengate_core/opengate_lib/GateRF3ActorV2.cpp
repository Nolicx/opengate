/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   ------------------------------------ -------------- */

#include "GateRF3ActorV2.h"
#include "G4Threading.hh"
#include "GateHelpers.h"
#include "GateHelpersDict.h"
#include "G4RunManager.hh"
#include "G4TransportationManager.hh"
#include "G4Navigator.hh"
#include "G4ThreeVector.hh"
#include <glm/vec3.hpp>
#include <RadFiled3D/storage/RadiationFieldStore.hpp>

namespace {
constexpr auto WORLD_NAME = "world";
// constexpr auto CARM_NAME  = "c_arm";

bool IsWorld(const G4VPhysicalVolume* vol) {
  return vol && vol->GetName() == WORLD_NAME;
}

// bool IsCArm(const G4VPhysicalVolume* vol) {
//   return vol && vol->GetName() == CARM_NAME;
// }

bool IsObject(const G4VPhysicalVolume* vol) {
  return vol && !IsWorld(vol);
}
}

GateRF3ActorV2::GateRF3ActorV2(py::dict &user_info): GateVActor(user_info, true) {
  fActions.insert("StartSimulationAction");
  fActions.insert("BeginOfRunAction");
  fActions.insert("BeginOfEventAction");
  // fActions.insert("PreUserTrackingAction");
  // fActions.insert("PostUserTrackingAction");
  fActions.insert("SteppingAction");
  fActions.insert("EndOfRunAction");
  fActions.insert("EndOfEventAction");
  // fActions.insert("EndOfSimulationWorkerAction");
  fActions.insert("EndSimulationAction");
  fActions.insert("BeginOfRunActionMasterThread");
  fActions.insert("EndOfRunActionMasterThread");

  // Initialize atomic counters
  this->numberOfHits.store(0);
  this->numberOfAbsorbedEvents.store(0);
  this->evaluationFlag.store(false);
  this->runTerminationFlag.store(false);

  this->fAirMaterial = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
}

GateRF3ActorV2::~GateRF3ActorV2() {
  // for debug
}

void GateRF3ActorV2::InitializeUserInfo(py::dict &user_info) {
  GateVActor::InitializeUserInfo(user_info);
  this->worldSize = DictGetVecDouble(user_info, "world_size");
  this->eventsEvalSize = DictGetInt(user_info, "events_eval_size");
  this->relErrorThreshold = DictGetDouble(user_info, "rel_error_threshold");
  this->relErrorPercentile = DictGetDouble(user_info, "rel_error_percentile");
  this->maxEnergy = DictGetDouble(user_info, "max_energy") / 1000; //MeV
  this->numBins = DictGetInt(user_info, "num_bins");
  this->binWidth = this->maxEnergy / this->numBins;
  this->updateHistogramsThreshold = DictGetInt(user_info, "update_histograms_threshold");
  this->voxelSize = DictGetDouble(user_info, "voxel_size") / 1000; // Meter
  this->channelName = DictGetStr(user_info, "channel_name");
  this->outputPath = DictGetStr(user_info, "output_path");
  this->outputFileName = DictGetStr(user_info, "output_filename");
  this->tracerType = DictGetStr(user_info, "tracer_type");
}

void GateRF3ActorV2::InitializeCpp() {
  // GateVActor::InitializeCpp();

  // Reset atomic counters
  this->numberOfAbsorbedEvents.store(0);
  this->evaluationFlag.store(false);
  this->numberOfHits.store(0);

  this->crf = std::make_shared<RadFiled3D::CartesianRadiationField>(
    glm::vec3(this->worldSize[0] / 1000, this->worldSize[1] / 1000, this->worldSize[2] / 1000),
    glm::vec3(this->voxelSize));
  this->crf->add_channel(this->channelName);
  this->channel = crf->get_channel(this->channelName);

  this->half_field_dim = glm::vec3(
						  static_cast<float>(channel->get_voxel_counts().x * channel->get_voxel_dimensions().x * 1000) / 2.f,
              static_cast<float>(channel->get_voxel_counts().y * channel->get_voxel_dimensions().y * 1000) / 2.f,
						  static_cast<float>(channel->get_voxel_counts().z * channel->get_voxel_dimensions().z * 1000) / 2.f
					  );

  this->channel->add_layer<float>("energies", 0.f, "MeV");  // energy_grid
  this->channel->add_layer<int>("hits", 0, "counts");     // histogram_hits_grid
  this->channel->add_layer<int>("update_counts", 0, "counts");     // histogram_update_grid
  this->channel->add_custom_layer<RadFiled3D::HistogramVoxel>(  // histogram_grid
          "histograms", RadFiled3D::HistogramVoxel(this->numBins, this->binWidth, nullptr), 0.f, "MeV");

  this->channel->add_custom_layer<RadFiled3D::HistogramVoxel>(
            "histogram_variances_means", RadFiled3D::HistogramVoxel(this->numBins, this->binWidth, nullptr), 0.f, "variances_means");
  this->channel->add_custom_layer<RadFiled3D::HistogramVoxel>(
            "histogram_variances", RadFiled3D::HistogramVoxel(this->numBins, this->binWidth, nullptr), 0.f, "variances");
  this->channel->add_layer<float>("eps_rel", 1.f, "percent");

  this->channel->add_layer<float>("energies_BEAM", 0.f, "MeV");
  this->channel->add_layer<float>("energies_ROOM", 0.f, "MeV");
  this->channel->add_layer<float>("energies_OBJECT", 0.f, "MeV");

  this->channel->add_layer<int>("voxel_region", static_cast<int>(VoxelRegion::WORLD), "label");

  this->mutexes = std::make_shared<std::vector<std::shared_mutex>>(this->channel->get_voxel_count());
  
  if (this->tracerType == "Linetracing") {
  this->tracer = std::make_shared<RadFiled3D::LinetracingGridTracer>(*channel);
  } else if (this->tracerType == "Sampling") {
    this->tracer = std::make_shared<RadFiled3D::SamplingGridTracer>(*channel);
  } else if (this->tracerType == "Bresenham") {
    this->tracer = std::make_shared<RadFiled3D::BresenhamGridTracer>(*channel);
  } else if (this->tracerType == "DDA") {
    this->tracer = std::make_shared<RadFiled3D::DDAGridTracer>(*channel);
  }

  // Navigator für Geometrie-Abfragen
  this->InitializeVoxelRegions();
}

void GateRF3ActorV2::InitializeVoxelRegions() {
  auto *transportMgr = G4TransportationManager::GetTransportationManager();
  G4Navigator *navigator = transportMgr->GetNavigatorForTracking();

  auto counts = channel->get_voxel_counts();
  auto dims   = channel->get_voxel_dimensions();

  for (int ix = 0; ix < counts.x; ++ix) {
    for (int iy = 0; iy < counts.y; ++iy) {
      for (int iz = 0; iz < counts.z; ++iz) {
        const double x_mm = (ix + 0.5) * dims.x * 1000.0 - half_field_dim.x;
        const double y_mm = (iy + 0.5) * dims.y * 1000.0 - half_field_dim.y;
        const double z_mm = (iz + 0.5) * dims.z * 1000.0 - half_field_dim.z;
        G4ThreeVector pos(x_mm, y_mm, z_mm);

        G4VPhysicalVolume *vol = navigator->LocateGlobalPointAndSetup(pos);

        VoxelRegion region = VoxelRegion::WORLD;
        if (vol) {
          const auto &name = vol->GetName();
          if (name == "world") {
            region = VoxelRegion::WORLD;
          // } else if (name == "c_arm") {
          //   region = VoxelRegion::CARM;
          } else {
            region = VoxelRegion::OBJECT; // alles andere
          }
        }

        auto &labelVoxel =
          channel->get_voxel<RadFiled3D::ScalarVoxel<int>>("voxel_region", ix, iy, iz);
        labelVoxel = static_cast<int>(region);
      }
    }
  }
}

void GateRF3ActorV2::StartSimulationAction() {
}

void GateRF3ActorV2::BeginOfRunActionMasterThread(int run_id) {
  this->runTerminationFlag.store(false);
  
  // this->channel->reinitialize_layer<float>("energies", 0);
  // this->channel->reinitialize_layer<int>("hits", 0);
  // this->channel->reinitialize_layer<int>("update_counts", 0);
  // this->channel->reinitialize_layer<RadFiled3D::HistogramVoxel>("histograms", 0.f);
  // this->channel->reinitialize_layer<RadFiled3D::HistogramVoxel>("histogram_variances_means", 0.f);
  // this->channel->reinitialize_layer<RadFiled3D::HistogramVoxel>("histogram_variances", 0.f);
  // this->channel->reinitialize_layer<float>("eps_rel", 1.f);
}

void GateRF3ActorV2::BeginOfRunAction(const G4Run *run) {
}

void GateRF3ActorV2::EndOfRunAction(const G4Run * /*run*/) {
}

int GateRF3ActorV2::EndOfRunActionMasterThread(int run_id) {
  return 0;
}

void GateRF3ActorV2::BeginOfEventAction(const G4Event * /*event*/) {
  auto &tls = fThreadLocalData.Get();
  tls.trackStages.clear();
  tls.everInObject.clear();

  size_t currentEvents = numberOfAbsorbedEvents.fetch_add(1) + 1;
  if (currentEvents % this->eventsEvalSize == 0) {
      evaluationFlag.store(true);
  }
}

// void GateRF3ActorV2::PreUserTrackingAction(const G4Track *track) {}

TrackStage GateRF3ActorV2::InferStageFromTrack(const G4Track *track) const {
  auto *vol = track->GetVolume();
  if (!vol || IsWorld(vol)) {
    // Kein Volume oder world → konservativ als Streustrahlung im Raum behandeln
    return TrackStage::ROOM;
  } else {
    return TrackStage::OBJECT;
  }
}

TrackStage &GateRF3ActorV2::GetOrInitTrackStage(const G4Track *track)
{
  G4int trackID  = track->GetTrackID();
  G4int parentID = track->GetParentID();
  auto &tls = fThreadLocalData.Get();
  auto &trackStages = tls.trackStages;
  auto &everInObject = tls.everInObject;
  auto &hasScattered = tls.hasScattered;

  auto it = trackStages.find(trackID);
  if (it != trackStages.end()) {
    return it->second;
  }

  TrackStage stage;
  if (parentID == 0) {
    // Primärteilchen: starten immer als BEAM
    stage = TrackStage::BEAM;
    everInObject[trackID] = false;
    hasScattered[trackID] = false;
  } else {
    bool parentEverInObject = false;
    bool parentScattered = false;

    auto parentStageIt = trackStages.find(parentID);
    if (parentStageIt != trackStages.end()) {
      auto parentStage = parentStageIt->second;

      // Flag vom Parent erben (wenn nicht vorhanden, default = false)
      auto parentEverIt = everInObject.find(parentID);
      if (parentEverIt != everInObject.end()) {
        parentEverInObject = parentEverIt->second;
      }
      auto it_parentScat = hasScattered.find(parentID);
      if (it_parentScat != hasScattered.end()) {
        parentScattered = it_parentScat->second;
      }

      if (parentStage == TrackStage::BEAM) {
        // Sekundär aus BEAM: nie BEAM, direkt ROOM/OBJECT
        stage = InferStageFromTrack(track);
      } else {
        // Sekundär erbt Streustatus (ROOM/OBJECT)
        stage = parentStage;
      }
    } else {
      // Parent unbekannt → konservativ über Geometrie
      stage = InferStageFromTrack(track);
    }

    // EverInObject für das Kind:
    // wahr, wenn Parent jemals in Objekt oder wir selbst schon als OBJECT starten
    everInObject[trackID] = parentEverInObject || (stage == TrackStage::OBJECT);
    hasScattered[trackID] = parentScattered; // initial unverändert
  }

  return trackStages.emplace(trackID, stage).first->second;
}

void GateRF3ActorV2::UpdateTrackStage(const G4Step *step, TrackStage &stage)
{
  auto *pre  = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();

  auto *preVol  = pre->GetPhysicalVolume();
  auto *postVol = post->GetPhysicalVolume();

  bool preIsWorld  = IsWorld(preVol);
  bool postIsWorld = IsWorld(postVol);
  // bool preIsCArm   = IsCArm(preVol);
  // bool postIsCArm  = IsCArm(postVol);
  bool preIsObject = IsObject(preVol);
  bool postIsObject= IsObject(postVol);

  // auto &tls          = fThreadLocalData.Get();
  // auto &everInObject = tls.everInObject;
  // G4int trackID      = step->GetTrack()->GetTrackID();

  // auto it_obj = everInObject.find(trackID);
  // bool &thisEverInObject = (it_obj != everInObject.end()) ? it_obj->second : everInObject[trackID];

  // bool enteringObject = postIsObject && !postIsWorld; // && !postIsCArm;
  // if (enteringObject) {
  //   thisEverInObject = true;
  // }

  // if (stage == TrackStage::BEAM) {
  //   if (thisEverInObject) {
  //     stage = TrackStage::OBJECT;
  //   }
  // } else {
  //   if (postIsWorld) {
  //     stage = TrackStage::ROOM;
  //   } else if (postIsObject) {
  //     stage = TrackStage::OBJECT;
  //   }
  // }

auto &tls          = fThreadLocalData.Get();
auto &hasScattered = tls.hasScattered;
auto &everInObject = tls.everInObject;

G4int trackID = step->GetTrack()->GetTrackID();
bool scattered = false;
auto it_s = hasScattered.find(trackID);
if (it_s != hasScattered.end()) {
  scattered = it_s->second;
}

// bool postIsWorld  = IsWorld(postVol);
// bool postIsObject = IsObject(postVol);

// Optional: everInObject weiterführen, falls du Patient vs reinen Roomscatter trennen willst
if (postIsObject && !postIsWorld) {
  everInObject[trackID] = true;
}

// Logik:
if (!scattered) {
  // Noch nie eine Streuung -> immer BEAM, egal wo er gerade ist
  stage = TrackStage::BEAM;
} else {
  // Bereits gestreut -> ROOM oder OBJECT je nach Geometrie
  if (postIsWorld) {
    stage = TrackStage::ROOM;
  } else {
    stage = TrackStage::OBJECT;
  }
}

}

void GateRF3ActorV2::SteppingAction(G4Step *step) {
  G4Track *track = step->GetTrack();
  if (track->GetKineticEnergy() <= 0.0){
    return;
  }
  // if (track->GetDefinition() != G4Gamma::Definition()) {
  //   return;
  // }

  auto *pre = step->GetPreStepPoint();
  // auto *post = step->GetPostStepPoint();

  TrackStage &stage = GetOrInitTrackStage(track);

auto &tls          = fThreadLocalData.Get();
auto &hasScattered = tls.hasScattered;

G4int trackID = track->GetTrackID();
auto *post = step->GetPostStepPoint();
auto *proc = post->GetProcessDefinedStep();
if (proc) {
  auto pname = proc->GetProcessName();
  // Für Photonen z.B.:
  if (pname == "compt" || pname == "phot" || pname == "Rayl") {
    hasScattered[trackID] = true;
  }
}

  this->UpdateTrackStage(step, stage);

  auto prePos = pre->GetPosition();
  auto postPos = post->GetPosition();
  auto energy = pre->GetKineticEnergy();

  std::vector<size_t> voxel_indices = this->tracer->trace(
				(glm::vec3(prePos[0], prePos[1], prePos[2]) + this->half_field_dim) / glm::vec3(1000),
	      (glm::vec3(postPos[0], postPos[1], postPos[2]) + this->half_field_dim) / glm::vec3(1000)
      );

  
  for (size_t voxel_index : voxel_indices)
  {
    { // Lock the mutex for each voxel to ensure exclusive access
      std::unique_lock lock((*this->mutexes)[voxel_index]);
      this->AccumulateVoxelHit(voxel_index, energy, stage);
    }
  }

  if (this->evaluationFlag.load())
  {
    this->MaybeEvaluateAndStop();
  }
}

void GateRF3ActorV2::AccumulateVoxelHit(size_t voxel_index, float energy, TrackStage stage){
  this->numberOfHits.fetch_add(1);

  auto& hits = this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("hits", voxel_index).get_data();
  auto& voxel_energy = this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies", voxel_index);
  auto& hist_voxel = this->channel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histograms", voxel_index);      
  
  auto* hist_data = &hist_voxel.get_data();

  hits += 1;
  voxel_energy += energy;
  size_t bin_index = static_cast<size_t>(energy / this->binWidth);
  if (bin_index >= this->numBins) {
    bin_index = this->numBins - 1;  // Clamp to max bin
  }
  hist_data[bin_index] += 1.f;

  auto &regionVoxel = this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("voxel_region", voxel_index).get_data();
  VoxelRegion region = static_cast<VoxelRegion>(regionVoxel);

  switch (stage) {
    case TrackStage::BEAM: {
      if (region == VoxelRegion::WORLD) {// || region == VoxelRegion::CARM) {
        auto &energy_beam =
          this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies_BEAM", voxel_index);
        energy_beam += energy;
      }
      break;
    }
    case TrackStage::ROOM: {
      if (region == VoxelRegion::WORLD) {
        auto &energy_room =
          this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies_ROOM", voxel_index);
        energy_room += energy;
      }
      break;
    }
    case TrackStage::OBJECT: {
      if (region == VoxelRegion::OBJECT) {// || region == VoxelRegion::CARM) {
        auto &energy_object =
          this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<float>>("energies_OBJECT", voxel_index);
        energy_object += energy;
      }
      break;
    }
  }

  if (hits % this->updateHistogramsThreshold == 0)
  {
    auto& update_counts = this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", voxel_index).get_data();
    auto& variances_voxel = this->channel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histogram_variances", voxel_index);
    auto& variances_means_voxel = this->channel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histogram_variances_means", voxel_index);
    
    auto* variances_data = &variances_voxel.get_data();
    auto* variances_means_data = &variances_means_voxel.get_data();

    update_counts += 1;

    for (size_t j = 0; j < this->numBins; j++) {
      float hist_mean = hist_data[j] / hits;
      float delta_j = hist_mean - variances_means_data[j];
      variances_means_data[j] += delta_j / update_counts;

      float delta_j2 = hist_mean - variances_means_data[j];
      variances_data[j] += delta_j * delta_j2; 
    }
  }

}

void GateRF3ActorV2::MaybeEvaluateAndStop() {
  if (this->evalMutex.try_lock())
  {
    if (this->evaluationFlag.load())
    {
      this->evaluationFlag.store(false); // Reset the evaluation flag
      size_t num_voxel = this->channel->get_voxel_count();
      std::vector<float> errors;
      errors.reserve(num_voxel);

      for (size_t i = 0; i < num_voxel; i++)
      {
        // Use shared_lock for reading during evaluation
        std::shared_lock read_lock((*this->mutexes)[i]);

        auto& update_counts = this->channel->get_voxel_flat<RadFiled3D::ScalarVoxel<int>>("update_counts", i).get_data();
        if (update_counts <= this->MIN_UPDATE_COUNTS) {
          errors.push_back(this->DEFAULT_ERROR_VALUE);
          continue; // Skip voxels with insufficient counts
        }

        float sum_m2_over_counts = 0.f;
        auto& variances = this->channel->get_voxel_flat<RadFiled3D::HistogramVoxel>("histogram_variances", i);
        auto* var_data = &variances.get_data();
        for (size_t j = 0; j < this->numBins; j++) {
          sum_m2_over_counts += var_data[j] / update_counts;
        }
        auto& eps_rel = this->channel->get_voxel_flat<float>("eps_rel", i);
        eps_rel = sum_m2_over_counts * (this->VARIANCE_SCALING_FACTOR / this->numBins);
        errors.push_back(eps_rel);
      }

      std::sort(errors.begin(), errors.end());
      size_t percentile_idx = static_cast<size_t>(errors.size() * this->relErrorPercentile);
      float quantile_value = errors[percentile_idx];
      float cleared_percentage = 0.f;
      for (const auto& err : errors) {
        if (err < quantile_value) {
          cleared_percentage += 1.f;
        }
      }
      cleared_percentage = (cleared_percentage / errors.size()) * 100.f;

      size_t currentAbsorbedEvents = numberOfAbsorbedEvents.load();
      G4cout << "Evaluating with " << currentAbsorbedEvents << " Photons." << G4endl;
      G4cout << "Eps_rel " << this->relErrorPercentile * 100 << "% Quantile: " << quantile_value << G4endl;
      G4cout << "Percentage of voxels with eps_rel < " << quantile_value << ": " << cleared_percentage << "%" << G4endl;

      if (quantile_value <= this->relErrorThreshold) {
        this->StopSimulation();
        G4cout << "Threshold cleared, stopping simulation." << G4endl;
      } else {
        G4cout << "Threshold not cleared, continuing simulation." << G4endl;
      }
    }
    this->evalMutex.unlock();
  } 
}

void GateRF3ActorV2::EndOfEventAction(const G4Event *event) {
  // Stop the simulation if enough photons have been tracked
  if (runTerminationFlag.load()){
    fSourceManager->SetRunTerminationFlag(true);
  }
}

void GateRF3ActorV2::StopSimulation() {
  runTerminationFlag.store(true);
}

// void GateRF3ActorV2::EndOfSimulationWorkerAction(const G4Run * /*unused*/) {
// }

void GateRF3ActorV2::EndSimulationAction() {

  // Using default
	std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
		RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
			0,  // Number of primary particles
			"", // Geometry Infos
			"", // Physics List
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
				glm::vec3(0.f, 0.f, 0.f), // Translation
				glm::vec3(0.f),           // Rotation
				0.f,                      // Voltage?
				""                        // Tube ID
			)
		),
		RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
			"", // Software Name
			"", // Version
			"", // Repo
			""  // Commit
		)
	);
  RadFiled3D::Storage::FieldStore::store(this->crf,
    metadata,
    this->outputPath + "/" + this->outputFileName,
    RadFiled3D::Storage::StoreVersion::V1);
}

void GateRF3ActorV2::SetCallbackFunction(CallbackFunctionType &f) {
    fCallbackFunction = f;
}


