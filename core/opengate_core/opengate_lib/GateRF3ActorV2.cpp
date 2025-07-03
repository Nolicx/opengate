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
#include <glm/vec3.hpp>
#include <RadFiled3D/storage/RadiationFieldStore.hpp>

GateRF3ActorV2::GateRF3ActorV2(py::dict &user_info): GateVActor(user_info, true) {
  fActions.insert("StartSimulationAction");
  fActions.insert("BeginOfRunAction");
  fActions.insert("BeginOfEventAction");
  fActions.insert("PreUserTrackingAction");
  fActions.insert("PostUserTrackingAction");
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

  this->sNumThreads = 0;
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
  this->sNumThreads = G4Threading::GetNumberOfRunningWorkerThreads();

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
}

void GateRF3ActorV2::StartSimulationAction() {
}

void GateRF3ActorV2::BeginOfRunActionMasterThread(int run_id) {
  this->runTerminationFlag.store(false);
  this->sNumThreads = G4Threading::GetNumberOfRunningWorkerThreads();
  
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
  size_t currentEvents = numberOfAbsorbedEvents.fetch_add(1) + 1;
  if (currentEvents % this->eventsEvalSize == 0) {
      evaluationFlag.store(true);
  }
}

// void GateRF3ActorV2::PreUserTrackingAction(const G4Track *track) {}

void GateRF3ActorV2::SteppingAction(G4Step *step) {
  auto *pre = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();
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
  }

  if (this->evaluationFlag.load())
  {
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
}

void GateRF3ActorV2::EndOfEventAction(const G4Event *event) {
  // Stop the simulation if enough photons have been tracked
  if (runTerminationFlag){
    fSourceManager->SetRunTerminationFlag(true);
  }
}

void GateRF3ActorV2::StopSimulation() {
  runTerminationFlag = true;
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


