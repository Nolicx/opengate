/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   ------------------------------------ -------------- */

#include "GateRF3Actor.h"
#include "G4Threading.hh"
#include "GateHelpers.h"
#include "GateHelpersDict.h"
#include "G4RunManager.hh"
#include "digitizer/GateHelpersDigitizer.h"

G4Mutex LocalThreadDataMutex = G4MUTEX_INITIALIZER;
G4Condition sBarrierCondition = G4CONDITION_INITIALIZER; 

GateRF3Actor::GateRF3Actor(py::dict &user_info): GateVActor(user_info, true) {
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

  fHitsBatchSize = 0;
  fNumberOfHits = 0;
  fNumberOfAbsorbedEvents = 0;
  runTerminationFlag = false;

  sBarrierCount = 0;
  sNumThreads = 0;
}

GateRF3Actor::~GateRF3Actor() {
  // for debug
}

void GateRF3Actor::InitializeUserInfo(py::dict &user_info) {
  GateVActor::InitializeUserInfo(user_info);
  fHitsBatchSize = DictGetInt(user_info, "hits_batch_size");
}

void GateRF3Actor::InitializeCpp() {
  // GateVActor::InitializeCpp();
  sNumThreads = G4Threading::GetNumberOfRunningWorkerThreads();
  fNumberOfAbsorbedEvents = 0;
  fNumberOfHits = 0;
}

void GateRF3Actor::StartSimulationAction() {
  // fTotalNumberOfEntries = 0;
  // fNumberOfAbsorbedEvents = 0;
}

void GateRF3Actor::BeginOfRunActionMasterThread(int run_id) {
  runTerminationFlag = false;
  sNumThreads = G4Threading::GetNumberOfRunningWorkerThreads();
}

void GateRF3Actor::BeginOfRunAction(const G4Run *run) {
  auto &l = fThreadLocalData.Get();
  // l.fCurrentRunId = run->GetRunID();
  l.fCurrentNumberOfHits = 0;
}

void GateRF3Actor::EndOfRunAction(const G4Run * /*run*/) {
  // Synchronize all threads at the end of the run
  {
    G4AutoLock mutex(&LocalThreadDataMutex);
    sBarrierCount++;
    sBarrierCondition.wait(mutex, [this]() {
      return sBarrierCount >= sNumThreads; 
    });

    if (sBarrierCount == sNumThreads){
      sBarrierCondition.notify_all();
    }
  }
  
  auto &l = fThreadLocalData.Get();  // When the run ends, we send the current remaining hits to the ARF
  if (l.fCurrentNumberOfHits > 0) {
    // {
    //   G4AutoLock mutex(&LocalThreadDataMutex);
      // G4cout << "Acquired EndOfRunAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
    fNumberOfHits += l.fCurrentNumberOfHits;
    fCallbackFunction(this);
      // G4cout << "Released EndOfRunAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
    // }
    ClearfThreadLocalData(l);
  }

  {
    G4AutoLock mutex(&LocalThreadDataMutex);
    sBarrierCount++;
    sBarrierCondition.wait(mutex, [this]() {
      return sBarrierCount >= sNumThreads; 
    });

    if (sBarrierCount == sNumThreads){
      sBarrierCondition.notify_all();
    }
  }
}

int GateRF3Actor::EndOfRunActionMasterThread(int run_id) {
  // Bereinige thread-lokale Daten
  // fThreadLocalData.Get().fEnergy.clear();
  // fThreadLocalData.Get().fPrePositionX.clear();
  // fThreadLocalData.Get().fPrePositionY.clear();
  // fThreadLocalData.Get().fPrePositionZ.clear();
  // fThreadLocalData.Get().fPostPositionX.clear();
  // fThreadLocalData.Get().fPostPositionY.clear();
  // fThreadLocalData.Get().fPostPositionZ.clear();
  return 0;
}

void GateRF3Actor::ClearfThreadLocalData(threadLocalT &l) {
  l.fEnergy.clear();
  l.fPrePositionX.clear();
  l.fPrePositionY.clear();
  l.fPrePositionZ.clear();
  l.fPostPositionX.clear();
  l.fPostPositionY.clear();
  l.fPostPositionZ.clear();
  // l.fDirectionX.clear();
  // l.fDirectionY.clear();
  // l.fDirectionZ.clear();
  l.fCurrentNumberOfHits = 0;
  // l.fCurrentRunId = 0;
}

void GateRF3Actor::BeginOfEventAction(const G4Event * /*event*/) {
  {
    G4AutoLock mutex(&LocalThreadDataMutex);
    fNumberOfAbsorbedEvents += 1; // Increment generated photon count 
  }
}

// void GateRF3Actor::PreUserTrackingAction(const G4Track *track) {}

void GateRF3Actor::SteppingAction(G4Step *step) {
  auto &l = fThreadLocalData.Get();

  auto *pre = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();
  auto prePos = pre->GetPosition();
  auto postPos = post->GetPosition();
  auto energy = pre->GetKineticEnergy();


  // auto *track = step->GetTrack();
  // auto *particle = track->GetDefinition();

  // G4cout  << particle->GetParticleName() << G4endl;

  l.fCurrentNumberOfHits++;
  l.fPrePositionX.push_back(prePos[0]);
  l.fPrePositionY.push_back(prePos[1]);
  l.fPrePositionZ.push_back(prePos[2]);
  l.fPostPositionX.push_back(postPos[0]);
  l.fPostPositionY.push_back(postPos[1]);
  l.fPostPositionZ.push_back(postPos[2]);
  l.fEnergy.push_back(energy);

  if (l.fCurrentNumberOfHits >= fHitsBatchSize) { //Maybe use BatchSize
    // {
    //   G4AutoLock mutex(&LocalThreadDataMutex);
    //   G4cout << "Acquired SteppingAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
      fNumberOfHits += l.fCurrentNumberOfHits; 
      fCallbackFunction(this);
    //   G4cout << "Released SteppingAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
    // }
    ClearfThreadLocalData(l);
  }
}

void GateRF3Actor::EndOfEventAction(const G4Event *event) {
  // Stop the simulation if enough photons have been tracked
  if (runTerminationFlag){
    fSourceManager->SetRunTerminationFlag(true);
  }
}

void GateRF3Actor::StopSimulation() {
  runTerminationFlag = true;
}

// void GateRF3Actor::EndOfSimulationWorkerAction(const G4Run * /*unused*/) {
// }

void GateRF3Actor::EndSimulationAction() {
}

void GateRF3Actor::SetCallbackFunction(CallbackFunctionType &f) {
    fCallbackFunction = f;
}

// int GateRF3Actor::GetCurrentNumberOfHits() const {
//   return fThreadLocalData.Get().fCurrentNumberOfHits;
// }

// int GateRF3Actor::GetCurrentRunId() const {
//   return fThreadLocalData.Get().fCurrentRunId;
// }

std::vector<double> GateRF3Actor::GetEnergy() const {
  return fThreadLocalData.Get().fEnergy;
}

// std::vector<double> GateRF3Actor::GetPrePositionX() const {
//   return fThreadLocalData.Get().fPrePositionX;
// }

// std::vector<double> GateRF3Actor::GetPrePositionY() const {
//   return fThreadLocalData.Get().fPrePositionY;
// }

// std::vector<double> GateRF3Actor::GetPrePositionZ() const {
//   return fThreadLocalData.Get().fPrePositionZ;
// }

std::vector<std::array<double, 3>> GateRF3Actor::GetPrePosition() const {
    auto& data = fThreadLocalData.Get();
    auto& x = data.fPrePositionX;
    auto& y = data.fPrePositionY;
    auto& z = data.fPrePositionZ;
    std::vector<std::array<double, 3>> result;
    for (size_t i = 0; i < x.size(); ++i) {
        result.push_back({x[i], y[i], z[i]});
    }
    return result;
}

// std::vector<double> GateRF3Actor::GetPostPositionX() const {
//   return fThreadLocalData.Get().fPostPositionX;
// }

// std::vector<double> GateRF3Actor::GetPostPositionY() const {
//   return fThreadLocalData.Get().fPostPositionY;
// }

// std::vector<double> GateRF3Actor::GetPostPositionZ() const {
//   return fThreadLocalData.Get().fPostPositionZ;
// }

std::vector<std::array<double, 3>> GateRF3Actor::GetPostPosition() const {
    auto& data = fThreadLocalData.Get();
    auto& x = data.fPostPositionX;
    auto& y = data.fPostPositionY;
    auto& z = data.fPostPositionZ;
    std::vector<std::array<double, 3>> result;
    for (size_t i = 0; i < x.size(); ++i) {
        result.push_back({x[i], y[i], z[i]});
    }
    return result;
}

// std::vector<double> GateRF3Actor::GetDirectionX() const {
//   return fThreadLocalData.Get().fDirectionX;
// }

// std::vector<double> GateRF3Actor::GetDirectionY() const {
//   return fThreadLocalData.Get().fDirectionY;
// }

// std::vector<double> GateRF3Actor::GetDirectionZ() const {
//   return fThreadLocalData.Get().fDirectionZ;
// }


