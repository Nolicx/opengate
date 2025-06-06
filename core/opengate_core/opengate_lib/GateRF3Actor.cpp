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
#include "digitizer/GateHelpersDigitizer.h"

G4RecursiveMutex LocalThreadDataMutex = G4MUTEX_INITIALIZER;

std::mutex GateRF3Actor::sBarrierMutex;
std::condition_variable GateRF3Actor::sBarrierCond;
int GateRF3Actor::sBarrierCount = 0;
int GateRF3Actor::sNumThreads = 0;
int GateRF3Actor::instanceCount = 0;

GateRF3Actor::GateRF3Actor(py::dict &user_info): GateVActor(user_info, true) {
  G4RecursiveAutoLock mutex(&LocalThreadDataMutex);
  instanceCount++;
  G4cout << "Creating GateRF3Actor, instance: " << instanceCount 
          << ", thread ID: " << G4Threading::G4GetThreadId() << G4endl;
  if (instanceCount > 1) {
      G4cerr << "Warning: Multiple GateRF3Actor instances detected!" << G4endl;
  }

  // fActions.insert("StartSimulationAction");
  fActions.insert("BeginOfRunAction");
  // fActions.insert("BeginOfEventAction");
  // fActions.insert("PreUserTrackingAction");
  fActions.insert("SteppingAction");
  fActions.insert("EndOfRunAction");
  // fActions.insert("EndOfEventAction");
  // fActions.insert("EndOfSimulationWorkerAction");
  // fActions.insert("EndSimulationAction");
  fActions.insert("BeginOfRunActionMasterThread");
  fActions.insert("EndOfRunActionMasterThread");
}

// GateRF3Actor::~GateRF3Actor() {
//   // for debug
// }

void GateRF3Actor::InitializeUserInfo(py::dict &user_info) {
  G4cout << "Initializing GateRF3Actor user info" << G4endl;
  GateVActor::InitializeUserInfo(user_info);
}

void GateRF3Actor::InitializeCpp() {
  G4cout << "Initializing GateRF3Actor C++" << G4endl;
  GateVActor::InitializeCpp();
}

// void GateRF3Actor::StartSimulationAction() {}

void GateRF3Actor::BeginOfRunAction(const G4Run *run) {
  G4RecursiveAutoLock mutex(&LocalThreadDataMutex);
  auto &l = fThreadLocalData.Get();
  l.fCurrentRunId = run->GetRunID();
  l.fCurrentNumberOfHits = 0;
  G4cout << "BeginOfRunAction, thread ID: " << G4Threading::G4GetThreadId()
           << ", instance ptr: " << this << G4endl;

  // std::unique_lock<std::mutex> lock(sBarrierMutex);
  sNumThreads += 1; //G4RunManager::GetRunManager()->GetNumberOfThreads();
  G4cout << "Initialized barrier with " << sNumThreads << " threads" << G4endl;
  // lock.unlock();
}

void GateRF3Actor::EndOfRunAction(const G4Run * /*run*/) {
  {
    std::unique_lock<std::mutex> lock(sBarrierMutex);
    sBarrierCount++;
    G4cout << "Thread " << G4Threading::G4GetThreadId() << " reached barrier, count: " 
            << sBarrierCount << "/" << sNumThreads << G4endl;
    // Wait for all threads to reach the barrier
    sBarrierCond.wait(lock, [this]() { 
        return sBarrierCount >= sNumThreads; 
    });
    
    // Log after waking up
    G4cout << "Thread " << G4Threading::G4GetThreadId() << " passed barrier, count: " 
            << sBarrierCount << "/" << sNumThreads << G4endl;

    // Last thread resets the barrier
    if (sBarrierCount == sNumThreads) {
        G4cout << "Thread " << G4Threading::G4GetThreadId() << " reset barrier" << G4endl;
        sBarrierCond.notify_all(); // Ensure all threads are notified
      }
  }
  {
    G4RecursiveAutoLock mutex(&LocalThreadDataMutex);
    auto &l = fThreadLocalData.Get();
    G4cout << "Acquired lock in C++ EndOfRunAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
    // When the run ends, we send the current remaining hits to the ARF
    if (l.fCurrentNumberOfHits > 0) {
      fCallbackFunction(this);
      l.fCurrentNumberOfHits = 0;
      // l.fCurrentRunId = 0;
      l.fPrePositionX.clear();
      l.fPrePositionY.clear();
      l.fPrePositionZ.clear();
      l.fPostPositionX.clear();
      l.fPostPositionY.clear();
      l.fPostPositionZ.clear();
      // l.fDirectionX.clear();
      // l.fDirectionY.clear();
      // l.fDirectionZ.clear();
      l.fEnergy.clear();
    }
    G4cout << "Released lock in C++ EndOfRunAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
  }
}

// void GateRF3Actor::BeginOfEventAction(const G4Event * /*event*/) {
//   G4RecursiveAutoLock mutex(&LocalThreadDataMutex);
//   photonCount += 1; // Increment photon count for each event
// }

// void GateRF3Actor::PreUserTrackingAction(const G4Track *track) {}

void GateRF3Actor::SteppingAction(G4Step *step) {
  G4RecursiveAutoLock mutex(&LocalThreadDataMutex);
  auto &l = fThreadLocalData.Get();
  //G4cout << "Acquired lock in C++ SteppingAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;

  auto *pre = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();
  auto prePos = pre->GetPosition();
  auto postPos = post->GetPosition();
  auto energy = pre->GetKineticEnergy();

  l.fCurrentNumberOfHits++;
  l.fPrePositionX.push_back(prePos[0]);
  l.fPrePositionY.push_back(prePos[1]);
  l.fPrePositionZ.push_back(prePos[2]);
  l.fPostPositionX.push_back(postPos[0]);
  l.fPostPositionY.push_back(postPos[1]);
  l.fPostPositionZ.push_back(postPos[2]);
  l.fEnergy.push_back(energy);
  G4cout << "Current number of hits: " << l.fCurrentNumberOfHits << ", thread ID: " << G4Threading::G4GetThreadId() << G4endl;

  if (l.fCurrentNumberOfHits >= 10000) { //Maybe use BatchSize
    G4cout << "Acquired lock in C++ SteppingAction, Running CallbackFunction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
    fCallbackFunction(this);
    G4cout << "CallbackFunction executed, clearing data, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
    l.fCurrentNumberOfHits = 0;
    // l.fCurrentRunId = 0;
    l.fPrePositionX.clear();
    l.fPrePositionY.clear();
    l.fPrePositionZ.clear();
    l.fPostPositionX.clear();
    l.fPostPositionY.clear();
    l.fPostPositionZ.clear();
    // l.fDirectionX.clear();
    // l.fDirectionY.clear();
    // l.fDirectionZ.clear();
    l.fEnergy.clear();
    G4cout << "Released lock in C++ SteppingAction, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
  }
  // fSourceManager->SetRunTerminationFlag(true);
      // G4RunManager::GetRunManager()->AbortEvent();
			// G4RunManager::GetRunManager()->AbortRun(false);
}

// void GateRF3Actor::EndOfEventAction(const G4Event *event) {}

// void GateRF3Actor::EndOfSimulationWorkerAction(const G4Run * /*unused*/) {
// }

// void GateRF3Actor::EndSimulationAction() {
// }

void GateRF3Actor::SetCallbackFunction(CallbackFunctionType &f) {
    fCallbackFunction = f;
}

// int GateRF3Actor::GetCurrentNumberOfHits() const {
//   return fThreadLocalData.Get().fCurrentNumberOfHits;
// }

// int GateRF3Actor::GetCurrentRunId() const {
//   return fThreadLocalData.Get().fCurrentRunId;
// }

const std::vector<double> GateRF3Actor::GetEnergy() const {
  G4cout << "GetEnergy called, thread ID: " << G4Threading::G4GetThreadId() << G4endl;
  return fThreadLocalData.Get().fEnergy;
}

const std::vector<double> GateRF3Actor::GetPrePositionX() const {
  return fThreadLocalData.Get().fPrePositionX;
}

const std::vector<double> GateRF3Actor::GetPrePositionY() const {
  return fThreadLocalData.Get().fPrePositionY;
}

const std::vector<double> GateRF3Actor::GetPrePositionZ() const {
  return fThreadLocalData.Get().fPrePositionZ;
}

const std::vector<double> GateRF3Actor::GetPostPositionX() const {
  return fThreadLocalData.Get().fPostPositionX;
}

const std::vector<double> GateRF3Actor::GetPostPositionY() const {
  return fThreadLocalData.Get().fPostPositionY;
}

const std::vector<double> GateRF3Actor::GetPostPositionZ() const {
  return fThreadLocalData.Get().fPostPositionZ;
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

void GateRF3Actor::BeginOfRunActionMasterThread(int run_id) {
  // Initialisiere thread-lokale Daten
  fThreadLocalData.Get().fEnergy.clear();
  fThreadLocalData.Get().fPrePositionX.clear();
  fThreadLocalData.Get().fPrePositionY.clear();
  fThreadLocalData.Get().fPrePositionZ.clear();
  fThreadLocalData.Get().fPostPositionX.clear();
  fThreadLocalData.Get().fPostPositionY.clear();
  fThreadLocalData.Get().fPostPositionZ.clear();
  // ... für alle Vektoren ...
  fThreadLocalData.Get().fCurrentNumberOfHits = 0;
  fThreadLocalData.Get().fCurrentRunId = run_id;
}

int GateRF3Actor::EndOfRunActionMasterThread(int run_id) {
  // Bereinige thread-lokale Daten
  fThreadLocalData.Get().fEnergy.clear();
  fThreadLocalData.Get().fPrePositionX.clear();
  fThreadLocalData.Get().fPrePositionY.clear();
  fThreadLocalData.Get().fPrePositionZ.clear();
  fThreadLocalData.Get().fPostPositionX.clear();
  fThreadLocalData.Get().fPostPositionY.clear();
  fThreadLocalData.Get().fPostPositionZ.clear();
  return 0;
}