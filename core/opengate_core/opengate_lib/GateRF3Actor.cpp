/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   ------------------------------------ -------------- */

#include "GateRF3Actor.h"
#include "G4RunManager.hh"
#include "GateHelpers.h"
#include "GateHelpersDict.h"

GateRF3Actor::GateRF3Actor(py::dict &user_info): GateVActor(user_info, true) {
    fActions.insert("SteppingAction");
    fActions.insert("BeginOfRunAction");
    fActions.insert("EndOfRunAction");
}

GateRF3Actor::~GateRF3Actor() {
  // for debug
}

void GateRF3Actor::InitializeUserInfo(py::dict &user_info) {
    GateVActor::InitializeUserInfo(user_info);
}

void GateRF3Actor::SetCallbackFunction(CallbackFunctionType &f) {
    fCallbackFunction = f;
}

void GateRF3Actor::BeginOfRunAction(const G4Run *run) {
    auto &l = fThreadLocalData.Get();
    l.fCurrentRunId = run->GetRunID();
    l.fCurrentNumberOfHits = 0;
}

void GateRF3Actor::EndOfRunAction(const G4Run * /*run*/) {
    auto &l = fThreadLocalData.Get();
    // When the run ends, we send the current remaining hits to the ARF
    if (l.fCurrentNumberOfHits > 0) {
        fCallbackFunction(this);
        l.fCurrentNumberOfHits = 0;
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
}

void GateRF3Actor::SteppingAction(G4Step *step) {
    auto &l = fThreadLocalData.Get();

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

    if (l.fCurrentNumberOfHits >= 50) { //Maybe use BatchSize
        fCallbackFunction(this);
        l.fCurrentNumberOfHits = 0;
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
}

int GateRF3Actor::GetCurrentNumberOfHits() const {
  return fThreadLocalData.Get().fCurrentNumberOfHits;
}

int GateRF3Actor::GetCurrentRunId() const {
  return fThreadLocalData.Get().fCurrentRunId;
}

std::vector<double> GateRF3Actor::GetEnergy() const {
  return fThreadLocalData.Get().fEnergy;
}

std::vector<double> GateRF3Actor::GetPrePositionX() const {
  return fThreadLocalData.Get().fPrePositionX;
}

std::vector<double> GateRF3Actor::GetPrePositionY() const {
  return fThreadLocalData.Get().fPrePositionY;
}

std::vector<double> GateRF3Actor::GetPrePositionZ() const {
  return fThreadLocalData.Get().fPrePositionZ;
}

std::vector<double> GateRF3Actor::GetPostPositionX() const {
  return fThreadLocalData.Get().fPrePositionX;
}

std::vector<double> GateRF3Actor::GetPostPositionY() const {
  return fThreadLocalData.Get().fPrePositionY;
}

std::vector<double> GateRF3Actor::GetPostPositionZ() const {
  return fThreadLocalData.Get().fPrePositionZ;
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