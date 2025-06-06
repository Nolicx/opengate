/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef RF3Actor_h
#define RF3Actor_h

#include "G4Cache.hh"
#include "GateHelpers.h"
#include "GateVActor.h"
#include <pybind11/stl.h>
#include <mutex>
#include <condition_variable>

namespace py = pybind11;

class GateRF3Actor : public GateVActor {

public:
    // Callback function
    using CallbackFunctionType = std::function<void(GateRF3Actor *)>;

    explicit GateRF3Actor(py::dict &user_info);

    // ~GateRF3Actor() override;
    
    void InitializeUserInfo(py::dict &user_info) override;

    void InitializeCpp() override;
    // void StartSimulationAction() override;

    void BeginOfRunAction(const G4Run * /*run*/) override;  // Called at simulation start
    void EndOfRunAction(const G4Run * /*run*/) override;    // Called at simulation end

    // void PreUserTrackingAction(const G4Track *track) override;

    void SteppingAction(G4Step *) override;     //Called when step in attached volume
    
    // void EndOfEventAction(const G4Event *event) override;
    // void EndOfRunAction(const G4Run *run) override;
    // void EndOfSimulationWorkerAction(const G4Run *run) override;
    // void EndSimulationAction() override;
    
    void SetCallbackFunction(CallbackFunctionType &f);  // Set the user "apply" function (python)

    // int GetCurrentNumberOfHits() const;
    // int GetCurrentRunId() const;

    const std::vector<double> GetEnergy() const;
    const std::vector<double> GetPrePositionX() const;
    const std::vector<double> GetPrePositionY() const;
    const std::vector<double> GetPrePositionZ() const;
    const std::vector<double> GetPostPositionX() const;
    const std::vector<double> GetPostPositionY() const;
    const std::vector<double> GetPostPositionZ() const;
    // std::vector<double> GetDirectionX() const;
    // std::vector<double> GetDirectionY() const;
    // std::vector<double> GetDirectionZ() const;

    void BeginOfRunActionMasterThread(int run_id) override;  // Called at simulation start (master thread only)
    int EndOfRunActionMasterThread(int run_id) override;  // Called at simulation end (master thread only)

protected:
    CallbackFunctionType fCallbackFunction;
    struct threadLocalT {
        std::vector<double> fEnergy;    //Always Pre
        std::vector<double> fPrePositionX;
        std::vector<double> fPrePositionY;
        std::vector<double> fPrePositionZ;
        std::vector<double> fPostPositionX;
        std::vector<double> fPostPositionY;
        std::vector<double> fPostPositionZ;
        // std::vector<double> fDirectionX;    //Always Post
        // std::vector<double> fDirectionY;    //Always Post
        // std::vector<double> fDirectionZ;    //Always Post
        int fCurrentNumberOfHits;
        int fCurrentRunId;
    };
    G4Cache<threadLocalT> fThreadLocalData;
    int photonCount;

    static std::mutex sBarrierMutex;
    static std::condition_variable sBarrierCond;
    static int sBarrierCount;
    static int sNumThreads;
    static int instanceCount;
};

#endif // RadFiled3DActor_h