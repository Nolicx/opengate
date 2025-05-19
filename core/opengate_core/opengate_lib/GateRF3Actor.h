/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef RF3Actor_h
#define RF3Actor_h

#include "GateHelpers.h"
#include "GateVActor.h"
#include <pybind11/stl.h>

namespace py = pybind11;

class GateRF3Actor : public GateVActor {

public:
    // Callback function
    using CallbackFunctionType = std::function<void(GateRF3Actor *)>;

    explicit GateRF3Actor(py::dict &user_info);
    ~GateRF3Actor() override;
    
    void BeginOfRunAction(const G4Run * /*run*/) override;  // Called at simulation start
    void EndOfRunAction(const G4Run * /*run*/) override;    // Called at simulation end
    // For master thread only
    // void BeginOfRunActionMasterThread(int run_id) override {
    //     PYBIND11_OVERLOAD(void, GateVActor, BeginOfRunActionMasterThread, run_id);
    // }
    // int EndOfRunActionMasterThread(int run_id) override {
    //     PYBIND11_OVERLOAD(int, GateVActor, EndOfRunActionMasterThread, run_id);
    // }
    void InitializeUserInfo(py::dict &user_info) override;
    void SteppingAction(G4Step *) override;     //Called when step in attached volume
    void SetCallbackFunction(CallbackFunctionType &f);  // Set the user "apply" function (python)

    int GetCurrentNumberOfHits() const;
    int GetCurrentRunId() const;

    std::vector<double> GetEnergy() const;
    std::vector<double> GetPrePositionX() const;
    std::vector<double> GetPrePositionY() const;
    std::vector<double> GetPrePositionZ() const;
    std::vector<double> GetPostPositionX() const;
    std::vector<double> GetPostPositionY() const;
    std::vector<double> GetPostPositionZ() const;
    // std::vector<double> GetDirectionX() const;
    // std::vector<double> GetDirectionY() const;
    // std::vector<double> GetDirectionZ() const;

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
};

#endif // RadFiled3DActor_h