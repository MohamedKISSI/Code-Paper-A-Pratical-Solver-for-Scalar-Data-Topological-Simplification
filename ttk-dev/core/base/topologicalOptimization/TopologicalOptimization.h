


#pragma once

// base code includes
#include <ATen/ops/round_meta.h>
#include <numeric>
#include <sstream>
#ifdef TTK_ENABLE_TORCH
#include <torch/torch.h>
#include <torch/optim.h>
#endif

#include "DataTypes.h"
#include "ImplicitPreconditions.h"
#include "MultiresTriangulation.h"
#include "PersistenceDiagramUtils.h"
#include "Timer.h"
#include <ATen/Context.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/requires_grad_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zero.h>
#include <ATen/ops/zeros.h>
#include <Debug.h>
#include <Triangulation.h>
#include <array>
#include <c10/core/Backend.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/typeid.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <ostream>
#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <tuple>
#include <vector>
#include <PersistenceDiagram.h>
#include <PersistenceDiagramClustering.h>
#include <OrderDisambiguation.h>
#include <ProgressiveTopology.h>

#include <cmath>
#include <fstream>
#include <KDTree.h>
#include <ScalarFieldSmoother.h>
#include <TopologicalSimplification.h>
#include <AssignmentExhaustive.h>




namespace ttk {

  class TopologicalOptimization : virtual public Debug {
  public:
    TopologicalOptimization();

    template <typename dataType, typename triangulationType>
    int execute(dataType *const inputScalars,
                dataType *const outputScalars,
                SimplexId *const inputOffsets,
                triangulationType * triangulation,
                ttk::DiagramType &constraintDiagram,
                int *const modificationNumber,
                int *const lastChange,
                int *const idBlock
                ) const;


    template <typename dataType, typename triangulationType>
    int executeOneBlock(dataType *const inputScalars,
                dataType *const outputScalars,
                SimplexId *const inputOffsets,
                triangulationType *triangulation1,
                ttk::Triangulation *triangulationOneBlock,
                ttk::DiagramType &constraintDiagram,
                int *const modificationNumber,
                int *const lastChange,
                SimplexId vertexNumber,
                std::vector<SimplexId> &pairOnEdgeToRemove,
                std::vector<float> &listTimePersistenceDiagram,
                std::vector<float> &listTimePersistenceDiagramClustering,
                std::vector<float> &listTimeBackPropagation,
                std::vector<float> &listAveragePercentageOfModifiedVertices,
                std::vector<float> &listTimeInteration,
                std::vector<float> &listAveragePercentageOfImmobilePersistencePairs, 
                SimplexId &numberPairsInputDiagram, 
                std::vector<SimplexId> &localToGlobal,
                bool lastBlock=false,
                float lr=0.0001,
                int threadNumber=-1,
                int epochNumber=0,
                std::vector<float> maxAndMinCoordinateNode={}, 
                double stopCondition = -1
                ) const;

    template <typename dataType, typename triangulationType>
    int executeMultiBlock(dataType *const inputScalars,
                dataType *const outputScalars,
                SimplexId *const inputOffsets,
                triangulationType *triangulation,
                ttk::DiagramType &constraintDiagram,
                int *const modificationNumber,
                int *const lastChange,
                int *const idBlock, 
                double &stopCondition, 
                SimplexId numberPairsInputDiagram
                ) const;
                
    inline int preconditionTriangulation(AbstractTriangulation *triangulation) {
      if(triangulation) {
        vertexNumber_ = triangulation->getNumberOfVertices();
        triangulation->preconditionVertexNeighbors();
      }
      return 0;
    }

    template <typename dataType, typename triangulationType>
    std::tuple<std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >,
    std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >>
      getIndices(triangulationType * triangulation, 
                SimplexId* & inputOffsets,
                dataType* const inputScalars,
                ttk::DiagramType& constraintDiagram, 
                SimplexId vertexNumber,
                int epoch,
                std::vector<int64_t>  &listAllIndicesToChange, 
                std::vector<std::vector<SimplexId>> &pair2MatchedPair, 
                std::vector<std::vector<SimplexId>> &pair2Delete,
                std::vector<SimplexId> &pairChangeMatchingPair,
                SimplexId &numberPairsInputDiagram,
                std::vector<SimplexId> &pairOnEdgeToRemove,
                bool lastBlock=false, 
                std::vector<float> &listTimePersistenceDiagram={}, 
                std::vector<float> &listTimePersistenceDiagramClustering={},
                std::vector<float> &listAveragePercentageOfImmobilePersistencePairs={},
                std::vector<std::vector<SimplexId>> &currentVertex2PairsCurrentDiagram={}, 
                int threadNumber=-1,
                std::vector<SimplexId> &localToGlobal={}, 
                std::vector<float> maxAndMinCoordinateNode={}, 
                ttk::Triangulation * triangulationOneBlock=nullptr 
                ) const;

    template <typename dataType, typename triangulationType>
    int getStopCondition(triangulationType * triangulation, 
                            SimplexId *const inputOffsets,
                            dataType *const inputScalars,
                            ttk::DiagramType& constraintDiagram, 
                            double & stopCondition, 
                            SimplexId &numberPairsInputDiagram
                          ) const;

    template <typename triangulationType>
      int getNeighborsIndices(const triangulationType& triangulation, 
                            const int64_t& i, 
                            std::vector<int64_t>& neighborsIndices
                            ) const;

      int tensorToVectorFast(const torch::Tensor& tensor, 
                            std::vector<double>& result
                            ) const ; 

    bool isOn2DifferentFaces(std::vector<float> coordinatePointA, std::vector<float> coordinatePointB) const; 
    bool isPointInExtrema(std::vector<float> maxAndMinCoordinateNode, std::vector<float> coordinatePoint) const; 
    

      std::vector<std::vector<double>> getCoordinatesInformations(std::vector<float> coordinatesVertices) const; 

  protected:
    SimplexId vertexNumber_{};

    bool UseLazyDiscretGradientUpdate{false};

    bool UseTheMultiBlocksApproach{false};
    double NumberOfBlocksPerThread{1.0};
    int NumberEpochMultiBlock{1};
    bool MultiBlockOneLevel{false};

    bool UseTimeThreshold{false};
    double TimeThreshold{0.01};

    int EpochNumber{1000};

    bool UseTopologicalSimplification{false};

    bool SemiDirectMatching{false}; 


    // if PDCMethod == 0 then we use Progressive approach 
    // if PDCMethod == 1 then we use Classical Auction approach
    int PDCMethod{0}; 

    //
    bool UseAdditionalPrecisionPDC{false}; 
    double DeltaLim{0.01}; 

    // if Method == 0 then we use direct optimization  
    // if Method == 1 then we use Adam  
    int Method{0}; 

    // if FinePairManagement == 0 then we let the algorithm choose
    // if FinePairManagement == 1 then we fill the domain
    // if FinePairManagement == 2 then we cut the domain
    int FinePairManagement{0}; 

    // Adam 
    bool ChooseLearningRate{false};
    double LearningRate{0.0001}; 

    // Direct Optimization 
    double EpsilonPenalisation{0.75}; 

    // Stopping criterion: when the loss becomes less than a percentage (e.g. 1%) of the original loss (between input diagram and simplified diagram)
    double CoefStopCondition{0.01}; 

    // SaveData == true :  to save execution time measurements
    bool SaveData{false}; 
  };

} // namespace ttk

class PersistenceDiagramGradientDescent : public torch::nn::Module, public ttk::TopologicalOptimization{
    public:
      PersistenceDiagramGradientDescent(torch::Tensor X_tensor) : torch::nn::Module(){
        X = register_parameter("X", X_tensor, true);
      }
      torch::Tensor X;
};


/*
  Find all neighbors of a vertex i.
  Variable : 
    -   triangulation : domain triangulation
    -   i : vertex for which we want to find his neighbors
    -   neighborsIndices : vector which contains the neighboring vertices of vertex i
*/
template <typename triangulationType>
int ttk::TopologicalOptimization::getNeighborsIndices(
  const triangulationType& triangulation,
  const int64_t& i,
  std::vector<int64_t>& neighborsIndices
  ) const {

  size_t nNeighbors = triangulation->getVertexNeighborNumber(i);
  ttk::SimplexId neighborId{-1};
  for(size_t j = 0; j < nNeighbors; j++) {
    triangulation->getVertexNeighbor(static_cast<SimplexId>(i), j, neighborId);
    neighborsIndices.push_back(static_cast<int64_t>(neighborId));
  }

  return 0;
}

template <typename dataType, typename triangulationType>
int ttk::TopologicalOptimization::getStopCondition(
  triangulationType * triangulation, 
  SimplexId *const inputOffsets,
  dataType *const inputScalars,
  ttk::DiagramType& constraintDiagram, 
  double & stopCondition, 
  SimplexId &numberPairsInputDiagram
  ) const {

  ttk::PersistenceDiagram diagram;
  std::vector<ttk::PersistencePair> diagramOutput;
  ttk::preconditionOrderArray<dataType>(vertexNumber_, inputScalars, inputOffsets, threadNumber_);
  diagram.setDebugLevel(debugLevel_);
  diagram.setThreadNumber(threadNumber_);
  diagram.preconditionTriangulation(triangulation);

  diagram.execute(
    diagramOutput, inputScalars, -1, inputOffsets, triangulation);
  
  numberPairsInputDiagram = (SimplexId)diagramOutput.size();

  std::vector<std::vector<SimplexId>> vertex2PairsCurrentDiagram(vertexNumber_, std::vector<SimplexId>()); 
  for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
    auto pair = diagramOutput[i]; 
    vertex2PairsCurrentDiagram[pair.birth.id].push_back(i);  
    vertex2PairsCurrentDiagram[pair.death.id].push_back(i);  
  }

  std::vector<std::vector<SimplexId>> vertex2PairsTargetDiagram(vertexNumber_, std::vector<SimplexId>()); 
  for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
    auto pair = constraintDiagram[i]; 
    vertex2PairsTargetDiagram[pair.birth.id].push_back(i);  
    vertex2PairsTargetDiagram[pair.death.id].push_back(i);  
  }

  std::vector<SimplexId> matchingPairCurrentDiagram((SimplexId)diagramOutput.size(), -1); 
  std::vector<SimplexId> matchingPairTargetDiagram((SimplexId)constraintDiagram.size(), -1); 

  // pairs to change
  std::vector<int64_t> birthPairToChangeCurrentDiagram{};
  std::vector<double> birthPairToChangeTargetDiagram{};
  std::vector<int64_t> deathPairToChangeCurrentDiagram{};
  std::vector<double> deathPairToChangeTargetDiagram{};

  // pairs to delete
  std::vector<int64_t> birthPairToDeleteCurrentDiagram{};
  std::vector<double> birthPairToDeleteTargetDiagram{};
  std::vector<int64_t> deathPairToDeleteCurrentDiagram{};
  std::vector<double> deathPairToDeleteTargetDiagram{};

  for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
    auto pair = constraintDiagram[i]; 

    SimplexId birthId = pair.birth.id; 
    SimplexId deathId = pair.death.id; 

    for(auto &idPairBirth : vertex2PairsCurrentDiagram[birthId]){
      for(auto &idPairDeath : vertex2PairsCurrentDiagram[deathId]){
        if(idPairBirth == idPairDeath){
          matchingPairCurrentDiagram[idPairBirth] = 1;
          matchingPairTargetDiagram[i] = 1;
        }
      }
    }
  }

  ttk::DiagramType thresholdCurrentDiagram{}; 
  for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
    auto pair = diagramOutput[i]; 
    if(matchingPairCurrentDiagram[i] == -1){
      thresholdCurrentDiagram.push_back(pair); 
    }
  }

  for(SimplexId i = 0; i < (SimplexId)thresholdCurrentDiagram.size(); i++){
    auto pair = thresholdCurrentDiagram[i]; 

    if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) && (vertex2PairsTargetDiagram[pair.death.id].size() == 0)){
      deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
      deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
      continue; 
    }

    if((vertex2PairsTargetDiagram[pair.birth.id].size() == 0) && (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
      birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
      birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
      continue; 
    }

    if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) || (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
      continue; 
    }

    birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
    birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
    deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
    deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
  }

  // Loss : delete pairs
  double lossDeletePairs = 0; 
  for(size_t i = 0; i <  birthPairToDeleteCurrentDiagram.size(); i++) {
    lossDeletePairs += std::pow(inputScalars[birthPairToDeleteCurrentDiagram[i]] - birthPairToDeleteTargetDiagram[i], 2); 
  }

  for(size_t i = 0; i <  deathPairToDeleteCurrentDiagram.size(); i++) {
    lossDeletePairs += std::pow(inputScalars[deathPairToDeleteCurrentDiagram[i]] - deathPairToDeleteTargetDiagram[i], 2); 
  }
  
  this->printMsg(
  "GetStopCondition | loss Delete Pairs = " + std::to_string(lossDeletePairs), debug::Priority::DETAIL);

  // Calcul stop condition 
  stopCondition = CoefStopCondition * lossDeletePairs; 
  
  this->printMsg(
  "GetStopCondition | StopCondition = " + std::to_string(stopCondition), debug::Priority::PERFORMANCE);

  return 0; 
}
                      
bool ttk::TopologicalOptimization::isOn2DifferentFaces(std::vector<float> coordinatePointA, std::vector<float> coordinatePointB) const {
  bool resultat = true; 

  float xA = coordinatePointA[0]; 
  float yA = coordinatePointA[1]; 
  float zA = coordinatePointA[2];

  float xB = coordinatePointB[0]; 
  float yB = coordinatePointB[1]; 
  float zB = coordinatePointB[2];

  if((xA <= (xB*(1+1e-6))) && (xA >= xB*(1-1e-6))){
    resultat = false; 
  }
  else if((yA <= (yB*(1+1e-6))) && (yA >= yB*(1-1e-6))){
    resultat = false; 
  }
  else if((zA <= (zB*(1+1e-6))) && (zA >= zB*(1-1e-6))){
    resultat = false; 
  }

  return resultat; 
}

bool ttk::TopologicalOptimization::isPointInExtrema(std::vector<float> maxAndMinCoordinateNode, std::vector<float> coordinatePoint) const {
  bool resultat = false; 

  float x_min = maxAndMinCoordinateNode[0]; 
  float y_min = maxAndMinCoordinateNode[1]; 
  float z_min = maxAndMinCoordinateNode[2];

  float x_max = maxAndMinCoordinateNode[3]; 
  float y_max = maxAndMinCoordinateNode[4]; 
  float z_max = maxAndMinCoordinateNode[5];  

  float x = coordinatePoint[0]; 
  float y = coordinatePoint[1]; 
  float z = coordinatePoint[2];


  // if x == xmin
  if((x <= (x_min*(1+1e-6))) && (x >= x_min*(1-1e-6))){
    resultat = true; 
  }
  // if x == xmax
  else if((x <= (x_max*(1+1e-6))) && (x >= x_max*(1-1e-6))){
    resultat = true; 
  }
  // if y == ymin
  else if((y <= (y_min*(1+1e-6))) && (y >= y_min*(1-1e-6))){
    resultat = true; 
  }
  // if y == ymax
  else if((y <= (y_max*(1+1e-6))) && (y >= y_max*(1-1e-6))){
    resultat = true; 
  }
  // if z == zmin
  else if((z <= (z_min*(1+1e-6))) && (z >= z_min*(1-1e-6))){
    resultat = true; 
  }
  // if z == zmax
  else if((z <= (z_max*(1+1e-6))) && (z >= z_max*(1-1e-6))){
    resultat = true; 
  }

  return resultat; 
}

/*
  This function allows us to retrieve the indices of the critical points
  that we must modify in order to match our current diagram to our target 
  diagram.
*/
template <typename dataType, typename triangulationType>
std::tuple<std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >,
std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >>
  ttk::TopologicalOptimization::getIndices(
    triangulationType * triangulation,
    SimplexId* & inputOffsets, 
    dataType* const inputScalars,
    ttk::DiagramType &constraintDiagram, 
    SimplexId vertexNumber, 
    int epoch, 
    std::vector<int64_t> & listAllIndicesToChange,  
    std::vector<std::vector<SimplexId>> &pair2MatchedPair,
    std::vector<std::vector<SimplexId>> &pair2Delete,
    std::vector<SimplexId> &pairChangeMatchingPair,
    SimplexId &numberPairsInputDiagram,
    std::vector<SimplexId> &pairOnEdgeToRemove,
    bool lastBlock, 
    std::vector<float> &listTimePersistenceDiagram, 
    std::vector<float> &listTimePersistenceDiagramClustering,
    std::vector<float> &listAveragePercentageOfImmobilePersistencePairs,
    std::vector<std::vector<SimplexId>> &currentVertex2PairsCurrentDiagram, 
    int threadNumber, 
    std::vector<SimplexId> &localToGlobal, 
    std::vector<float> maxAndMinCoordinateNode, 
    ttk::Triangulation * triangulationOneBlock
    ) const {

  //=========================================
  //            Lazy Gradient 
  //=========================================

  bool needUpdateDefaultValue
    = (UseLazyDiscretGradientUpdate ? (epoch == 0 || epoch < 0 ? true : false)
                                    : true);
  std::vector<bool> needUpdate(vertexNumber, needUpdateDefaultValue);
  if(UseLazyDiscretGradientUpdate) {
    /*
      There is a 10% loss of performance
    */
    this->printMsg(
      "Get Indices | UseLazyDiscretGradientUpdate", debug::Priority::DETAIL);

    if(not(epoch == 0 || epoch < 0)) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber)
#endif
      for(size_t indice = 0; indice < listAllIndicesToChange.size(); indice++) {
        if(listAllIndicesToChange[indice] == 1) {
          needUpdate[indice] = true;
          // Find all the neighbors of the vertex
          std::vector<int64_t> neighborsIndices;
          if(UseTheMultiBlocksApproach && !(lastBlock)){
            getNeighborsIndices(triangulationOneBlock, indice, neighborsIndices);
          }
          else{
            getNeighborsIndices(triangulation, indice, neighborsIndices);
          }
          for (int64_t neighborsIndice : neighborsIndices){
            needUpdate[neighborsIndice] = true;
          }
        }
      }
    }
  }

  SimplexId count = std::count(needUpdate.begin(), needUpdate.end(), true);
  this->printMsg(
    "Get Indices | The number of elements that need to be updated is: " + std::to_string(count),  debug::Priority::DETAIL);


  //=========================================
  //     Compute the persistence diagram
  //=========================================
  ttk::Timer timePersistenceDiagram;

  ttk::PersistenceDiagram diagram;
  std::vector<ttk::PersistencePair> diagramOutput;
  ttk::preconditionOrderArray<dataType>(vertexNumber, inputScalars, inputOffsets, threadNumber);
  diagram.setDebugLevel(debugLevel_);
  diagram.setThreadNumber(threadNumber);

  if(UseTheMultiBlocksApproach && !(lastBlock)){
    diagram.preconditionTriangulation(triangulationOneBlock);
  }
  else{
    diagram.preconditionTriangulation(triangulation);
  }

  if (UseLazyDiscretGradientUpdate){
    if(UseTheMultiBlocksApproach && !(lastBlock)){
      diagram.execute(
        diagramOutput, inputScalars, 0, inputOffsets, triangulationOneBlock, &needUpdate);
    }
    else{
      diagram.execute(
        diagramOutput, inputScalars, 0, inputOffsets, triangulation, &needUpdate);
    }
  }
  else{
    if(UseTheMultiBlocksApproach && !(lastBlock)){
      diagram.execute(
        diagramOutput, inputScalars, epoch, inputOffsets, triangulationOneBlock);
    }
    else{
      diagram.execute(
        diagramOutput, inputScalars, epoch, inputOffsets, triangulation);
    }
  }

  if((epoch == 0) && !(UseTheMultiBlocksApproach)){
    numberPairsInputDiagram = (SimplexId)diagramOutput.size();
    this->printMsg(
      "Get Indices | Number Pairs Input Diagram : " + std::to_string(numberPairsInputDiagram) ,  debug::Priority::DETAIL);
  }

  listTimePersistenceDiagram.push_back(timePersistenceDiagram.getElapsedTime());

  //=====================================
  //          Matching Pairs             
  //=====================================

  // pairs to change
  std::vector<int64_t> birthPairToChangeCurrentDiagram{};
  std::vector<double> birthPairToChangeTargetDiagram{};
  std::vector<int64_t> deathPairToChangeCurrentDiagram{};
  std::vector<double> deathPairToChangeTargetDiagram{};

  // pairs to delete
  std::vector<int64_t> birthPairToDeleteCurrentDiagram{};
  std::vector<double> birthPairToDeleteTargetDiagram{};
  std::vector<int64_t> deathPairToDeleteCurrentDiagram{};
  std::vector<double> deathPairToDeleteTargetDiagram{};
 
  if(UseTheMultiBlocksApproach && !(lastBlock)){

    std::vector<std::vector<SimplexId>> criticalVertex2PairTarget(vertexNumber_, std::vector<SimplexId>()); 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto pair = constraintDiagram[i]; 
      criticalVertex2PairTarget[pair.birth.id].push_back(i); 
      criticalVertex2PairTarget[pair.death.id].push_back(i); 
    }

    std::vector<std::vector<SimplexId>> cricalVertex2PairCurrentData(vertexNumber_, std::vector<SimplexId>()); 
    for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i ++){
      auto pair = diagramOutput[i]; 
      cricalVertex2PairCurrentData[localToGlobal[pair.birth.id]].push_back(i); 
      cricalVertex2PairCurrentData[localToGlobal[pair.death.id]].push_back(i); 
    }

    std::vector<std::vector<SimplexId>> matchedPairs; 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto &pair = constraintDiagram[i]; 

      SimplexId birthId = -1; 
      SimplexId deathId = -1;  

      if(pairChangeMatchingPair[i] == 1){
        birthId = pair2MatchedPair[i][0]; 
        deathId = pair2MatchedPair[i][1]; 
      }
      else{
        birthId = pair.birth.id; 
        deathId = pair.death.id; 
      }

      if((cricalVertex2PairCurrentData[birthId].size() == 1) && (cricalVertex2PairCurrentData[deathId].size() == 1)){
        if(cricalVertex2PairCurrentData[birthId][0] == cricalVertex2PairCurrentData[deathId][0]){
          matchedPairs.push_back({i, cricalVertex2PairCurrentData[deathId][0]}); 
        }
      }
    }

    std::vector<SimplexId> matchingPairCurrentDiagram((SimplexId)diagramOutput.size(), -1); 
    std::vector<SimplexId> matchingPairTargetDiagram((SimplexId)constraintDiagram.size(), -1); 

    for(auto &match : matchedPairs){
      auto &indicePairTargetDiagram = match[0]; 
      auto &indicePairCurrentDiagram = match[1]; 

      auto &pairCurrentDiagram = diagramOutput[indicePairCurrentDiagram]; 
      // auto &pairTargetDiagram = constraintDiagram[indicePairTargetDiagram]; 

      pair2MatchedPair[indicePairTargetDiagram][0] = localToGlobal[pairCurrentDiagram.birth.id]; 
      pair2MatchedPair[indicePairTargetDiagram][1] = localToGlobal[pairCurrentDiagram.death.id]; 

      matchingPairCurrentDiagram[indicePairCurrentDiagram] = 1;
      matchingPairTargetDiagram[indicePairTargetDiagram] = 1;
    }

    ttk::DiagramType thresholdCurrentDiagram{}; 
    for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
      auto &pair = diagramOutput[i]; 
      
      // We check if it is not a noise detected at the beginning 
      if((pairOnEdgeToRemove[localToGlobal[pair.birth.id]] == pairOnEdgeToRemove[localToGlobal[pair.death.id]])
        && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != -1)){

        // pair preserved because the two points are on target pairs
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)){
          continue; 
        }
        // The multiple one: a critical point of the pair is in a target pair and a noise pair 
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() == 0)){
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
          deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }
        // The multiple one: a critical point of the pair is in a target pair and a noise pair
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)){
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
          birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }
        
        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        continue; 
      }

      if((pair2Delete[pair.birth.id].size() == 1) && (pair2Delete[pair.death.id].size() == 1) && (pair2Delete[pair.birth.id] == pair2Delete[pair.death.id])){
        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        continue; 
      }      
      if(matchingPairCurrentDiagram[i] == -1){
        thresholdCurrentDiagram.push_back(pair); 
      }
    }

    ttk::DiagramType thresholdConstraintDiagram{}; 
    std::vector<SimplexId> pairIndiceLocal2Global{}; 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto &pair = constraintDiagram[i]; 
      
      if(matchingPairTargetDiagram[i] == -1){
        thresholdConstraintDiagram.push_back(pair); 
        pairIndiceLocal2Global.push_back(i); 
      }
    }


    if(thresholdConstraintDiagram.size() == 0){
      for(SimplexId i = 0; i < (SimplexId)thresholdCurrentDiagram.size(); i++){
        auto &pair = thresholdCurrentDiagram[i]; 

        if((triangulationOneBlock->isVertexOnBoundary(pair.birth.id)) && (triangulationOneBlock->isVertexOnBoundary(pair.death.id))){
          float x_birth = 0.0, y_birth = 0.0, z_birth = 0.0;
          triangulation->getVertexPoint(localToGlobal[pair.birth.id], x_birth, y_birth, z_birth);
          
          float x_death = 0.0, y_death = 0.0, z_death = 0.0;
          triangulation->getVertexPoint(localToGlobal[pair.death.id], x_death, y_death, z_death);

          // If they are on the same side
          std::vector<float> coordinatePointA{x_birth, y_birth, z_birth}; 
          std::vector<float> coordinatePointB{x_death, y_death, z_death}; 
            
          if(isOn2DifferentFaces(coordinatePointA, coordinatePointB)){
            continue; 
          }

          // If one of the points is on the extrema
          if(isPointInExtrema(maxAndMinCoordinateNode, coordinatePointA) || isPointInExtrema(maxAndMinCoordinateNode, coordinatePointB)){
            continue; 
          }
        }
        
        // Otherwise if only one point is on the edge
        if((triangulationOneBlock->isVertexOnBoundary(pair.birth.id)) || (triangulationOneBlock->isVertexOnBoundary(pair.death.id))){
          continue; 
        }

        // pair preserved because the two points are on target pairs
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)){
          continue; 
        }

        // 
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() == 0)
          && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] == -1) && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != pairOnEdgeToRemove[localToGlobal[pair.death.id]])){
            continue; 
        }

        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)
          && (pairOnEdgeToRemove[localToGlobal[pair.death.id]] == -1) && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != pairOnEdgeToRemove[localToGlobal[pair.death.id]])){
            continue; 
        }
        
        if((pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != -1) && (pairOnEdgeToRemove[localToGlobal[pair.death.id]] != -1) && 
          (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != pairOnEdgeToRemove[localToGlobal[pair.death.id]])){
          continue; 
        }

        // The multiple one: a critical point of the pair is in a target pair and a noise pair
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() == 0)){
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
          deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }
        // The multiple one: a critical point of the pair is in a target pair and a noise pair
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)){
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
          birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }
        
        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);

        pair2Delete[pair.birth.id].push_back(pair.birth.id);
        pair2Delete[pair.death.id].push_back(pair.birth.id);
        
      }
    }
    else {
      

      ttk::PersistenceDiagramClustering persistenceDiagramClustering;
      PersistenceDiagramBarycenter pdBarycenter{};
      std::vector<ttk::DiagramType> intermediateDiagrams{thresholdConstraintDiagram, thresholdCurrentDiagram};
      std::vector<ttk::DiagramType> centroids;
      std::vector<std::vector<std::vector<ttk::MatchingType>>> allMatchings;


      if(PDCMethod == 0){
        persistenceDiagramClustering.setDebugLevel(debugLevel_);
        persistenceDiagramClustering.setThreadNumber(threadNumber);
        // SetForceUseOfAlgorithm ==> Force the progressive approch if 2 inputs
        persistenceDiagramClustering.setForceUseOfAlgorithm(false);
        // setDeterministic ==> Deterministic algorithm
        persistenceDiagramClustering.setDeterministic(true);
        // setUseProgressive ==> Compute Progressive Barycenter
        persistenceDiagramClustering.setUseProgressive(true);
        // setUseInterruptible ==> Interruptible algorithm
        // persistenceDiagramClustering.setUseInterruptible(true);
        persistenceDiagramClustering.setUseInterruptible(UseTimeThreshold);
        // // setTimeLimit ==> Maximal computation time (s)
        persistenceDiagramClustering.setTimeLimit(TimeThreshold);
        // setUseAdditionalPrecision ==> Force minimum precision on matchings
        persistenceDiagramClustering.setUseAdditionalPrecision(true);
        // setDeltaLim ==> Minimal relative precision
        persistenceDiagramClustering.setDeltaLim(DeltaLim);
        // setUseAccelerated ==> Use Accelerated KMeans
        persistenceDiagramClustering.setUseAccelerated(false);
        // setUseKmeansppInit ==> KMeanspp Initialization
        persistenceDiagramClustering.setUseKmeansppInit(false);
      
        std::vector<int> clusterIds = persistenceDiagramClustering.execute(
          intermediateDiagrams, centroids, allMatchings);
      }
      else{
        centroids.resize(1);
        const auto wassersteinMetric = std::to_string(2);
        pdBarycenter.setWasserstein(wassersteinMetric);
        pdBarycenter.setMethod(2);
        pdBarycenter.setNumberOfInputs(2);
        pdBarycenter.setDeterministic(1);
        pdBarycenter.setUseProgressive(1);
        pdBarycenter.setDebugLevel(debugLevel_);
        pdBarycenter.setThreadNumber(threadNumber);
        pdBarycenter.setAlpha(1);
        pdBarycenter.setLambda(1);
        if(UseAdditionalPrecisionPDC){
          pdBarycenter.setDeltaLim(DeltaLim); 
        }
        pdBarycenter.execute(
          intermediateDiagrams, centroids[0], allMatchings);
      }
    
      //=========================================
      //             Find matched pairs
      //=========================================

      std::vector<std::vector<SimplexId>> allPairsSelected{};
      std::vector<std::vector<SimplexId>> matchingsBlockPairs(centroids[0].size());

      for(auto i = 1; i >= 0; --i){
        std::vector<ttk::MatchingType> &matching = allMatchings[0][i];

        const auto &diag{intermediateDiagrams[i]};

        for(SimplexId j = 0; j < (SimplexId)matching.size(); j++){

          const auto &m{matching[j]};
          const auto &bidderId{std::get<0>(m)};
          const auto &goodId{std::get<1>(m)};

          if((goodId == -1) | (bidderId == -1))
            continue;

          if (diag[bidderId].persistence() != 0){
            if(i == 1){
              matchingsBlockPairs[goodId].push_back(bidderId);
            }
            else if (matchingsBlockPairs[goodId].size() > 0){
              matchingsBlockPairs[goodId].push_back(bidderId);
            }
            allPairsSelected.push_back({diag[bidderId].birth.id, diag[bidderId].death.id});
          }
        }
      }

      std::vector<ttk::PersistencePair> pairsToErase{};

      std::map<std::vector<SimplexId>, SimplexId> currentToTarget;
      for(auto &pair : allPairsSelected){
        currentToTarget[{pair[0], pair[1]}] = 1;
      }

      for(auto &pair : intermediateDiagrams[1]){
        if(pair.isFinite != 0){
          if(!(currentToTarget.count({pair.birth.id, pair.death.id }) > 0)){
            pairsToErase.push_back(pair);
          }
        }
      }

      for(auto &pair : pairsToErase){

        if((triangulationOneBlock->isVertexOnBoundary(pair.birth.id)) && (triangulationOneBlock->isVertexOnBoundary(pair.death.id))){
          float x_birth = 0.0, y_birth = 0.0, z_birth = 0.0;
          triangulation->getVertexPoint(localToGlobal[pair.birth.id], x_birth, y_birth, z_birth);
          
          float x_death = 0.0, y_death = 0.0, z_death = 0.0;
          triangulation->getVertexPoint(localToGlobal[pair.death.id], x_death, y_death, z_death);

          std::vector<float> coordinatePointA{x_birth, y_birth, z_birth}; 
          std::vector<float> coordinatePointB{x_death, y_death, z_death}; 
            
          if(isOn2DifferentFaces(coordinatePointA, coordinatePointB)){
            continue; 
          }

          if(isPointInExtrema(maxAndMinCoordinateNode, coordinatePointA) || isPointInExtrema(maxAndMinCoordinateNode, coordinatePointB)){
            continue; 
          }
        }
        
        if((triangulationOneBlock->isVertexOnBoundary(pair.birth.id)) || (triangulationOneBlock->isVertexOnBoundary(pair.death.id))){
          continue; 
        }
        
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)){
          continue; 
        }

        // 
        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() == 0)
          && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] == -1) && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != pairOnEdgeToRemove[localToGlobal[pair.death.id]])){
            continue; 
        }

        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)
          && (pairOnEdgeToRemove[localToGlobal[pair.death.id]] == -1) && (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != pairOnEdgeToRemove[localToGlobal[pair.death.id]])){
            continue; 
        }

        if((pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != -1) && (pairOnEdgeToRemove[localToGlobal[pair.death.id]] != -1) && 
          (pairOnEdgeToRemove[localToGlobal[pair.birth.id]] != pairOnEdgeToRemove[localToGlobal[pair.death.id]])){
          continue; 
        }

        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() == 0)){
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
          deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }

        if((criticalVertex2PairTarget[localToGlobal[pair.birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[pair.death.id]].size() >= 1)){
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
          birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }
        
        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);

        pair2Delete[pair.birth.id].push_back(pair.birth.id);
        pair2Delete[pair.death.id].push_back(pair.birth.id);
      }

      for(const auto& entry : matchingsBlockPairs){
        // Delete pairs that have no matching
        if(entry.size() == 1){

          if((triangulationOneBlock->isVertexOnBoundary(thresholdCurrentDiagram[entry[0]].birth.id)) && (triangulationOneBlock->isVertexOnBoundary(thresholdCurrentDiagram[entry[0]].death.id))){
            float x_birth = 0.0, y_birth = 0.0, z_birth = 0.0;
            triangulation->getVertexPoint(localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id], x_birth, y_birth, z_birth);
            
            float x_death = 0.0, y_death = 0.0, z_death = 0.0;
            triangulation->getVertexPoint(localToGlobal[thresholdCurrentDiagram[entry[0]].death.id], x_death, y_death, z_death);

            std::vector<float> coordinatePointA{x_birth, y_birth, z_birth}; 
            std::vector<float> coordinatePointB{x_death, y_death, z_death}; 

            if(isOn2DifferentFaces(coordinatePointA, coordinatePointB)){
              continue; 
            }

            if(isPointInExtrema(maxAndMinCoordinateNode, coordinatePointA) || isPointInExtrema(maxAndMinCoordinateNode, coordinatePointB)){
              continue; 
            }
          }
          
          if((triangulationOneBlock->isVertexOnBoundary(thresholdCurrentDiagram[entry[0]].birth.id)) || (triangulationOneBlock->isVertexOnBoundary(thresholdCurrentDiagram[entry[0]].death.id))){
            continue; 
          }

          if((criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]].size() >= 1)){
            continue; 
          }

          // 
          if((criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]].size() == 0)
            && (pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]] == -1) && (pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]] != pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]])){
              continue; 
          }

          if((criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]].size() >= 1)
            && (pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]] == -1) && (pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]] != pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]])){
              continue; 
          }
          
          if((pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]] != -1) && (pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]] != -1) && 
            (pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]] != pairOnEdgeToRemove[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]])){
            continue; 
          }

          if((criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]].size() >= 1) && (criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]].size() == 0)){
            deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].death.id));
            deathPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
            continue; 
          }
          if((criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id]].size() == 0) && (criticalVertex2PairTarget[localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]].size() >= 1)){
            birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].birth.id));
            birthPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
            continue; 
          }
            
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].birth.id));
          birthPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].death.id));
          deathPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);

          pair2Delete[thresholdCurrentDiagram[entry[0]].birth.id].push_back(thresholdCurrentDiagram[entry[0]].birth.id);
          pair2Delete[thresholdCurrentDiagram[entry[0]].death.id].push_back(thresholdCurrentDiagram[entry[0]].birth.id);

          continue;
        } else if(entry.empty())
          continue;

        pair2MatchedPair[pairIndiceLocal2Global[entry[1]]][0] = localToGlobal[thresholdCurrentDiagram[entry[0]].birth.id];
        pair2MatchedPair[pairIndiceLocal2Global[entry[1]]][1] = localToGlobal[thresholdCurrentDiagram[entry[0]].death.id]; 

        pairChangeMatchingPair[pairIndiceLocal2Global[entry[1]]] = 1; 
       
      }
    }
  } 

  else if(SemiDirectMatching || lastBlock){
    
    // (!(UseTheMultiBlocksApproach)) || (lastBlock)
    std::vector<std::vector<SimplexId>> vertex2PairsCurrentDiagram(vertexNumber, std::vector<SimplexId>()); 
    for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
      auto &pair = diagramOutput[i]; 
      vertex2PairsCurrentDiagram[pair.birth.id].push_back(i);  
      vertex2PairsCurrentDiagram[pair.death.id].push_back(i);  
    }

    std::vector<std::vector<SimplexId>> vertex2PairsTargetDiagram(vertexNumber, std::vector<SimplexId>()); 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto &pair = constraintDiagram[i]; 
      vertex2PairsTargetDiagram[pair.birth.id].push_back(i);  
      vertex2PairsTargetDiagram[pair.death.id].push_back(i);  
    }

    std::vector<std::vector<SimplexId>> matchedPairs; 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto &pair = constraintDiagram[i]; 

      SimplexId birthId = -1; 
      SimplexId deathId = -1;  

      if(pairChangeMatchingPair[i] == 1){
        birthId = pair2MatchedPair[i][0]; 
        deathId = pair2MatchedPair[i][1]; 
      }
      else{
        birthId = pair.birth.id; 
        deathId = pair.death.id; 
      }

      if((epoch == 0) && !(UseTheMultiBlocksApproach)){
        for(auto &idPairBirth : vertex2PairsCurrentDiagram[birthId]){
          for(auto &idPairDeath : vertex2PairsCurrentDiagram[deathId]){
            if(idPairBirth == idPairDeath){
              matchedPairs.push_back({i, idPairBirth}); 
            }
          }
        }
      }
      else if((vertex2PairsCurrentDiagram[birthId].size() == 1) && (vertex2PairsCurrentDiagram[deathId].size() == 1)){
        if(vertex2PairsCurrentDiagram[birthId][0] == vertex2PairsCurrentDiagram[deathId][0]){
          matchedPairs.push_back({i, vertex2PairsCurrentDiagram[deathId][0]}); 
        }
      }
    }

    std::vector<SimplexId> matchingPairCurrentDiagram((SimplexId)diagramOutput.size(), -1); 
    std::vector<SimplexId> matchingPairTargetDiagram((SimplexId)constraintDiagram.size(), -1); 

    for(auto &match : matchedPairs){
      auto &indicePairTargetDiagram = match[0]; 
      auto &indicePairCurrentDiagram = match[1]; 

      auto &pairCurrentDiagram = diagramOutput[indicePairCurrentDiagram]; 
      auto &pairTargetDiagram = constraintDiagram[indicePairTargetDiagram]; 

      pair2MatchedPair[indicePairTargetDiagram][0] = pairCurrentDiagram.birth.id; 
      pair2MatchedPair[indicePairTargetDiagram][1] = pairCurrentDiagram.death.id; 

      matchingPairCurrentDiagram[indicePairCurrentDiagram] = 1;
      matchingPairTargetDiagram[indicePairTargetDiagram] = 1;

      int64_t valueBirthPairToChangeCurrentDiagram = (int64_t)(pairCurrentDiagram.birth.id);
      int64_t valueDeathPairToChangeCurrentDiagram = (int64_t)(pairCurrentDiagram.death.id);

      double valueBirthPairToChangeTargetDiagram = pairTargetDiagram.birth.sfValue;
      double valueDeathPairToChangeTargetDiagram = pairTargetDiagram.death.sfValue;

      birthPairToChangeCurrentDiagram.push_back(valueBirthPairToChangeCurrentDiagram);
      birthPairToChangeTargetDiagram.push_back(valueBirthPairToChangeTargetDiagram);
      deathPairToChangeCurrentDiagram.push_back(valueDeathPairToChangeCurrentDiagram);
      deathPairToChangeTargetDiagram.push_back(valueDeathPairToChangeTargetDiagram);

    }

    ttk::DiagramType thresholdCurrentDiagram{}; 
    for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
      auto &pair = diagramOutput[i]; 

      if((pair2Delete[pair.birth.id].size() == 1) && (pair2Delete[pair.death.id].size() == 1) && (pair2Delete[pair.birth.id] == pair2Delete[pair.death.id])){

        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        continue; 
      }      
      if(matchingPairCurrentDiagram[i] == -1){
        thresholdCurrentDiagram.push_back(pair); 
      }
    }

    ttk::DiagramType thresholdConstraintDiagram{}; 
    std::vector<SimplexId> pairIndiceLocal2Global{}; 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto &pair = constraintDiagram[i]; 
      
      if(matchingPairTargetDiagram[i] == -1){
        thresholdConstraintDiagram.push_back(pair); 
        pairIndiceLocal2Global.push_back(i); 
      }
    }
    
    this->printMsg(
    "Get Indices | thresholdCurrentDiagram.size() : "  + std::to_string(thresholdCurrentDiagram.size()), debug::Priority::DETAIL);
    
    this->printMsg(
    "Get Indices | thresholdConstraintDiagram.size() : " + std::to_string(thresholdConstraintDiagram.size()), debug::Priority::DETAIL);

    // Average Percentage Of Immobile Persistence Pairs
    listAveragePercentageOfImmobilePersistencePairs.push_back((double)(diagramOutput.size() - thresholdCurrentDiagram.size()) / diagramOutput.size()); 

    if(thresholdConstraintDiagram.size() == 0){
      for(SimplexId i = 0; i < (SimplexId)thresholdCurrentDiagram.size(); i++){
        auto &pair = thresholdCurrentDiagram[i]; 

        if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) && (vertex2PairsTargetDiagram[pair.death.id].size() == 0)){
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
          deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }

        if((vertex2PairsTargetDiagram[pair.birth.id].size() == 0) && (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
          birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }

        if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) || (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
          continue; 
        }

        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);

        pair2Delete[pair.birth.id].push_back(i);
        pair2Delete[pair.death.id].push_back(i);
      }
    }
    else{

      ttk::Timer timePersistenceDiagramClustering;

      ttk::PersistenceDiagramClustering persistenceDiagramClustering;
      PersistenceDiagramBarycenter pdBarycenter{};
      std::vector<ttk::DiagramType> intermediateDiagrams{thresholdConstraintDiagram, thresholdCurrentDiagram};
      std::vector<std::vector<std::vector<ttk::MatchingType>>> allMatchings;
      std::vector<ttk::DiagramType> centroids{};

      if(PDCMethod == 0){
        persistenceDiagramClustering.setDebugLevel(debugLevel_);
        persistenceDiagramClustering.setThreadNumber(threadNumber);
        // SetForceUseOfAlgorithm ==> Force the progressive approch if 2 inputs
        persistenceDiagramClustering.setForceUseOfAlgorithm(false);
        // setDeterministic ==> Deterministic algorithm
        persistenceDiagramClustering.setDeterministic(true);
        // setUseProgressive ==> Compute Progressive Barycenter
        persistenceDiagramClustering.setUseProgressive(true);
        // setUseInterruptible ==> Interruptible algorithm
        persistenceDiagramClustering.setUseInterruptible(UseTimeThreshold);
        // // setTimeLimit ==> Maximal computation time (s)
        persistenceDiagramClustering.setTimeLimit(TimeThreshold);
        // setUseAdditionalPrecision ==> Force minimum precision on matchings
        persistenceDiagramClustering.setUseAdditionalPrecision(true);
        // setDeltaLim ==> Minimal relative precision
        persistenceDiagramClustering.setDeltaLim(1e-5);
        // setUseAccelerated ==> Use Accelerated KMeans
        persistenceDiagramClustering.setUseAccelerated(false);
        // setUseKmeansppInit ==> KMeanspp Initialization
        persistenceDiagramClustering.setUseKmeansppInit(false);


        std::vector<int> clusterIds = persistenceDiagramClustering.execute(
            intermediateDiagrams, centroids, allMatchings);
      }
      else{
        
        centroids.resize(1);
        const auto wassersteinMetric = std::to_string(2);
        pdBarycenter.setWasserstein(wassersteinMetric);
        pdBarycenter.setMethod(2);
        pdBarycenter.setNumberOfInputs(2);
        pdBarycenter.setDeterministic(1);
        pdBarycenter.setUseProgressive(1);
        pdBarycenter.setDebugLevel(debugLevel_);
        pdBarycenter.setThreadNumber(threadNumber);
        pdBarycenter.setAlpha(1);
        pdBarycenter.setLambda(1);
        if(UseAdditionalPrecisionPDC){
          pdBarycenter.setDeltaLim(DeltaLim); 
        }
        pdBarycenter.execute(
          intermediateDiagrams, centroids[0], allMatchings);

      }

      if(!(UseTheMultiBlocksApproach))
        listTimePersistenceDiagramClustering.push_back(timePersistenceDiagramClustering.getElapsedTime());

      std::vector<std::vector<SimplexId>> allPairsSelected{};
      std::vector<std::vector<SimplexId>> matchingsBlockPairs(centroids[0].size());

      for(auto i = 1; i >= 0; --i){
        std::vector<ttk::MatchingType> &matching = allMatchings[0][i];

        const auto &diag{intermediateDiagrams[i]};

        for(SimplexId j = 0; j < (SimplexId)matching.size(); j++){

          const auto &m{matching[j]};
          const auto &bidderId{std::get<0>(m)};
          const auto &goodId{std::get<1>(m)};

          if((goodId == -1) | (bidderId == -1)){
            continue;
          }

          if (diag[bidderId].persistence() != 0){
            if(i == 1){
              matchingsBlockPairs[goodId].push_back(bidderId);
            }
            else if (matchingsBlockPairs[goodId].size() > 0){
              matchingsBlockPairs[goodId].push_back(bidderId);
            }
            allPairsSelected.push_back({diag[bidderId].birth.id, diag[bidderId].death.id});
          }
        } 
      }

      std::vector<ttk::PersistencePair> pairsToErase{};

      std::map<std::vector<SimplexId>, SimplexId> currentToTarget;
      for(auto &pair : allPairsSelected){
        currentToTarget[{pair[0], pair[1]}] = 1;
      }

      for(auto &pair : intermediateDiagrams[1]){
        if(pair.isFinite != 0){
          if(!(currentToTarget.count({pair.birth.id, pair.death.id }) > 0)){
            pairsToErase.push_back(pair);
          }
        }
      }

      for(auto &pair : pairsToErase){

        if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) && (vertex2PairsTargetDiagram[pair.death.id].size() == 0)){
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
          deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }

        if((vertex2PairsTargetDiagram[pair.birth.id].size() == 0) && (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
          birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
          continue; 
        }

        if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) || (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
          continue; 
        }

        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
      }

      for(const auto& entry : matchingsBlockPairs){
        // Delete pairs that have no equivalence
        if(entry.size() == 1){

          if((vertex2PairsTargetDiagram[thresholdCurrentDiagram[entry[0]].birth.id].size() >= 1) && (vertex2PairsTargetDiagram[thresholdCurrentDiagram[entry[0]].death.id].size() == 0)){
            deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].death.id));
            deathPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
            continue; 
          }

          if((vertex2PairsTargetDiagram[thresholdCurrentDiagram[entry[0]].birth.id].size() == 0) && (vertex2PairsTargetDiagram[thresholdCurrentDiagram[entry[0]].death.id].size() >= 1)){
            birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].birth.id));
            birthPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
            continue; 
          }

          if((vertex2PairsTargetDiagram[thresholdCurrentDiagram[entry[0]].birth.id].size() >= 1) || (vertex2PairsTargetDiagram[thresholdCurrentDiagram[entry[0]].death.id].size() >= 1)){
            continue; 
          }

          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].birth.id));
          birthPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].death.id));
          deathPairToDeleteTargetDiagram.push_back((thresholdCurrentDiagram[entry[0]].birth.sfValue + thresholdCurrentDiagram[entry[0]].death.sfValue) / 2);
          continue;
        }else if(entry.empty())
          continue;

        int64_t valueBirthPairToChangeCurrentDiagram = static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].birth.id);
        int64_t valueDeathPairToChangeCurrentDiagram = static_cast<int64_t>(thresholdCurrentDiagram[entry[0]].death.id);

        double valueBirthPairToChangeTargetDiagram = thresholdConstraintDiagram[entry[1]].birth.sfValue;
        double valueDeathPairToChangeTargetDiagram = thresholdConstraintDiagram[entry[1]].death.sfValue;


        pair2MatchedPair[pairIndiceLocal2Global[entry[1]]][0] = thresholdCurrentDiagram[entry[0]].birth.id;
        pair2MatchedPair[pairIndiceLocal2Global[entry[1]]][1] = thresholdCurrentDiagram[entry[0]].death.id; 

        pairChangeMatchingPair[pairIndiceLocal2Global[entry[1]]] = 1; 

        birthPairToChangeCurrentDiagram.push_back(valueBirthPairToChangeCurrentDiagram);
        birthPairToChangeTargetDiagram.push_back(valueBirthPairToChangeTargetDiagram);
        deathPairToChangeCurrentDiagram.push_back(valueDeathPairToChangeCurrentDiagram);
        deathPairToChangeTargetDiagram.push_back(valueDeathPairToChangeTargetDiagram);
      }
    }
  }
  //=====================================//
  //            Bassic Matching          //
  //=====================================//
  else{
    this->printMsg(
      "Get Indices | Compute wasserstein distance : ", debug::Priority::DETAIL);

    //========================================
    //
    //========================================

    if(epoch == 0){
      for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
        auto &pair = diagramOutput[i]; 
        currentVertex2PairsCurrentDiagram[pair.birth.id].push_back(i);  
        currentVertex2PairsCurrentDiagram[pair.death.id].push_back(i);  
      }
    }
    else{
      std::vector<std::vector<SimplexId>> newVertex2PairsCurrentDiagram(vertexNumber_, std::vector<SimplexId>()); 

      SimplexId numberPairsRemainedTheSame = 0; 
      for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
        auto &pair = diagramOutput[i]; 
        for(auto &pointBirth : currentVertex2PairsCurrentDiagram[pair.birth.id]){
          for(auto &pointDeath : currentVertex2PairsCurrentDiagram[pair.death.id]){
            if(pointBirth == pointDeath){
              numberPairsRemainedTheSame++; 
            }
          }
        } 

        newVertex2PairsCurrentDiagram[pair.birth.id].push_back(i);  
        newVertex2PairsCurrentDiagram[pair.death.id].push_back(i);    
      }

      listAveragePercentageOfImmobilePersistencePairs.push_back(((double)numberPairsRemainedTheSame / diagramOutput.size())); 

      currentVertex2PairsCurrentDiagram = newVertex2PairsCurrentDiagram; 
    }

    //========================================
    //
    //========================================
    
    std::vector<std::vector<SimplexId>> vertex2PairsCurrentDiagram(vertexNumber_, std::vector<SimplexId>()); 
    for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
      auto &pair = diagramOutput[i]; 
      vertex2PairsCurrentDiagram[pair.birth.id].push_back(i);  
      vertex2PairsCurrentDiagram[pair.death.id].push_back(i);  
    }

    std::vector<std::vector<SimplexId>> vertex2PairsTargetDiagram(vertexNumber_, std::vector<SimplexId>()); 
    for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
      auto &pair = constraintDiagram[i]; 
      vertex2PairsTargetDiagram[pair.birth.id].push_back(i);  
      vertex2PairsTargetDiagram[pair.death.id].push_back(i);  
    }

    //=========================================
    //     Compute wasserstein distance
    //=========================================
    ttk::Timer timePersistenceDiagramClustering;

    ttk::PersistenceDiagramClustering persistenceDiagramClustering;
    PersistenceDiagramBarycenter pdBarycenter{};
    std::vector<ttk::DiagramType> intermediateDiagrams{constraintDiagram, diagramOutput};
    std::vector<ttk::DiagramType> centroids;
    std::vector<std::vector<std::vector<ttk::MatchingType>>> allMatchings;


    if(PDCMethod == 0){
      persistenceDiagramClustering.setDebugLevel(debugLevel_);
      persistenceDiagramClustering.setThreadNumber(threadNumber);
      // SetForceUseOfAlgorithm ==> Force the progressive approch if 2 inputs
      persistenceDiagramClustering.setForceUseOfAlgorithm(false);
      // setDeterministic ==> Deterministic algorithm
      persistenceDiagramClustering.setDeterministic(true);
      // setUseProgressive ==> Compute Progressive Barycenter
      persistenceDiagramClustering.setUseProgressive(true);
      // setUseInterruptible ==> Interruptible algorithm
      // persistenceDiagramClustering.setUseInterruptible(true);
      persistenceDiagramClustering.setUseInterruptible(UseTimeThreshold);
      // // setTimeLimit ==> Maximal computation time (s)
      persistenceDiagramClustering.setTimeLimit(TimeThreshold);
      // setUseAdditionalPrecision ==> Force minimum precision on matchings
      persistenceDiagramClustering.setUseAdditionalPrecision(true);
      // setDeltaLim ==> Minimal relative precision
      persistenceDiagramClustering.setDeltaLim(0.00000001);
      // setUseAccelerated ==> Use Accelerated KMeans
      persistenceDiagramClustering.setUseAccelerated(false);
      // setUseKmeansppInit ==> KMeanspp Initialization
      persistenceDiagramClustering.setUseKmeansppInit(false);
    
      std::vector<int> clusterIds = persistenceDiagramClustering.execute(
        intermediateDiagrams, centroids, allMatchings);
    }
    else{
      centroids.resize(1);
      const auto wassersteinMetric = std::to_string(2);
      pdBarycenter.setWasserstein(wassersteinMetric);
      pdBarycenter.setMethod(2);
      pdBarycenter.setNumberOfInputs(2);
      pdBarycenter.setDeterministic(1);
      pdBarycenter.setUseProgressive(1);
      pdBarycenter.setDebugLevel(debugLevel_);
      pdBarycenter.setThreadNumber(threadNumber);
      pdBarycenter.setAlpha(1);
      pdBarycenter.setLambda(1);
      if(UseAdditionalPrecisionPDC){
        pdBarycenter.setDeltaLim(DeltaLim); 
      }
      pdBarycenter.execute(
        intermediateDiagrams, centroids[0], allMatchings);
    }
  
    
    listTimePersistenceDiagramClustering.push_back(timePersistenceDiagramClustering.getElapsedTime());
    this->printMsg(
      "Get Indices | Time Persistence Diagram Clustering : " + std::to_string(timePersistenceDiagramClustering.getElapsedTime()) , debug::Priority::DETAIL);

    //=========================================
    //             Find matched pairs
    //=========================================

    std::vector<std::vector<SimplexId>> allPairsSelected{};
    std::vector<std::vector<std::vector<double>>> matchingsBlock(centroids[0].size());
    std::vector<std::vector<ttk::PersistencePair>> matchingsBlockPairs(centroids[0].size());

    for(auto i = 1; i >= 0; --i){
      std::vector<ttk::MatchingType> &matching = allMatchings[0][i];

      const auto &diag{intermediateDiagrams[i]};

      for(SimplexId j = 0; j < (SimplexId)matching.size(); j++){

        const auto &m{matching[j]};
        const auto &bidderId{std::get<0>(m)};
        const auto &goodId{std::get<1>(m)};

        if((goodId == -1) | (bidderId == -1))
          continue;

        if (diag[bidderId].persistence() != 0){
          matchingsBlock[goodId].push_back({static_cast<double>(diag[bidderId].birth.id), static_cast<double>(diag[bidderId].death.id), diag[bidderId].persistence()});
          if(i == 1){
            matchingsBlockPairs[goodId].push_back(diag[bidderId]);
          }
          else if (matchingsBlockPairs[goodId].size() > 0){
            matchingsBlockPairs[goodId].push_back(diag[bidderId]);
          }
          allPairsSelected.push_back({diag[bidderId].birth.id, diag[bidderId].death.id});
        }
      }
    }

    std::vector<ttk::PersistencePair> pairsToErase{};

    std::map<std::vector<SimplexId>, SimplexId> currentToTarget;
    for(auto &pair : allPairsSelected){
      currentToTarget[{pair[0], pair[1]}] = 1;
    }

    for(auto &pair : intermediateDiagrams[1]){
      if(pair.isFinite != 0){
        if(!(currentToTarget.count({pair.birth.id, pair.death.id }) > 0)){
          pairsToErase.push_back(pair);
        }
      }
    }

    for(auto &pair : pairsToErase){

      if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) && (vertex2PairsTargetDiagram[pair.death.id].size() == 0)){
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
        deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        continue; 
      }

      if((vertex2PairsTargetDiagram[pair.birth.id].size() == 0) && (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
        birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
        continue; 
      }

      if((vertex2PairsTargetDiagram[pair.birth.id].size() >= 1) || (vertex2PairsTargetDiagram[pair.death.id].size() >= 1)){
        continue; 
      }
      birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.birth.id));
      birthPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);
      deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(pair.death.id));
      deathPairToDeleteTargetDiagram.push_back((pair.birth.sfValue + pair.death.sfValue) / 2);

    }

    for(const auto& entry : matchingsBlockPairs){
      // Delete pairs that have no equivalence
      if(entry.size() == 1){

        if((vertex2PairsTargetDiagram[entry[0].birth.id].size() >= 1) && (vertex2PairsTargetDiagram[entry[0].death.id].size() == 0)){
          deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(entry[0].death.id));
          deathPairToDeleteTargetDiagram.push_back((entry[0].birth.sfValue + entry[0].death.sfValue) / 2);
          continue; 
        }

        if((vertex2PairsTargetDiagram[entry[0].birth.id].size() == 0) && (vertex2PairsTargetDiagram[entry[0].death.id].size() >= 1)){
          birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(entry[0].birth.id));
          birthPairToDeleteTargetDiagram.push_back((entry[0].birth.sfValue + entry[0].death.sfValue) / 2);
          continue; 
        }

        if((vertex2PairsTargetDiagram[entry[0].birth.id].size() >= 1) || (vertex2PairsTargetDiagram[entry[0].death.id].size() >= 1)){
          continue;
        }

        birthPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(entry[0].birth.id));
        birthPairToDeleteTargetDiagram.push_back((entry[0].birth.sfValue + entry[0].death.sfValue) / 2);
        deathPairToDeleteCurrentDiagram.push_back(static_cast<int64_t>(entry[0].death.id));
        deathPairToDeleteTargetDiagram.push_back((entry[0].birth.sfValue + entry[0].death.sfValue) / 2);
        // if(entry[0].persistence() > 0.01){
        //   std::cout << "matchingBlockPair : la pair " << entry[0].persistence() << " est supprimé !" << std::endl; 
        // }
        continue;
      } else if(entry.empty())
        continue;

      int64_t valueBirthPairToChangeCurrentDiagram = static_cast<int64_t>(entry[0].birth.id);
      int64_t valueDeathPairToChangeCurrentDiagram = static_cast<int64_t>(entry[0].death.id);

      double valueBirthPairToChangeTargetDiagram = entry[1].birth.sfValue;
      double valueDeathPairToChangeTargetDiagram = entry[1].death.sfValue;

      birthPairToChangeCurrentDiagram.push_back(valueBirthPairToChangeCurrentDiagram);
      birthPairToChangeTargetDiagram.push_back(valueBirthPairToChangeTargetDiagram);
      deathPairToChangeCurrentDiagram.push_back(valueDeathPairToChangeCurrentDiagram);
      deathPairToChangeTargetDiagram.push_back(valueDeathPairToChangeTargetDiagram);
    }

  }

  std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double>> indexToDelete(birthPairToDeleteCurrentDiagram,
                                                birthPairToDeleteTargetDiagram, deathPairToDeleteCurrentDiagram, deathPairToDeleteTargetDiagram);

  std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> > indexToChange(birthPairToChangeCurrentDiagram,
                                                birthPairToChangeTargetDiagram, deathPairToChangeCurrentDiagram, deathPairToChangeTargetDiagram);


  std::tuple<std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >,
              std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >> index(indexToDelete, indexToChange);

  return index;
}


/*
  This function allows you to copy the values of a pytorch tensor 
  to a vector in an optimized way.
*/
int ttk::TopologicalOptimization::tensorToVectorFast(
  const torch::Tensor& tensor, 
  std::vector<double>& result
  ) const {
  TORCH_CHECK(tensor.dtype() == torch::kDouble, "The tensor must be of double type");
  const double* dataPtr = tensor.data_ptr<double>();
  result.assign(dataPtr, dataPtr + tensor.numel());

  return 0; 
}

/*
  Given a coordinate vector this function returns the value of maximum 
  and minimum for each axis and the number of coordinates per axis.
*/
std::vector<std::vector<double>> ttk::TopologicalOptimization::getCoordinatesInformations(
  std::vector<float> coordinatesVertices
  ) const {
  std::vector<double> firstPointCoordinates{};

  double x_min = std::numeric_limits<double>::max();
  double x_max = std::numeric_limits<double>::min();

  double y_min = std::numeric_limits<double>::max();
  double y_max = std::numeric_limits<double>::min();

  double z_min = std::numeric_limits<double>::max();
  double z_max = std::numeric_limits<double>::min();

  std::set<float> uniqueXValues;
  std::set<float> uniqueYValues;
  std::set<float> uniqueZValues;

  for(size_t i = 0; i < coordinatesVertices.size()-2 ; i+=3){
    double x = coordinatesVertices[i];
    double y = coordinatesVertices[i+1];
    double z = coordinatesVertices[i+2];

    uniqueXValues.insert(x);
    uniqueYValues.insert(y);
    uniqueZValues.insert(z);

    if(x_min > x){
      x_min = x;
    }
    if(x_max < x){
      x_max = x;
    }

    if(y_min > y){
      y_min = y;
    }
    if(y_max < y){
      y_max = y;
    }

    if(z_min > z){
      z_min = z;
    }
    if(z_max < z){
      z_max = z;
    }
  }

  firstPointCoordinates.push_back(x_min);
  firstPointCoordinates.push_back(x_max);
  firstPointCoordinates.push_back(y_min);
  firstPointCoordinates.push_back(y_max);
  firstPointCoordinates.push_back(z_min);
  firstPointCoordinates.push_back(z_max);


  std::vector<double> numberOfVerticesAlongXYZ;
  numberOfVerticesAlongXYZ.push_back(static_cast<double>(uniqueXValues.size()));
  numberOfVerticesAlongXYZ.push_back(static_cast<double>(uniqueYValues.size()));
  numberOfVerticesAlongXYZ.push_back(static_cast<double>(uniqueZValues.size()));

  std::vector<std::vector<double>> resultat;
  resultat.push_back(firstPointCoordinates);
  resultat.push_back(numberOfVerticesAlongXYZ);

  return resultat;
}


#ifdef TTK_ENABLE_TORCH
template <typename dataType, typename triangulationType>
int ttk::TopologicalOptimization::execute(
  dataType *const inputScalars,
  dataType *const outputScalars,
  SimplexId *const inputOffsets,
  triangulationType *triangulation,
  ttk::DiagramType &constraintDiagram,
  int *const modificationNumber,
  int *const lastChange,
  int *const idBlock
  ) const {

  
  Timer t;

  //=======================================
  //            Stop Condition
  //=======================================

  double stopCondition = 0; 

  //=======================================
  //      Topological Simplification 
  //=======================================

  std::vector<float> listTimePersistenceDiagram;
  std::vector<float> listTimePersistenceDiagramClustering;
  std::vector<float> listTimeBackPropagation;
  std::vector<float> listAveragePercentageOfModifiedVertices; 
  std::vector<float> listTimeInteration;
  std::vector<float> listAveragePercentageOfImmobilePersistencePairs; 
  SimplexId numberPairsInputDiagram = -1; 
  std::vector<SimplexId> pairOnEdgeToRemove;
  
  getStopCondition(triangulation, inputOffsets, inputScalars, constraintDiagram, stopCondition, numberPairsInputDiagram); 

  if(UseTopologicalSimplification){

    std::vector<dataType> height(vertexNumber_, 0); 
     
    #ifdef TTK_ENABLE_OPENMP
    #pragma omp parallel for num_threads(threadNumber_)
    #endif
    for(SimplexId i = 0; i < vertexNumber_; i++){
      height[i] = inputScalars[i]; 
    }

    std::vector<ttk::SimplexId> order(vertexNumber_);
    ttk::preconditionOrderArray(vertexNumber_, height.data(), order.data(), 1);

    std::vector<dataType> simplifiedHeight = height;
    std::vector<ttk::SimplexId> authorizedCriticalPoints, simplifiedOrder = order;

    for(auto &pair : constraintDiagram){
      authorizedCriticalPoints.push_back(pair.birth.id);
      authorizedCriticalPoints.push_back(pair.death.id);
    }

    ttk::TopologicalSimplification simplification;
    simplification.preconditionTriangulation(triangulation);
    simplification.execute(height.data(), simplifiedHeight.data(),
                                authorizedCriticalPoints.data(), order.data(),
                                simplifiedOrder.data(),
                                authorizedCriticalPoints.size(), true , *triangulation);
  
    if(UseTheMultiBlocksApproach){
      this->printMsg(
      "Execute | ExecuteMultiBlock " , debug::Priority::PERFORMANCE);
      executeMultiBlock(simplifiedHeight.data(), outputScalars, inputOffsets, triangulation, constraintDiagram, modificationNumber, lastChange, idBlock, stopCondition, numberPairsInputDiagram);
    }
    else{
      this->printMsg(
      "Execute | ExecuteOneBlock " ,  debug::Priority::PERFORMANCE);
      ttk::Triangulation * triangulationOneBlock = nullptr; 
      std::vector<SimplexId> localToGlobal = {}; 
      std::vector<float> maxAndMinCoordinateNode = {}; 


      executeOneBlock(simplifiedHeight.data(), outputScalars, inputOffsets, triangulation, triangulationOneBlock, constraintDiagram, modificationNumber, lastChange, vertexNumber_, pairOnEdgeToRemove,
      listTimePersistenceDiagram,listTimePersistenceDiagramClustering,  listTimeBackPropagation, listAveragePercentageOfModifiedVertices, listTimeInteration, listAveragePercentageOfImmobilePersistencePairs, numberPairsInputDiagram, localToGlobal, false, LearningRate, -1, EpochNumber, maxAndMinCoordinateNode, stopCondition);
    }
  }
  else{
    if(UseTheMultiBlocksApproach){
      this->printMsg(
      "Execute | ExecuteMultiBlock " , debug::Priority::PERFORMANCE);
      executeMultiBlock(inputScalars, outputScalars, inputOffsets, triangulation, constraintDiagram, modificationNumber, lastChange, idBlock, stopCondition, numberPairsInputDiagram);
    }
    else{
      this->printMsg(
      "Execute | ExecuteOneBlock " , debug::Priority::PERFORMANCE);
      ttk::Triangulation * triangulationOneBlock = nullptr; 
      std::vector<SimplexId> localToGlobal = {}; 
      std::vector<float> maxAndMinCoordinateNode = {}; 


      executeOneBlock(inputScalars, outputScalars, inputOffsets, triangulation, triangulationOneBlock, constraintDiagram, modificationNumber, lastChange, vertexNumber_, pairOnEdgeToRemove,
      listTimePersistenceDiagram,listTimePersistenceDiagramClustering,  listTimeBackPropagation, listAveragePercentageOfModifiedVertices, listTimeInteration, listAveragePercentageOfImmobilePersistencePairs, numberPairsInputDiagram, localToGlobal, false, LearningRate, -1, EpochNumber, maxAndMinCoordinateNode, stopCondition);
    }
  }

  //========================================
  //            Information display
  //========================================

  // Total execution time
  double time = t.getElapsedTime(); 
  this->printMsg(
        "ExecuteOneBlock | Total execution time =  " + std::to_string(time) , debug::Priority::PERFORMANCE);

  // Number Pairs Input Diagram 
  this->printMsg(
        "ExecuteOneBlock | Number Pairs Input Diagram =  " + std::to_string(numberPairsInputDiagram) , debug::Priority::PERFORMANCE);

  // Number Pairs Constraint Diagram
  SimplexId numberPairsConstraintDiagram = (SimplexId)constraintDiagram.size(); 
  this->printMsg(
        "ExecuteOneBlock | Number Pairs Constraint Diagram =  " + std::to_string(numberPairsConstraintDiagram) , debug::Priority::PERFORMANCE);

  if(!UseTheMultiBlocksApproach){
    // Time Persistence Diagram
    double averagePersistenceDiagram = std::accumulate(listTimePersistenceDiagram.begin(), listTimePersistenceDiagram.end(), 0.0) / listTimePersistenceDiagram.size();
    double totalTimePersistenceDiagram = std::accumulate(listTimePersistenceDiagram.begin(), listTimePersistenceDiagram.end(), 0.0);
    this->printMsg(
          "ExecuteOneBlock | Total time spent calculating the persistence diagram : " + std::to_string(totalTimePersistenceDiagram) , debug::Priority::PERFORMANCE);
    this->printMsg(
          "ExecuteOneBlock | The average time spent on persistence diagrams : " + std::to_string(averagePersistenceDiagram), debug::Priority::PERFORMANCE);


    // Time Persistence Diagram Clustering
    double averagePersistenceDiagramClustering = std::accumulate(listTimePersistenceDiagramClustering.begin(), listTimePersistenceDiagramClustering.end(), 0.0) / listTimePersistenceDiagramClustering.size();
    double totalTimeSpentCalculatingThePersistenceDiagramClustering =  std::accumulate(listTimePersistenceDiagramClustering.begin(), listTimePersistenceDiagramClustering.end(), 0.0); 
    this->printMsg(
          "ExecuteOneBlock | Total time spent calculating the persistence diagram clustering : " + std::to_string(totalTimeSpentCalculatingThePersistenceDiagramClustering) , debug::Priority::PERFORMANCE);
    this->printMsg(
          "ExecuteOneBlock | The average time spent on persistence diagrams clustering : " + std::to_string(averagePersistenceDiagramClustering) , debug::Priority::PERFORMANCE);

    // Average Percentage Of Modified Vertices
    double totalAveragePercentageOfModifiedVertices = std::accumulate(listAveragePercentageOfModifiedVertices.begin(), listAveragePercentageOfModifiedVertices.end(), 0.0) / listAveragePercentageOfModifiedVertices.size(); 
    this->printMsg( 
          "ExecuteOneBlock | Total Average Percentage Of Modified Vertices : " + std::to_string(totalAveragePercentageOfModifiedVertices) , debug::Priority::PERFORMANCE);

    // Average Percentage Of Immobile Persistence Pairs
    double totalAveragePercentageOfImmobilePersistencePairs = 0; 
    if(listAveragePercentageOfImmobilePersistencePairs.size() != 0){
      totalAveragePercentageOfImmobilePersistencePairs = std::accumulate(listAveragePercentageOfImmobilePersistencePairs.begin(), listAveragePercentageOfImmobilePersistencePairs.end(), 0.0) / listAveragePercentageOfImmobilePersistencePairs.size(); 
      this->printMsg(
            "ExecuteOneBlock | Total Average Percentage Of Immobile Persistence Pairs : " + std::to_string(totalAveragePercentageOfImmobilePersistencePairs) , debug::Priority::PERFORMANCE);
    }
  }

  this->printMsg(
        "Stop condition : " + std::to_string(stopCondition),  debug::Priority::PERFORMANCE);

  this->printMsg(
    "Optimization scalar field", 1.0, time, this->threadNumber_);

  

  return 0;
}
#endif


#ifdef TTK_ENABLE_TORCH
template <typename dataType, typename triangulationType>
int ttk::TopologicalOptimization::executeMultiBlock(
  dataType *const inputScalars,
  dataType *const outputScalars,
  SimplexId *const inputOffsets,
  triangulationType *triangulation,
  ttk::DiagramType &constraintDiagram,
  int *const modificationNumber,
  int *const lastChange,
  int *const idBlock, 
  double &stopCondition, 
  SimplexId numberPairsInputDiagram
  ) const {

  //========================================
  //       We set omp_nested to true           
  //========================================

  bool  nestedOptionWasDisabled = false;
  if(!(omp_get_nested())){
    omp_set_nested(true);
    nestedOptionWasDisabled = true;
  }

  //========================================
  //            Creation of blocks
  //========================================
  SimplexId vertexNumber;
  int numberOfBlocks;

  
  vertexNumber = vertexNumber_;
  numberOfBlocks = NumberOfBlocksPerThread * threadNumber_;

  Timer t_kdtree;
  //========================================================//
  //             Find the node number that we need          //
  //========================================================//

  int dimension = triangulation->getDimensionality();
  int maximumLevel = ceil(log2(numberOfBlocks + 1))-1;

  if(pow(2, maximumLevel) < numberOfBlocks){
    maximumLevel++;
  }
  int nodeNumber = pow(2, maximumLevel+1) - 1;
  
  this->printMsg(
  "ExecuteMultiBlock | NodeNumber = " + std::to_string(nodeNumber), debug::Priority::DETAIL);

  //========================================================//
  //                  Get point coordinates                 //
  //========================================================//

  std::vector<double> coordinates(dimension * vertexNumber);
  std::vector<std::vector<float>> verticesCoordinates(vertexNumber);

  #ifdef TTK_ENABLE_OPENMP
  #pragma omp parallel for num_threads(threadNumber_)
  #endif
  for(SimplexId vertexId = 0; vertexId < vertexNumber; vertexId++){
    float x = 0, y = 0, z = 0;
    triangulation->getVertexPoint(vertexId, x, y, z);
    coordinates[vertexId * dimension] = x;
    coordinates[vertexId * dimension + 1] = y;
    if(dimension == 3){
      coordinates[vertexId * dimension + 2] = z;
    }
    verticesCoordinates[vertexId].push_back(x);
    verticesCoordinates[vertexId].push_back(y);
    verticesCoordinates[vertexId].push_back(z);
  }

  //========================================================//
  //                  Building the kdtree                   //
  //========================================================//

  KDTree<double, std::array<double, 3>> kdtree(false, 2);
  std::vector<KDTree<double, std::array<double, 3>> *> treeMap
    = kdtree.build(coordinates.data(), vertexNumber, dimension, {}, 1 ,nodeNumber, true);

  //========================================================//
  //                 Get tree information                   //
  //========================================================//

  // Get the node that corresponds to the root and the parents of each node
  KDTree<double, std::array<double, 3>>* rootNode = nullptr;
  std::vector<std::vector<int>> levelToNodes(maximumLevel+1);
  std::vector<int> nodeIdToParentId(nodeNumber);
  // We associate each node in the tree with its id_
  std::vector<KDTree<double, std::array<double, 3>>*> nodeIdToNodeKDT(treeMap.size());

  for (KDTree<double, std::array<double, 3>>* node : treeMap){
    levelToNodes[node->level_].push_back(node->id_);
    nodeIdToNodeKDT[node->id_] = node;
    if(node->isRoot()){
      rootNode = node;
      nodeIdToParentId[node->id_] = -1;
    }else{
      nodeIdToParentId[node->id_] = node->parent_->id_;
    }
  }


  // Creation of inputScalars and inputOffsets variables for each node
  std::vector<std::vector<dataType>> nodeToInputScalars(nodeNumber);
  std::vector<std::vector<dataType>> nodeToOutputScalars(nodeNumber);
  std::vector<std::vector<SimplexId>> nodeToInputOffsets(nodeNumber);
  std::vector<std::vector<float>> nodeToPointDataArray(nodeNumber);
  std::vector<std::vector<int>> nodeToModificationNumber(nodeNumber);
  std::vector<std::vector<int>> nodeToLastChange(nodeNumber);
  
  std::vector<std::vector<std::vector<int>>> vertex2nodes(vertexNumber, std::vector<std::vector<int>>(maximumLevel+1, std::vector<int>()));
  std::vector<std::vector<SimplexId>> node2vertices(nodeNumber, std::vector<SimplexId>()); 

  std::vector<std::vector<SimplexId>> localToGlobal(nodeNumber); 
  std::vector<std::vector<SimplexId>> globalToLocal(nodeNumber, std::vector<SimplexId>(vertexNumber, -1)); 

  // Vertices that are in multiple blocks
  std::vector<SimplexId> vertexInMultipleBlocks(vertexNumber, -1);
  
  std::vector<std::vector<float>> maxAndMinCoordinateNode(nodeNumber); 

  for(SimplexId i = 0; i < vertexNumber; i++){
    float x = verticesCoordinates[i][0];
    float y = verticesCoordinates[i][1];
    float z = verticesCoordinates[i][2];

    vertex2nodes[i][0].push_back(rootNode->id_);
    node2vertices[rootNode->id_].push_back(i); 
    nodeToInputScalars[rootNode->id_].push_back(inputScalars[i]); 
    localToGlobal[rootNode->id_].push_back(i); 
    globalToLocal[rootNode->id_][i] = nodeToInputScalars[rootNode->id_].size()-1; 

    nodeToOutputScalars[rootNode->id_].push_back(0); 
    nodeToModificationNumber[rootNode->id_].push_back(0); 
    nodeToLastChange[rootNode->id_].push_back(0); 
    nodeToInputOffsets[rootNode->id_].push_back(nodeToInputScalars[rootNode->id_].size()-1);
    nodeToPointDataArray[rootNode->id_].push_back(x); 
    nodeToPointDataArray[rootNode->id_].push_back(y); 
    nodeToPointDataArray[rootNode->id_].push_back(z); 
    
    for(int level = 0; level < maximumLevel; level++){
      for(auto &idNode : vertex2nodes[i][level]){

        KDTree<double, std::array<double, 3>>* currentNode = nodeIdToNodeKDT[idNode];
        
        // We check if the vertex is in the right side
        int rightNodeId = currentNode->right_->id_;
        KDTree<double, std::array<double, 3>>* rightNode = nodeIdToNodeKDT[rightNodeId];
        auto coordsMinRightNode = rightNode->coords_min_;
        auto coordsMaxRightNode = rightNode->coords_max_;
        
        maxAndMinCoordinateNode[rightNodeId].push_back(coordsMinRightNode[0]); 
        maxAndMinCoordinateNode[rightNodeId].push_back(coordsMinRightNode[1]); 
        maxAndMinCoordinateNode[rightNodeId].push_back(coordsMinRightNode[2]); 
        maxAndMinCoordinateNode[rightNodeId].push_back(coordsMaxRightNode[0]); 
        maxAndMinCoordinateNode[rightNodeId].push_back(coordsMaxRightNode[1]); 
        maxAndMinCoordinateNode[rightNodeId].push_back(coordsMaxRightNode[2]); 

        bool presentInRightBlock = false; 

        if(dimension == 3){
          presentInRightBlock = ((((level+1) % dimension) == 1) && (coordsMinRightNode[0] <= x)) 
                                  || ((((level+1) % dimension) == 2) && (coordsMinRightNode[1] <= y)) 
                                  || ((((level+1) % dimension) == 0) && ((coordsMinRightNode[2] <= z))); 
        }
        else if(dimension == 2){
          presentInRightBlock = (((((level+1) % dimension) == 1) && (coordsMinRightNode[0] <= x)) 
                                || ((((level+1) % dimension) == 0) && (coordsMinRightNode[1] <= y))); 
        }


        if(presentInRightBlock){
  
          vertex2nodes[i][level+1].push_back(rightNodeId);

          node2vertices[rightNodeId].push_back(i); 
          nodeToInputScalars[rightNodeId].push_back(inputScalars[i]); 
          localToGlobal[rightNodeId].push_back(i); 
          globalToLocal[rightNodeId][i] = nodeToInputScalars[rightNodeId].size()-1; 

          nodeToOutputScalars[rightNodeId].push_back(0); 
          nodeToModificationNumber[rightNodeId].push_back(0); 
          nodeToLastChange[rightNodeId].push_back(0);   
          nodeToInputOffsets[rightNodeId].push_back(nodeToInputScalars[rightNodeId].size()-1);
          nodeToPointDataArray[rightNodeId].push_back(x); 
          nodeToPointDataArray[rightNodeId].push_back(y); 
          nodeToPointDataArray[rightNodeId].push_back(z); 
        }
        
        // We check if the vertex is in the left side
        int leftNodeId = currentNode->left_->id_;
        KDTree<double, std::array<double, 3>>* leftNode = nodeIdToNodeKDT[leftNodeId];

        auto coordsMinLeftNode = leftNode->coords_min_;
        auto coordsMaxLeftNode = leftNode->coords_max_;

        maxAndMinCoordinateNode[leftNodeId].push_back(coordsMinLeftNode[0]); 
        maxAndMinCoordinateNode[leftNodeId].push_back(coordsMinLeftNode[1]); 
        maxAndMinCoordinateNode[leftNodeId].push_back(coordsMinLeftNode[2]); 
        maxAndMinCoordinateNode[leftNodeId].push_back(coordsMaxLeftNode[0]); 
        maxAndMinCoordinateNode[leftNodeId].push_back(coordsMaxLeftNode[1]); 
        maxAndMinCoordinateNode[leftNodeId].push_back(coordsMaxLeftNode[2]); 

        bool presentInLeftBlock = false; 
        
        if(dimension == 3){
          presentInLeftBlock = ((((level+1) % dimension) == 1) && (coordsMinRightNode[0] >= x)) 
                                  || ((((level+1) % dimension) == 2) && (coordsMinRightNode[1] >= y)) 
                                  || ((((level+1) % dimension) == 0) && (coordsMinRightNode[2] >= z)); 
        }
        else if(dimension == 2){
          presentInLeftBlock = (((((level+1) % dimension) == 1) && (coordsMinRightNode[0] >= x)) 
                                  || ((((level+1) % dimension) == 0) && (coordsMinRightNode[1] >= y))); 
        }

        if(presentInLeftBlock){
          // Vertex are in multiple blocks
          if(vertex2nodes[i][level+1].size() > 0){
            vertexInMultipleBlocks[i] = 1; 
          }

          vertex2nodes[i][level+1].push_back(leftNodeId);

          node2vertices[leftNodeId].push_back(i); 
          nodeToInputScalars[leftNodeId].push_back(inputScalars[i]); 
          localToGlobal[leftNodeId].push_back(i); 
          globalToLocal[leftNodeId][i] = nodeToInputScalars[leftNodeId].size()-1; 

          nodeToOutputScalars[leftNodeId].push_back(0); 
          nodeToModificationNumber[leftNodeId].push_back(0); 
          nodeToLastChange[leftNodeId].push_back(0); 
          nodeToInputOffsets[leftNodeId].push_back(nodeToInputScalars[leftNodeId].size()-1);
          nodeToPointDataArray[leftNodeId].push_back(x); 
          nodeToPointDataArray[leftNodeId].push_back(y); 
          nodeToPointDataArray[leftNodeId].push_back(z); 

        }
      } 
    }
  }

  #ifdef TTK_ENABLE_OPENMP
  #pragma omp parallel for num_threads(threadNumber_)
  #endif
  for(SimplexId i = 0; i < vertexNumber; i++){
    // Visualization of the association between vertices and nodes
    if(vertexInMultipleBlocks[i] == -1){
      idBlock[i] = vertex2nodes[i][maximumLevel][0];
    }
    else{
      idBlock[i] = -1;
    }
  }

  //===============================================================//
  //      Compute the persistence diagram of the initial data      //
  //===============================================================//

  ttk::PersistenceDiagram diagram;
  std::vector<ttk::PersistencePair> diagramOutput;
  ttk::preconditionOrderArray<dataType>(vertexNumber, inputScalars, inputOffsets, threadNumber_);
  diagram.setDebugLevel(debugLevel_);
  diagram.setThreadNumber(threadNumber_);
  diagram.preconditionTriangulation(triangulation);
  diagram.execute(
    diagramOutput, inputScalars, 0, inputOffsets, triangulation);

  //===================================================//
  //       Get Stop Condition for the last block       //
  //===================================================//

  std::cout << "stopCondition = " << stopCondition << std::endl; 

  //========================================================//
  //                      Detect noise                      //
  //========================================================//

  std::vector<SimplexId> pairOnEdgeToRemove(vertexNumber, -1); 
  std::vector<std::vector<SimplexId>> vertex2PairsCurrentDiagram(vertexNumber_, std::vector<SimplexId>()); 
  for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
    auto &pair = diagramOutput[i]; 
    vertex2PairsCurrentDiagram[pair.birth.id].push_back(i);  
    vertex2PairsCurrentDiagram[pair.death.id].push_back(i);  
  }

  std::vector<std::vector<SimplexId>> vertex2PairsTargetDiagram(vertexNumber_, std::vector<SimplexId>()); 
  for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
    auto &pair = constraintDiagram[i]; 
    vertex2PairsTargetDiagram[pair.birth.id].push_back(i);  
    vertex2PairsTargetDiagram[pair.death.id].push_back(i);  
  }
 
  std::vector<SimplexId> matchingPairCurrentDiagram((SimplexId)diagramOutput.size(), -1); 

 
  for(SimplexId i = 0; i < (SimplexId)constraintDiagram.size(); i++){
    auto &pair = constraintDiagram[i]; 
    
    SimplexId birthId = pair.birth.id; 
    SimplexId deathId = pair.death.id; 

    for(auto &idPairBirth : vertex2PairsCurrentDiagram[birthId]){
      for(auto &idPairDeath : vertex2PairsCurrentDiagram[deathId]){
        if(idPairBirth == idPairDeath){
          matchingPairCurrentDiagram[idPairBirth] = 1;
        }
      }
    }
  }

  for(SimplexId i = 0; i < (SimplexId)diagramOutput.size(); i++){
    auto &pair = diagramOutput[i]; 
    if(matchingPairCurrentDiagram[i] == -1){
      pairOnEdgeToRemove[pair.birth.id] = -i-3; 
      pairOnEdgeToRemove[pair.death.id] = -i-3; 
    }
  }

  //========================================================//
  //            Creation of the triangulation               //
  //========================================================//

  std::vector<double> levelTimes(levelToNodes.size());

  //==================================================//
  //             Cases with several levels            // 
  //==================================================//

  if((!(MultiBlockOneLevel) && (UseTheMultiBlocksApproach))){
    int threadNumberPerBlock=0;
    int threadNumberPerLevel=0;

    for(size_t i = levelToNodes.size(); i > 1; --i){
      std::vector<int> listOfNode = levelToNodes[i-1];
      Timer t_level;

      //================================================================
      //      Distribution of threads for the execution of each block
      //================================================================
      if(static_cast<int>(listOfNode.size()) >= threadNumber_){ 
        // If it's the first level or we have more blocks than threads
        threadNumberPerBlock = 1;
        threadNumberPerLevel = threadNumber_;
      }
      else if((i == levelToNodes.size()) && (static_cast<int>(listOfNode.size()) >= threadNumber_)){
        threadNumberPerBlock = 1;
        threadNumberPerLevel = threadNumber_;
      }
      else if((i == levelToNodes.size()) && (static_cast<int>(listOfNode.size()) < threadNumber_)){
        threadNumberPerBlock = threadNumber_/static_cast<int>(listOfNode.size());
        threadNumberPerLevel = static_cast<int>(listOfNode.size());
      }
      else{
        threadNumberPerBlock *= 2;
        threadNumberPerLevel /= 2;
      }
 
      #ifdef TTK_ENABLE_OPENMP
      #pragma omp parallel for num_threads(threadNumberPerLevel)
      #endif 
      for(size_t j = 0; j < listOfNode.size(); j++){

        std::vector<float> listTimePersistenceDiagram;
        std::vector<float> listTimePersistenceDiagramClustering;
        std::vector<float> listTimeBackPropagation;
        std::vector<float> listAveragePercentageOfModifiedVertices; 
        std::vector<float> listTimeInteration; 
        std::vector<float> listAveragePercentageOfImmobilePersistencePairs;

        int idNode = listOfNode[j];

        // We will create the triangulation
        ttk::Triangulation triangulationOfTheNode;
        if((std::is_same<triangulationType, ttk::ImplicitNoPreconditions>::value) ||
                (std::is_same<triangulationType, ttk::ImplicitWithPreconditions>::value)||
                (std::is_same<triangulationType, ttk::PeriodicNoPreconditions >::value) ||
                (std::is_same<triangulationType, ttk::PeriodicWithPreconditions>::value)){

          ttk::ImplicitWithPreconditions* triangulationWithPreconditions = dynamic_cast<ttk::ImplicitWithPreconditions*>(triangulation);

          float* spacing = triangulationWithPreconditions->getSpacing();

          std::vector<std::vector<double>> coordinatesInformations = getCoordinatesInformations(nodeToPointDataArray[idNode]);
          std::vector<double> &maxAndMinCoordinates = coordinatesInformations[0];
          std::vector<double> &numberOfVerticesAlongXYZ = coordinatesInformations[1];

          triangulationOfTheNode.setInputGrid(maxAndMinCoordinates[0], maxAndMinCoordinates[2], maxAndMinCoordinates[4],
                                      spacing[0], spacing[1], spacing[2], numberOfVerticesAlongXYZ[0],
                                      numberOfVerticesAlongXYZ[1], numberOfVerticesAlongXYZ[2]);
        }
        else{
          std::vector<std::vector<double>> coordinatesInformations = getCoordinatesInformations(nodeToPointDataArray[idNode]);
          std::vector<double> maxAndMinCoordinates = coordinatesInformations[0];
          std::vector<double> numberOfVerticesAlongXYZ = coordinatesInformations[1];

          triangulationOfTheNode.setInputGrid(maxAndMinCoordinates[0], maxAndMinCoordinates[2], maxAndMinCoordinates[4],
                                      1, 1, 1, numberOfVerticesAlongXYZ[0],
                                      numberOfVerticesAlongXYZ[1], numberOfVerticesAlongXYZ[2]);
        }

        triangulationOfTheNode.preconditionBoundaryVertices(); 
        triangulationOfTheNode.preconditionVertexNeighbors();

        SimplexId vertexNumberOfTheNode = nodeToInputScalars[idNode].size();
        dataType * outputScalarsOfTheNode = nodeToOutputScalars[idNode].data();
        SimplexId * inputOffsetsOfTheNode = nodeToInputOffsets[idNode].data();
        int * modificationNumberOfTheNode = nodeToModificationNumber[idNode].data();
        int * lastChangeOfTheNode = nodeToLastChange[idNode].data();

        // We retrieve the other variables to call executeOneBlock
        dataType * inputScalarsOfTheNode = nodeToInputScalars[idNode].data();    
        
        this->printMsg(
        "ExecuteMultiBlock | Node Id = " + std::to_string(idNode) , debug::Priority::DETAIL);

        std::vector<ttk::PersistencePair> constraintDiagramThreshold;
        for(auto &pair : constraintDiagram){
          if((globalToLocal[idNode][pair.birth.id] != -1) || (globalToLocal[idNode][pair.death.id] != -1)){
            constraintDiagramThreshold.push_back(pair); 
          }
        }

        executeOneBlock(inputScalarsOfTheNode, outputScalarsOfTheNode, inputOffsetsOfTheNode,
                        triangulation, &triangulationOfTheNode, constraintDiagramThreshold, modificationNumberOfTheNode,
                        lastChangeOfTheNode, vertexNumberOfTheNode, pairOnEdgeToRemove,
                        listTimePersistenceDiagram, listTimePersistenceDiagramClustering,
                        listTimeBackPropagation, listAveragePercentageOfModifiedVertices, listTimeInteration, listAveragePercentageOfImmobilePersistencePairs, 
                        numberPairsInputDiagram, localToGlobal[idNode], false, LearningRate, threadNumberPerBlock,
                        NumberEpochMultiBlock,  maxAndMinCoordinateNode[idNode]);

        // We get the parent of the node
        int parentId = nodeIdToParentId[idNode];

        // We update the data of the parent
        #ifdef TTK_ENABLE_OPENMP
        #pragma omp parallel for num_threads(threadNumberPerBlock)
        #endif
        for(SimplexId k = 0; k < vertexNumberOfTheNode; k++){
          SimplexId idGlobalVertex = localToGlobal[idNode][k];
          SimplexId indexParentData = globalToLocal[parentId][idGlobalVertex];
          
          if(vertexInMultipleBlocks[idGlobalVertex] != -1){
            if(nodeToInputScalars[parentId][indexParentData] != outputScalarsOfTheNode[k]){
              #pragma omp critical
              nodeToInputScalars[parentId][indexParentData] = outputScalarsOfTheNode[k];
            }
          }
          else{
            nodeToInputScalars[parentId][indexParentData] = outputScalarsOfTheNode[k];
          }
        }
      }
      double time_level = t_level.getElapsedTime();
      levelTimes[i-1] = time_level;
    }
  }
  //==================================================//
  //             Cases with only one level            // 
  //==================================================//
  else{
    std::vector<int> listOfNode = levelToNodes[levelToNodes.size()-1];
    Timer t_level;

    #ifdef TTK_ENABLE_OPENMP
    #pragma omp parallel for num_threads(threadNumber_)
    #endif
    for(size_t j = 0; j < listOfNode.size(); j++){

      std::vector<float> listTimePersistenceDiagram;
      std::vector<float> listTimePersistenceDiagramClustering;
      std::vector<float> listTimeBackPropagation;
      std::vector<float> listAveragePercentageOfModifiedVertices; 
      std::vector<float> listTimeInteration;
      std::vector<float> listAveragePercentageOfImmobilePersistencePairs;

      int idNode = listOfNode[j];

      // We will create the triangulation
      ttk::Triangulation triangulationOfTheNode;
      // if(UseAllApproach){

      //   std::vector<std::vector<double>> coordinatesInformations = getCoordinatesInformations(nodeToPointDataArray[idNode]);
      //   std::vector<double> maxAndMinCoordinates = coordinatesInformations[0];
      //   std::vector<double> numberOfVerticesAlongXYZ = coordinatesInformations[1];

      //   triangulationOfTheNode.setInputGrid(maxAndMinCoordinates[0], maxAndMinCoordinates[2], maxAndMinCoordinates[4],
      //                               1, 1, 1, numberOfVerticesAlongXYZ[0],
      //                               numberOfVerticesAlongXYZ[1], numberOfVerticesAlongXYZ[2]);
      // }
      // else 
      if((std::is_same<triangulationType, ttk::ImplicitNoPreconditions>::value) ||
              (std::is_same<triangulationType, ttk::ImplicitWithPreconditions>::value)||
              (std::is_same<triangulationType, ttk::PeriodicNoPreconditions >::value) ||
              (std::is_same<triangulationType, ttk::PeriodicWithPreconditions>::value)){

        ttk::ImplicitWithPreconditions* triangulationWithPreconditions = dynamic_cast<ttk::ImplicitWithPreconditions*>(triangulation);

        float* spacing = triangulationWithPreconditions->getSpacing();

        std::vector<std::vector<double>> coordinatesInformations = getCoordinatesInformations(nodeToPointDataArray[idNode]);
        std::vector<double> &maxAndMinCoordinates = coordinatesInformations[0];
        std::vector<double> &numberOfVerticesAlongXYZ = coordinatesInformations[1];

        triangulationOfTheNode.setInputGrid(maxAndMinCoordinates[0], maxAndMinCoordinates[2], maxAndMinCoordinates[4],
                                    spacing[0], spacing[1], spacing[2], numberOfVerticesAlongXYZ[0],
                                    numberOfVerticesAlongXYZ[1], numberOfVerticesAlongXYZ[2]);
      }

      triangulationOfTheNode.preconditionBoundaryVertices(); 
      triangulationOfTheNode.preconditionVertexNeighbors();

      // We retrieve the other variables to call executeOneBlock
      dataType * outputScalarsOfTheNode = nodeToOutputScalars[idNode].data();
      SimplexId * inputOffsetsOfTheNode = nodeToInputOffsets[idNode].data();
      int * modificationNumberOfTheNode = nodeToModificationNumber[idNode].data();
      int * lastChangeOfTheNode = nodeToLastChange[idNode].data();
      SimplexId vertexNumberOfTheNode = nodeToInputScalars[idNode].size();      
      dataType * inputScalarsOfTheNode = nodeToInputScalars[idNode].data();

      std::vector<ttk::PersistencePair> constraintDiagramThreshold;
      for(auto &pair : constraintDiagram){
        if((globalToLocal[idNode][pair.birth.id] != -1) || (globalToLocal[idNode][pair.death.id] != -1)){
          constraintDiagramThreshold.push_back(pair); 
        }
      }

      executeOneBlock(inputScalarsOfTheNode, outputScalarsOfTheNode, inputOffsetsOfTheNode,
                      triangulation, &triangulationOfTheNode, constraintDiagramThreshold, modificationNumberOfTheNode,
                      lastChangeOfTheNode, vertexNumberOfTheNode, pairOnEdgeToRemove, 
                      listTimePersistenceDiagram, listTimePersistenceDiagramClustering,
                      listTimeBackPropagation, listAveragePercentageOfModifiedVertices, listTimeInteration, 
                      listAveragePercentageOfImmobilePersistencePairs, numberPairsInputDiagram, localToGlobal[idNode], false, LearningRate, 1,
                      NumberEpochMultiBlock, maxAndMinCoordinateNode[idNode]);

      // We update the data of the parent
      #ifdef TTK_ENABLE_OPENMP
      #pragma omp parallel for num_threads(threadNumber_)
      #endif
      for(SimplexId k = 0; k < vertexNumberOfTheNode; k++){
        SimplexId idGlobalVertex = node2vertices[idNode][k];
        if(vertexInMultipleBlocks[idGlobalVertex] != -1){
          if(nodeToInputScalars[0][idGlobalVertex] != outputScalarsOfTheNode[k]){
            #pragma omp critical
            nodeToInputScalars[0][idGlobalVertex] = (nodeToInputScalars[0][idGlobalVertex] + outputScalarsOfTheNode[k]) / 2;
          }
        }
        else{
          nodeToInputScalars[0][idGlobalVertex] = outputScalarsOfTheNode[k];
        }
      }
    }
    double time_level = t_level.getElapsedTime();
    levelTimes[1] = time_level;
  }

  //=======================================
  //        Root block optimization
  //=======================================

  std::vector<float> listTimePersistenceDiagram;
  std::vector<float> listTimePersistenceDiagramClustering;
  std::vector<float> listTimeBackPropagation;
  std::vector<float> listAveragePercentageOfModifiedVertices; 
  std::vector<float> listTimeInteration;
  std::vector<float> listAveragePercentageOfImmobilePersistencePairs; 
  std::vector<SimplexId> localToGlobal0 = {}; 
  pairOnEdgeToRemove = {}; 
  ttk::Triangulation * triangulationOneBlock = nullptr; 
  Timer timeLastLevel;
  // idNode == 0
  dataType * inputScalarsOfTheNode = nodeToInputScalars[0].data();

  // We call executeOneBlock
  executeOneBlock(inputScalarsOfTheNode, outputScalars, inputOffsets,
                  triangulation, triangulationOneBlock, constraintDiagram, modificationNumber,
                  lastChange, vertexNumber, pairOnEdgeToRemove, listTimePersistenceDiagram,
                  listTimePersistenceDiagramClustering, listTimeBackPropagation, 
                  listAveragePercentageOfModifiedVertices, listTimeInteration, 
                  listAveragePercentageOfImmobilePersistencePairs, numberPairsInputDiagram, localToGlobal0, 
                  true, LearningRate, threadNumber_, EpochNumber, maxAndMinCoordinateNode[0], stopCondition);

  

  levelTimes[0] = timeLastLevel.getElapsedTime();

  //=========================
  //       Time display
  //=========================


  this->printMsg(
        "ExecuteMultiBlock | level number : " + std::to_string(levelTimes.size()) , debug::Priority::DETAIL);

  this->printMsg(
        "ExecuteMultiBlock  | Times : " ,  debug::Priority::DETAIL);
  for(size_t i = 0; i < levelTimes.size(); i++){
    this->printMsg(
        " Level n° : " + std::to_string(i) + " && times : " + std::to_string(levelTimes[i]) + " s",  debug::Priority::DETAIL);

  }

  //=======================================
  //        Disable omp_set_nested
  //=======================================

  if(nestedOptionWasDisabled){
    omp_set_nested(false);
  }

  return 0;
}
#endif


#ifdef TTK_ENABLE_TORCH
template <typename dataType, typename triangulationType>
int ttk::TopologicalOptimization::executeOneBlock(
  dataType *const inputScalars,
  dataType *const outputScalars,
  SimplexId *const inputOffsets,
  triangulationType *triangulation,
  ttk::Triangulation *triangulationOneBlock,
  ttk::DiagramType &constraintDiagram,
  int *const modificationNumber,
  int *const lastChange,
  SimplexId vertexNumber,
  std::vector<SimplexId> &pairOnEdgeToRemove,
  std::vector<float> &listTimePersistenceDiagram,
  std::vector<float> &listTimePersistenceDiagramClustering,
  std::vector<float> &listTimeBackPropagation,
  std::vector<float> &listAveragePercentageOfModifiedVertices, 
  std::vector<float> &listTimeInteration,
  std::vector<float> &listAveragePercentageOfImmobilePersistencePairs, 
  SimplexId &numberPairsInputDiagram, 
  std::vector<SimplexId> &localToGlobal, 
  bool lastBlock,
  float lr,
  int threadNumber,
  int epochNumber,
  std::vector<float> maxAndMinCoordinateNode,
  double InitStopCondition
  ) const {


  //===============================================================
  //   Initialization of threadNumber and epochNumber variables
  //===============================================================
  if(threadNumber == -1){
    threadNumber = threadNumber_;
  }

  if(ChooseLearningRate){
    lr = LearningRate; 
  }

  if(epochNumber == 0){
    epochNumber = EpochNumber;
  }

  double stoppingCondition = 0; 

  //=======================
  //    Copy input data
  //=======================
  std::vector<double> dataVector(vertexNumber);
  SimplexId * inputOffsetsCopie = inputOffsets;

  #ifdef TTK_ENABLE_OPENMP
  #pragma omp parallel for num_threads(threadNumber)
  #endif
  for(SimplexId k = 0; k < vertexNumber; ++k) {
    outputScalars[k] = inputScalars[k];
    dataVector[k] = inputScalars[k];
    if(std::isnan((double)outputScalars[k]))
      outputScalars[k] = 0;
  }

  std::vector<double> losses;
  std::fill(lastChange, lastChange + vertexNumber, -1);
  std::fill(modificationNumber, modificationNumber + vertexNumber, 0);
  std::vector<double> inputScalarsX(vertexNumber);

  //==================================
  //          SmoothingSelectif
  //==================================
  if(Method == 0){
    std::vector<double> smoothedScalars = dataVector;
    ttk::DiagramType currentConstraintDiagram = constraintDiagram; 
    std::vector<int64_t> listAllIndicesToChangeSmoothing(vertexNumber, 0);
    std::vector<std::vector<SimplexId>> pair2MatchedPair(currentConstraintDiagram.size(), std::vector<SimplexId>(2)); 
    std::vector<SimplexId> pairChangeMatchingPair(currentConstraintDiagram.size(), -1); 
    std::vector<std::vector<SimplexId>> pair2Delete(vertexNumber, std::vector<SimplexId>()); 
    std::vector<std::vector<SimplexId>> currentVertex2PairsCurrentDiagram(vertexNumber_, std::vector<SimplexId>()); 
    
    for(int it = 0; it < epochNumber; it++){

      this->printMsg(
      "ExecuteOneBlock | SmoothingSelectif - iteration n° " + std::to_string(it),  debug::Priority::PERFORMANCE);
      std::tuple<std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >,
                        std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >> indices =
                        getIndices(triangulation, inputOffsetsCopie, dataVector.data(), currentConstraintDiagram, vertexNumber, it,
                        listAllIndicesToChangeSmoothing, pair2MatchedPair, pair2Delete, pairChangeMatchingPair, numberPairsInputDiagram, pairOnEdgeToRemove, lastBlock, listTimePersistenceDiagram, listTimePersistenceDiagramClustering, listAveragePercentageOfImmobilePersistencePairs, currentVertex2PairsCurrentDiagram, threadNumber, localToGlobal, maxAndMinCoordinateNode, triangulationOneBlock);
      std::fill(listAllIndicesToChangeSmoothing.begin(), listAllIndicesToChangeSmoothing.end(), 0);

      //==========================================================================
      //    Retrieve the indices for the pairs that we want to send diagonally
      //==========================================================================
      double lossDeletePairs = 0; 

      std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double>> &indexToDelete = std::get<0>(indices);
      std::vector<int64_t> &indexBirthPairToDelete =std::get<0>(indexToDelete);
      std::vector<double> &targetValueBirthPairToDelete = std::get<1>(indexToDelete);
      std::vector<int64_t> &indexDeathPairToDelete = std::get<2>(indexToDelete);
      std::vector<double> &targetValueDeathPairToDelete = std::get<3>(indexToDelete);

      this->printMsg(
      "ExecuteOneBlock | SmoothingSelectif - Number of pairs to delete : " + std::to_string(indexBirthPairToDelete.size()) ,  debug::Priority::DETAIL);

      if(indexBirthPairToDelete.size() == indexDeathPairToDelete.size()){
        // double lossDeletePairs = 0; 
        for(size_t i = 0; i <  indexBirthPairToDelete.size(); i++) {
          lossDeletePairs += std::pow(dataVector[indexBirthPairToDelete[i]] - targetValueBirthPairToDelete[i], 2) + std::pow(dataVector[indexDeathPairToDelete[i]] - targetValueDeathPairToDelete[i], 2); 
          SimplexId indexMax = indexBirthPairToDelete[i]; 
          SimplexId indexSelle = indexDeathPairToDelete[i]; 

          if(!(FinePairManagement == 2) && !(FinePairManagement == 1)){
            smoothedScalars[indexMax] = smoothedScalars[indexMax] - EpsilonPenalisation * 2 * (smoothedScalars[indexMax]-targetValueBirthPairToDelete[i]); 
            smoothedScalars[indexSelle] = smoothedScalars[indexSelle] - EpsilonPenalisation * 2 * (smoothedScalars[indexSelle]-targetValueDeathPairToDelete[i]); 
            listAllIndicesToChangeSmoothing[indexMax] = 1; 
            listAllIndicesToChangeSmoothing[indexSelle] = 1; 
          }
          else if(FinePairManagement == 1){
            smoothedScalars[indexSelle] = smoothedScalars[indexSelle] - EpsilonPenalisation * 2 * (smoothedScalars[indexSelle]-targetValueDeathPairToDelete[i]); 
            listAllIndicesToChangeSmoothing[indexSelle] = 1; 
          }
          else if(FinePairManagement == 2){
            smoothedScalars[indexMax] = smoothedScalars[indexMax] - EpsilonPenalisation * 2 * (smoothedScalars[indexMax]-targetValueBirthPairToDelete[i]); 
            listAllIndicesToChangeSmoothing[indexMax] = 1; 
          }
        }
      }
      else{
        for(size_t i = 0; i <  indexBirthPairToDelete.size(); i++) {
          lossDeletePairs += std::pow(dataVector[indexBirthPairToDelete[i]] - targetValueBirthPairToDelete[i], 2); 
          SimplexId indexMax = indexBirthPairToDelete[i]; 

          if(!(FinePairManagement == 1)){
            smoothedScalars[indexMax] = smoothedScalars[indexMax] - EpsilonPenalisation * 2 * (smoothedScalars[indexMax]-targetValueBirthPairToDelete[i]); 
            listAllIndicesToChangeSmoothing[indexMax] = 1; 
          }
          else{ // FinePairManagement == 1
            continue; 
          }
        }

        for(size_t i = 0; i <  indexDeathPairToDelete.size(); i++) {
          lossDeletePairs += std::pow(dataVector[indexDeathPairToDelete[i]] - targetValueDeathPairToDelete[i], 2); 
          SimplexId indexSelle = indexDeathPairToDelete[i]; 

          if(!(FinePairManagement == 2)){
            smoothedScalars[indexSelle] = smoothedScalars[indexSelle] - EpsilonPenalisation * 2 * (smoothedScalars[indexSelle]-targetValueDeathPairToDelete[i]); 
            listAllIndicesToChangeSmoothing[indexSelle] = 1; 
          }
          else{ // FinePairManagement == 2
            continue; 
          }
        }
      }
      this->printMsg(
      "ExecuteOneBlock | SmoothingSelectif - Loss Delete Pairs : " + std::to_string(lossDeletePairs) , debug::Priority::PERFORMANCE);

      //==========================================================================
      //      Retrieve the indices for the pairs that we want to change
      //==========================================================================
      double lossChangePairs = 0; 


      std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double>> &indexToChange = std::get<1>(indices);
      std::vector<int64_t> &indexBirthPairToChange = std::get<0>(indexToChange);
      std::vector<double> &targetValueBirthPairToChange = std::get<1>(indexToChange);
      std::vector<int64_t> &indexDeathPairToChange = std::get<2>(indexToChange);
      std::vector<double> &targetValueDeathPairToChange = std::get<3>(indexToChange);

      for(size_t i = 0; i <  indexBirthPairToChange.size(); i++) {
        lossChangePairs += std::pow(dataVector[indexBirthPairToChange[i]] - targetValueBirthPairToChange[i], 2) + std::pow(dataVector[indexDeathPairToChange[i]] - targetValueDeathPairToChange[i], 2); 
      
        SimplexId indexMax = indexBirthPairToChange[i]; 
        SimplexId indexSelle = indexDeathPairToChange[i]; 

        smoothedScalars[indexMax] = smoothedScalars[indexMax] - EpsilonPenalisation * 2 * (smoothedScalars[indexMax]-targetValueBirthPairToChange[i]) ; 
        smoothedScalars[indexSelle] = smoothedScalars[indexSelle] - EpsilonPenalisation * 2 * (smoothedScalars[indexSelle]-targetValueDeathPairToChange[i]); 
        listAllIndicesToChangeSmoothing[indexMax] = 1; 
        listAllIndicesToChangeSmoothing[indexSelle] = 1; 
      }
      this->printMsg(
      "ExecuteOneBlock | SmoothingSelectif - Loss Change Pairs : " + std::to_string(lossChangePairs), debug::Priority::PERFORMANCE);

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber)
#endif
      for(SimplexId k = 0; k < vertexNumber; ++k) {
        double diff = smoothedScalars[k] - dataVector[k];
        if(std::abs(diff) > 1e-6) {
          lastChange[k] = it;
        }
      }

      dataVector = smoothedScalars; 

      //===================================
      //      Modified vertex number
      //===================================
      SimplexId modifiedVertexNumber = (SimplexId)(indexBirthPairToDelete.size() + indexDeathPairToDelete.size() + indexBirthPairToChange.size() + indexDeathPairToChange.size()); 
      float percentageOfModifiedVertices = (double)modifiedVertexNumber / vertexNumber;

      listAveragePercentageOfModifiedVertices.push_back(percentageOfModifiedVertices); 

      //==================================
      //          Stop Condition
      //==================================

      if((InitStopCondition != -1) && (it == 0)){
        stoppingCondition = InitStopCondition; 
      }
      else if(it == 0){
        stoppingCondition = CoefStopCondition * (lossDeletePairs + lossChangePairs); 
      }

      if (((lossDeletePairs + lossChangePairs ) <= stoppingCondition))
        break;

    }

    //============================================
    //              Update output data
    //============================================
    #ifdef TTK_ENABLE_OPENMP
    #pragma omp parallel for num_threads(threadNumber)
    #endif
    for(SimplexId k = 0; k < vertexNumber; ++k) {
      outputScalars[k] = dataVector[k];
    }
  }
  
  //=======================================
  //           Adam Optimization
  //=======================================
  else if(Method == 1){
    //=====================================================
    //          Initialization of model parameters
    //=====================================================
    torch::Tensor F = torch::from_blob(dataVector.data(), {SimplexId(dataVector.size())}, torch::dtype(torch::kFloat64)).to(torch::kFloat64);
    PersistenceDiagramGradientDescent model(F);

    torch::optim::Adam optimizer(model.parameters(), lr);

    //=======================================
    //            Optimization
    //=======================================

    ttk::DiagramType currentConstraintDiagram = constraintDiagram; 
    std::vector<std::vector<SimplexId>> pair2MatchedPair(currentConstraintDiagram.size(), std::vector<SimplexId>(2)); 
    std::vector<SimplexId> pairChangeMatchingPair(currentConstraintDiagram.size(), -1); 
    std::vector<int64_t> listAllIndicesToChange(vertexNumber, 0);
    std::vector<std::vector<SimplexId>> pair2Delete(vertexNumber, std::vector<SimplexId>()); 
    std::vector<std::vector<SimplexId>> currentVertex2PairsCurrentDiagram(vertexNumber_, std::vector<SimplexId>()); 

    for(int i = 0; i < epochNumber; i++){

      this->printMsg(
      "ExecuteOneBlock | Adam - epoch : " + std::to_string(i) , debug::Priority::PERFORMANCE);

      ttk::Timer timeOneIteration; 

      // Update the tensor with the new optimized values
      tensorToVectorFast(model.X.to(torch::kDouble), inputScalarsX);

      // Retrieve the indices of the critical points that we must modify in order to match our current diagram to our target diagram.
      std::tuple<std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >,
                    std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double> >> indices =
                    getIndices(triangulation, inputOffsetsCopie, inputScalarsX.data(), currentConstraintDiagram, vertexNumber, i,
                    listAllIndicesToChange, pair2MatchedPair, pair2Delete, pairChangeMatchingPair, numberPairsInputDiagram, pairOnEdgeToRemove, lastBlock, listTimePersistenceDiagram, listTimePersistenceDiagramClustering, listAveragePercentageOfImmobilePersistencePairs, currentVertex2PairsCurrentDiagram, threadNumber, localToGlobal, maxAndMinCoordinateNode, triangulationOneBlock);

      std::fill(listAllIndicesToChange.begin(), listAllIndicesToChange.end(), 0);
      //==========================================================================
      //    Retrieve the indices for the pairs that we want to send diagonally
      //==========================================================================

      std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double>> &indexToDelete = std::get<0>(indices);
      torch::Tensor valueOfXDeleteBirth = torch::index_select(model.X, 0, torch::tensor(std::get<0>(indexToDelete)));
      auto valueDeleteBirth = torch::from_blob(std::get<1>(indexToDelete).data(), {static_cast<SimplexId>(std::get<1>(indexToDelete).size())}, torch::kDouble);
      torch::Tensor valueOfXDeleteDeath = torch::index_select(model.X, 0, torch::tensor(std::get<2>(indexToDelete)));
      auto valueDeleteDeath = torch::from_blob(std::get<3>(indexToDelete).data(), {static_cast<SimplexId>(std::get<3>(indexToDelete).size())}, torch::kDouble);

      torch::Tensor lossDeletePairs = torch::zeros({1}, torch::kFloat32);
      if(!(FinePairManagement == 2) && !(FinePairManagement == 1)){
        lossDeletePairs = torch::sum(torch::pow(valueOfXDeleteBirth-valueDeleteBirth, 2));
        lossDeletePairs = lossDeletePairs + torch::sum(torch::pow(valueOfXDeleteDeath-valueDeleteDeath, 2));
      }
      else if(FinePairManagement == 1){
        lossDeletePairs = torch::sum(torch::pow(valueOfXDeleteDeath-valueDeleteDeath, 2));
      }
      else if(FinePairManagement == 2){
        lossDeletePairs = torch::sum(torch::pow(valueOfXDeleteBirth-valueDeleteBirth, 2)); 
      }

      this->printMsg(
      "ExecuteOneBlock | Adam - Loss Delete Pairs : " + std::to_string(lossDeletePairs.item<double>()) , debug::Priority::PERFORMANCE);

      //==========================================================================
      //      Retrieve the indices for the pairs that we want to change
      //==========================================================================

      std::tuple<std::vector<int64_t>, std::vector<double>, std::vector<int64_t>, std::vector<double>> &indexToChange = std::get<1>(indices);
      torch::Tensor valueOfXChangeBirth = torch::index_select(model.X, 0, torch::tensor(std::get<0>(indexToChange)));
      auto valueChangeBirth = torch::from_blob(std::get<1>(indexToChange).data(), {static_cast<SimplexId>(std::get<1>(indexToChange).size())}, torch::kDouble);
      torch::Tensor valueOfXChangeDeath = torch::index_select(model.X, 0, torch::tensor(std::get<2>(indexToChange)));
      auto valueChangeDeath = torch::from_blob(std::get<3>(indexToChange).data(), {static_cast<SimplexId>(std::get<3>(indexToChange).size())}, torch::kDouble);

      auto lossChangePairs = torch::sum((torch::pow(valueOfXChangeBirth-valueChangeBirth, 2) + torch::pow(valueOfXChangeDeath-valueChangeDeath, 2)));

      this->printMsg(
      "ExecuteOneBlock | Adam - Loss Change Pairs : " + std::to_string(lossChangePairs.item<double>()), debug::Priority::PERFORMANCE);

      //====================================
      //      Definition of final loss
      //====================================

      auto loss = lossDeletePairs + lossChangePairs;
      
      this->printMsg(
      "ExecuteOneBlock | Adam - Loss : " + std::to_string(loss.item<double>()), debug::Priority::PERFORMANCE);

      //===================================
      //      Modified vertex number
      //===================================

      SimplexId modifiedVertexNumber = std::get<0>(indexToDelete).size() + std::get<2>(indexToDelete).size() + std::get<0>(indexToChange).size() + std::get<2>(indexToChange).size(); 
      float percentageOfModifiedVertices = (double)modifiedVertexNumber / vertexNumber; 

      listAveragePercentageOfModifiedVertices.push_back(percentageOfModifiedVertices); 
      //======================================================
      //     Find the number of times a vertex has changed
      //======================================================

      std::vector<double> concatenatedVector;

      concatenatedVector.reserve(std::get<0>(indexToDelete).size() + std::get<2>(indexToDelete).size() + 
                                std::get<0>(indexToChange).size() + std::get<2>(indexToChange).size());

      concatenatedVector.insert(concatenatedVector.end(), std::get<0>(indexToDelete).begin(), std::get<0>(indexToDelete).end());
      concatenatedVector.insert(concatenatedVector.end(), std::get<2>(indexToDelete).begin(), std::get<2>(indexToDelete).end());

      concatenatedVector.insert(concatenatedVector.end(), std::get<0>(indexToChange).begin(), std::get<0>(indexToChange).end());
      concatenatedVector.insert(concatenatedVector.end(), std::get<2>(indexToChange).begin(), std::get<2>(indexToChange).end());

      for(auto &indice : concatenatedVector){
        modificationNumber[static_cast<SimplexId>(indice)]++;
      }

      //==========================================
      //            Back Propagation
      //==========================================

      losses.push_back(loss.item<double>());

      ttk::Timer timeBackPropagation;
      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      listTimeBackPropagation.push_back(timeBackPropagation.getElapsedTime());

      //==========================================
      //         Modified index checking
      //==========================================

      // On trouve les indices qui ont changé
      std::vector<double> NewinputScalarsX(vertexNumber);
      tensorToVectorFast(model.X.to(torch::kDouble), NewinputScalarsX);

      #ifdef TTK_ENABLE_OPENMP
      #pragma omp parallel for num_threads(threadNumber)
      #endif
      for(SimplexId k = 0; k < vertexNumber; ++k){
        double diff = NewinputScalarsX[k] - inputScalarsX[k];
        if(diff != 0){
          listAllIndicesToChange[k] = 1;
          lastChange[k] = i;
        }
      }


      listTimeInteration.push_back(timeOneIteration.getElapsedTime());

      
      //=======================================
      //              Stop condition
      //=======================================
      if((InitStopCondition != -1) && (i == 0)){
        stoppingCondition = InitStopCondition; 
      }
      else if(i == 0){
        stoppingCondition = CoefStopCondition * loss.item<double>();  
      }

      if (loss.item<double>() < stoppingCondition)
        break;

    }

    //============================================
    //              Update output data
    //============================================
    #ifdef TTK_ENABLE_OPENMP
    #pragma omp parallel for num_threads(threadNumber)
    #endif
    for(SimplexId k = 0; k < vertexNumber; ++k) {
      outputScalars[k] = model.X[k].item().to<double>();
      if(std::isnan((double)outputScalars[k]))
        outputScalars[k] = 0;
    }
  }


  return 0; 
}
#endif
