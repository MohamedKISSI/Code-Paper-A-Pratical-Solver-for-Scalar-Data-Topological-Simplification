/// \ingroup base
/// \class ttk::SignedDistanceField
/// \author Julien Tierny <julien.tierny@lip6.fr>
/// \author Mohamed Amine Kissi <mohamed.kissi@lip6.fr>
/// \author Mathieu Pont <mathieu.pont@lip6.fr>
/// \date August 2023

#pragma once
#include <iostream>

// base code includes
#include "DataTypes.h"
#include "Geometry.h"
#include "ImplicitTriangulation.h"
#include "SurfaceGeometrySmoother.h"
#include "VisitedMask.h"
#include <BoundingVolumeHierarchy.h>
#include <Debug.h>
#include <KDTree.h>
#include <Triangulation.h>

namespace ttk {

  class SignedDistanceField : virtual public Debug {
  public:
    SignedDistanceField();

    template <typename dataType,
              typename triangulationType,
              typename triangulationType2>
    int execute(dataType *const outputScalars,
                triangulationType *const triangulation,
                triangulationType2 *const englobanteTriangulation,
                int *const edgeCrossing,
                int *const isInterior) const;

    inline int
      preconditionTriangulation(AbstractTriangulation *triangulation) const {
      if(triangulation) {
        triangulation->preconditionVertexNeighbors();
        triangulation->preconditionBoundaryEdges();
        triangulation->preconditionVertexTriangles();
        triangulation->preconditionVertexEdges();
        triangulation->preconditionEdgeTriangles();
        triangulation->preconditionBoundaryTriangles();
        triangulation->preconditionBoundaryVertices();
        triangulation->preconditionEdges();
      }
      return 0;
    }

    template <typename triangulationType>
    void findOutsideVertices(const SimplexId vertexId,
                             triangulationType *const englobanteTriangulation,
                             const std::vector<bool> &vertexIntersection,
                             int *const isInterior) const;

    inline int
      getNeighbor(unsigned int vertexId, unsigned int dim, int dir) const {
      unsigned int res[3] = {xResolution_, yResolution_, zResolution_};
      unsigned int coord[3]
        = {vertexId % res[0], (vertexId % (res[0] * res[1])) / res[0],
           vertexId / (res[0] * res[1])};

      if(not(static_cast<int>(coord[dim]) + dir >= 0
             and coord[dim] + dir < res[dim]))
        return -1;
      coord[dim] += dir;

      return coord[0] + coord[1] * res[0] + coord[2] * res[0] * res[1];
    }

    enum class VertexMarchingType { FAR, NARROW, FROZEN };

    template <typename dataType>
    void fastMarching(std::vector<bool> &vertexIntersection,
                      dataType *const distances,
                      int *const isInterior) const;

    template <typename dataType>
    void fastMarchingIterativeNode(std::vector<VertexMarchingType> &vertexType,
                                   dataType *const distances,
                                   int *const isInterior) const;

    template <typename dataType>
    void fastMarchingIterativeBand(std::vector<VertexMarchingType> &vertexType,
                                   dataType *const distances,
                                   int *const isInterior) const;

    template <typename dataType>
    void fastMarchingIterativePartialBand(
      std::vector<VertexMarchingType> &vertexType,
      dataType *const distances,
      int *const isInterior) const;

    template <typename dataType>
    dataType fastMarchingUpdatePoint(
      unsigned int vertexId,
      dataType *const distances,
      int *const isInterior,
      std::vector<VertexMarchingType> &vertexType) const;

    template <typename dataType>
    dataType fastMarchingUpdatePointOrderTwo(
      unsigned int vertexId,
      dataType *const distances,
      int *const isInterior,
      std::vector<VertexMarchingType> &vertexType) const;

    template <typename dataType>
    dataType fastMarchingUpdatePointOrderOne(
      unsigned int vertexId,
      dataType *const distances,
      int *const isInterior,
      std::vector<VertexMarchingType> &vertexType) const;

    template <typename dataType>
    bool fastMarchingSolveQuadratic(unsigned int vertexId,
                                    dataType a,
                                    dataType b,
                                    dataType c,
                                    int *const isInterior,
                                    dataType &out) const;

  protected:
    unsigned int xResolution_{1}, yResolution_{1}, zResolution_{1};
    std::array<double, 3> spacing_{1.0, 1.0, 1.0};
    std::array<double, 3> invSpacingSquared_{1.0, 1.0, 1.0};
    bool naiveMethod_ = false;
    bool naiveMethodAllEdges_ = false;
    bool fastMarching_ = false;
    int fastMarchingOrder_ = 1;
    bool fastMarchingIterativeBand_ = false;
    double fastMarchingIterativeBandRatio_ = 1.0;
  };

} // namespace ttk

template <typename dataType>
void ttk::SignedDistanceField::fastMarching(
  std::vector<bool> &vertexIntersection,
  dataType *const distances,
  int *const isInterior) const {
  std::vector<VertexMarchingType> vertexType(
    vertexIntersection.size(), VertexMarchingType::FAR);
  for(unsigned int i = 0; i < vertexIntersection.size(); ++i)
    if(vertexIntersection[i])
      vertexType[i] = VertexMarchingType::FROZEN;

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
  for(unsigned int i = 0; i < vertexType.size(); ++i) {
    // Compute narrow band
    if(vertexType[i] == VertexMarchingType::FAR) {
      for(int dim = 0; dim < 3; ++dim) {
        // each direction
        for(int j = -1; j < 2; j += 2) {
          int neighborId = getNeighbor(i, dim, j);
          if(neighborId != -1
             and vertexType[neighborId] == VertexMarchingType::FROZEN
             and vertexType[i] == VertexMarchingType::FAR) {
            vertexType[i] = VertexMarchingType::NARROW;
            distances[i]
              = fastMarchingUpdatePoint(i, distances, isInterior, vertexType);
          }
        }
      }
    }
  }

  if(fastMarchingIterativeBand_)
    if(fastMarchingIterativeBandRatio_ == 1.0)
      fastMarchingIterativeBand(vertexType, distances, isInterior);
    else
      fastMarchingIterativePartialBand(vertexType, distances, isInterior);
  else
    fastMarchingIterativeNode(vertexType, distances, isInterior);
}

template <typename dataType>
void ttk::SignedDistanceField::fastMarchingIterativeNode(
  std::vector<VertexMarchingType> &vertexType,
  dataType *const distances,
  int *const isInterior) const {
  std::cout << "fastMarchingIterativeNode" << std::endl;
  struct Compare {
    constexpr bool
      operator()(std::pair<int, dataType> const &a,
                 std::pair<int, dataType> const &b) const noexcept {
      return a.second > b.second;
    }
  };
  std::priority_queue<std::pair<int, dataType>,
                      std::vector<std::pair<int, dataType>>, Compare>
    heap;
  for(unsigned int i = 0; i < vertexType.size(); ++i)
    if(vertexType[i] == VertexMarchingType::NARROW)
      heap.push(std::make_pair(i, fabs(distances[i])));
  std::vector<int> toFreeze;

  while(!heap.empty()) {
    toFreeze.clear();
    std::pair<int, dataType> pair = heap.top();
    auto vertexId = std::get<0>(pair);
    auto value = std::get<1>(pair);
    heap.pop();
    if(not(value <= fabs(distances[vertexId]) + 1e-6
           and value >= fabs(distances[vertexId]) - 1e-6))
      continue;

    vertexType[vertexId] = VertexMarchingType::FROZEN;
    toFreeze.emplace_back(vertexId);

    bool done = false;
    while(!done) {
      if(!heap.empty()) {
        std::pair<int, dataType> l_pair = heap.top();
        auto l_vertexId = std::get<0>(l_pair);
        auto l_value = std::get<1>(l_pair);
        if(value <= l_value + 1e-6 and value >= l_value - 1e-6) {
          heap.pop();
          vertexType[l_vertexId] = VertexMarchingType::FROZEN;
          toFreeze.emplace_back(l_vertexId);
        } else
          done = true;
      } else
        done = true;
    }

    for(unsigned int k = 0; k < toFreeze.size(); ++k) {
      int id = toFreeze[k];

      for(unsigned int dim = 0; dim < 3; ++dim) {
        // each direction
        for(int j = -1; j < 2; j += 2) {
          int neighborId = getNeighbor(id, dim, j);
          if(neighborId != -1
             and vertexType[neighborId] != VertexMarchingType::FROZEN
             and (vertexType[neighborId] == VertexMarchingType::NARROW
                  or vertexType[neighborId] == VertexMarchingType::FAR)) {
            dataType d = fastMarchingUpdatePoint(
              neighborId, distances, isInterior, vertexType);
            if(d) {
              distances[neighborId] = d;
              heap.push(std::make_pair(neighborId, fabs(d)));
              if(vertexType[neighborId] == VertexMarchingType::FAR)
                vertexType[neighborId] = VertexMarchingType::NARROW;
            } else
              std::cout << "not d : " << d << std::endl;
          }
          // update the far point in the second order stencil
          // "jump" over a Frozen point if needed
          if(fastMarchingOrder_ == 2) {
            if(neighborId != -1
               and vertexType[neighborId] == VertexMarchingType::FROZEN) {
              int neighbor2Id = getNeighbor(id, dim, j * 2);
              if(neighbor2Id != -1
                 and vertexType[neighbor2Id] == VertexMarchingType::NARROW) {
                dataType d = fastMarchingUpdatePointOrderTwo(
                  neighbor2Id, distances, isInterior, vertexType);
                if(d) {
                  heap.push(std::make_pair(neighbor2Id, fabs(d)));
                  distances[neighbor2Id] = d;
                }
              }
            }
          }
        } // for each direction
      } // for each dimension
    }
  } // main loop of Fast Marching Method
}

template <typename dataType>
void ttk::SignedDistanceField::fastMarchingIterativeBand(
  std::vector<VertexMarchingType> &vertexType,
  dataType *const distances,
  int *const isInterior) const {
  std::cout << "fastMarchingIterativeBand" << std::endl;
  std::vector<int> toProcess;
  for(unsigned int i = 0; i < vertexType.size(); ++i)
    if(vertexType[i] == VertexMarchingType::NARROW)
      toProcess.emplace_back(i);

  while(!toProcess.empty()) {
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(unsigned int i = 0; i < toProcess.size(); ++i) {
      auto vertexId = toProcess[i];
      vertexType[vertexId] = VertexMarchingType::FROZEN;

      for(unsigned int dim = 0; dim < 3; ++dim) {
        for(int j = -1; j < 2; j += 2) {
          int neighborId = getNeighbor(vertexId, dim, j);
          if(neighborId != -1
             and vertexType[neighborId] == VertexMarchingType::FAR)
#pragma omp atomic write
            vertexType[neighborId] = VertexMarchingType::NARROW;
        }
      }
    }
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
    for(unsigned int i = 0; i < toProcess.size(); ++i)
      if(vertexType[i] == VertexMarchingType::NARROW)
        distances[i]
          = fastMarchingUpdatePoint(i, distances, isInterior, vertexType);
    toProcess.clear();
    for(unsigned int i = 0; i < vertexType.size(); ++i)
      if(vertexType[i] == VertexMarchingType::NARROW)
        toProcess.emplace_back(i);
  }
}

template <typename dataType>
void ttk::SignedDistanceField::fastMarchingIterativePartialBand(
  std::vector<VertexMarchingType> &vertexType,
  dataType *const distances,
  int *const isInterior) const {
  std::cout << "fastMarchingIterativePartialBand "
            << fastMarchingIterativeBandRatio_ << std::endl;
  struct Compare {
    constexpr bool
      operator()(std::pair<int, dataType> const &a,
                 std::pair<int, dataType> const &b) const noexcept {
      return a.second > b.second;
    }
  };
  std::priority_queue<std::pair<int, dataType>,
                      std::vector<std::pair<int, dataType>>, Compare>
    heap;
  std::vector<int> toProcess;
  std::vector<bool> inHeap(vertexType.size(), false);
  for(unsigned int i = 0; i < vertexType.size(); ++i) {
    if(vertexType[i] == VertexMarchingType::NARROW) {
      if(fastMarchingIterativeBandRatio_ == 1.0) {
        toProcess.emplace_back(i);
      } else {
        heap.push(std::make_pair(i, fabs(distances[i])));
        inHeap[i] = true;
      }
    }
  }

  while(!heap.empty() or !toProcess.empty()) {
    if(fastMarchingIterativeBandRatio_ != 1.0) {
      unsigned int size = std::max(
        1, static_cast<int>(heap.size() * fastMarchingIterativeBandRatio_));
      toProcess.resize(size);
      for(unsigned int i = 0; i < size; ++i) {
        std::pair<int, dataType> pair = heap.top();
        auto vertexId = std::get<0>(pair);
        heap.pop();
        inHeap[vertexId] = false;
        toProcess[i] = vertexId;
      }
    }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
    for(unsigned int i = 0; i < toProcess.size(); ++i) {
      auto vertexId = toProcess[i];
      vertexType[vertexId] = VertexMarchingType::FROZEN;

      for(unsigned int dim = 0; dim < 3; ++dim) {
        for(int j = -1; j < 2; j += 2) {
          int neighborId = getNeighbor(vertexId, dim, j);
          if(neighborId != -1
             and vertexType[neighborId] == VertexMarchingType::FAR)
#pragma omp atomic write
            vertexType[neighborId] = VertexMarchingType::NARROW;
        }
      }
    }
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
    for(unsigned int i = 0; i < vertexType.size(); ++i)
      if(vertexType[i] == VertexMarchingType::NARROW and not inHeap[i])
        distances[i]
          = fastMarchingUpdatePoint(i, distances, isInterior, vertexType);
    if(fastMarchingIterativeBandRatio_ == 1.0)
      toProcess.clear();
    for(unsigned int i = 0; i < vertexType.size(); ++i) {
      if(vertexType[i] == VertexMarchingType::NARROW and not inHeap[i]) {
        if(fastMarchingIterativeBandRatio_ == 1.0) {
          toProcess.emplace_back(i);
        } else {
          heap.push(std::make_pair(i, fabs(distances[i])));
          inHeap[i] = true;
        }
      }
    }
  }
}

template <typename dataType>
dataType ttk::SignedDistanceField::fastMarchingUpdatePoint(
  unsigned int vertexId,
  dataType *const distances,
  int *const isInterior,
  std::vector<VertexMarchingType> &vertexType) const {
  dataType d;
  if(fastMarchingOrder_ == 2)
    d = fastMarchingUpdatePointOrderTwo(
      vertexId, distances, isInterior, vertexType);
  else
    d = fastMarchingUpdatePointOrderOne(
      vertexId, distances, isInterior, vertexType);
  return d;
}

template <typename dataType>
dataType ttk::SignedDistanceField::fastMarchingUpdatePointOrderTwo(
  unsigned int vertexId,
  dataType *const distances,
  int *const isInterior,
  std::vector<VertexMarchingType> &vertexType) const {
  const dataType aa = 9.0 / 4.0;
  const dataType oneThird = 1.0 / 3.0;
  dataType a, b, c;
  a = b = c = 0;
  for(int dim = 0; dim < 3; dim++) {
    dataType value1 = std::numeric_limits<dataType>::max();
    dataType value2 = std::numeric_limits<dataType>::max();
    bool found1 = false, found2 = false;
    // each direction
    for(int j = -1; j < 2; j += 2) {
      int neighborId = getNeighbor(vertexId, dim, j);
      if(neighborId != -1
         and vertexType[neighborId] == VertexMarchingType::FROZEN
         and fabs(distances[neighborId]) < fabs(value1)) {
        value1 = distances[neighborId];
        found1 = true;
        int neighbor2Id = getNeighbor(vertexId, dim, j * 2);
        if(neighbor2Id != -1
           and vertexType[neighbor2Id] == VertexMarchingType::FROZEN
           and ((distances[neighbor2Id] <= value1 && value1 >= 0)
                || (distances[neighbor2Id] >= value1 && value1 <= 0))) {
          value2 = distances[neighbor2Id];
          found2 = true;
        }
      }
    }
    if(found2) {
      dataType tp = oneThird * (4 * value1 - value2);
      a += invSpacingSquared_[dim] * aa;
      b -= invSpacingSquared_[dim] * 2 * aa * tp;
      c += invSpacingSquared_[dim] * aa * pow(tp, 2);
    } else if(found1) {
      a += invSpacingSquared_[dim];
      b -= invSpacingSquared_[dim] * 2 * value1;
      c += invSpacingSquared_[dim] * pow(value1, 2);
    }
  }
  dataType out;
  if(not fastMarchingSolveQuadratic(vertexId, a, b, c, isInterior, out)) {
    // if the second order method fails, try the first order method instead
    return fastMarchingUpdatePointOrderOne(
      vertexId, distances, isInterior, vertexType);
  }
  return out;
}

template <typename dataType>
dataType ttk::SignedDistanceField::fastMarchingUpdatePointOrderOne(
  unsigned int vertexId,
  dataType *const distances,
  int *const isInterior,
  std::vector<VertexMarchingType> &vertexType) const {
  dataType a, b, c;
  a = b = c = 0;
  for(unsigned int dim = 0; dim < 3; ++dim) {
    dataType value = std::numeric_limits<dataType>::max();
    bool found = false;
    // each direction
    for(int j = -1; j < 2; j += 2) {
      int neighborId = getNeighbor(vertexId, dim, j);
      if(neighborId != -1
         and vertexType[neighborId] == VertexMarchingType::FROZEN
         and fabs(distances[neighborId]) < fabs(value)) {
        value = distances[neighborId];
        found = true;
      }
    }
    if(found) {
      a += invSpacingSquared_[dim];
      b -= invSpacingSquared_[dim] * 2 * value;
      c += invSpacingSquared_[dim] * pow(value, 2);
    }
  }
  dataType out;
  if(not fastMarchingSolveQuadratic(vertexId, a, b, c, isInterior, out)) {
    // if the quadratic equation can't be solved, use the
    // position of the minimum as a more reasonable approximation
    // std::cout << "cant solve" << std::endl;
    return -b / (2.0 * a);
  }
  return out;
}

template <typename dataType>
bool ttk::SignedDistanceField::fastMarchingSolveQuadratic(unsigned int vertexId,
                                                          dataType a,
                                                          dataType b,
                                                          dataType c,
                                                          int *const isInterior,
                                                          dataType &out) const {
  c -= 1;
  dataType det = pow(b, 2) - 4 * a * c;
  if(det >= 0) {
    if(isInterior[vertexId] == 0) {
      out = (-b + sqrt(det)) / 2.0 / a;
    } else {
      out = (-b - sqrt(det)) / 2.0 / a;
    }
    return true;
  }
  return false;
}

template <typename triangulationType>
void ttk::SignedDistanceField::findOutsideVertices(
  const SimplexId startVertexId,
  triangulationType *const englobanteTriangulation,
  const std::vector<bool> &vertexIntersection,
  int *const isInterior) const {
  std::stack<SimplexId> vertexStack;
  std::vector<bool> outsideVertexSelected(
    englobanteTriangulation->getNumberOfVertices(), false);

  vertexStack.push(startVertexId);
  outsideVertexSelected[startVertexId] = 1;
  isInterior[startVertexId] = 0;

  int cpt = 1;
  while(!vertexStack.empty()) {
    SimplexId vertexId = vertexStack.top();
    vertexStack.pop();

    if(vertexIntersection[vertexId])
      continue;

    std::vector<SimplexId> neighbors;
    if(naiveMethod_ and naiveMethodAllEdges_) {
      int vertexNeighborNumber
        = englobanteTriangulation->getVertexNeighborNumber(vertexId);
      neighbors.resize(vertexNeighborNumber);
      for(int i = 0; i < vertexNeighborNumber; i++) {
        SimplexId neighborId = 0;
        englobanteTriangulation->getVertexNeighbor(vertexId, i, neighborId);
        neighbors[i] = neighborId;
      }
    } else {
      for(unsigned int dim = 0; dim < 3; ++dim) {
        for(int j = -1; j < 2; j += 2) {
          int neighborId = getNeighbor(vertexId, dim, j);
          if(neighborId != -1)
            neighbors.emplace_back(neighborId);
        }
      }
    }
    for(const auto &neighborId : neighbors) {
      if(not outsideVertexSelected[neighborId]
         and not vertexIntersection[vertexId]) {
        vertexStack.push(neighborId);
        outsideVertexSelected[neighborId] = true;
        isInterior[neighborId] = 0;
        cpt++;
      }
    }
  }
  std::cout << "number of outside vertices = " << cpt << std::endl;
}

template <typename dataType,
          typename triangulationType,
          typename triangulationType2>
int ttk::SignedDistanceField::execute(
  dataType *const outputScalars,
  triangulationType *const triangulation,
  triangulationType2 *const englobanteTriangulation,
  int *const edgeCrossing,
  int *const isInterior) const {

  Timer t;

  auto noVertices = englobanteTriangulation->getNumberOfVertices();
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < noVertices; i++) {
    isInterior[i] = 1;
    edgeCrossing[i] = 0;
  }

  // ======================================================
  // === BVH
  // ======================================================
  Timer t_bvh;
  SimplexId vertexNumber = triangulation->getNumberOfVertices();
  std::vector<float> bvhCoordinates(3 * vertexNumber);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int vertexId = 0; vertexId < vertexNumber; vertexId++) {
    float x = 0, y = 0, z = 0;
    triangulation->getVertexPoint(vertexId, x, y, z);
    bvhCoordinates[vertexId * 3 + 0] = x;
    bvhCoordinates[vertexId * 3 + 1] = y;
    bvhCoordinates[vertexId * 3 + 2] = z;
  }

  SimplexId triangleNumber = triangulation->getNumberOfTriangles();
  std::vector<SimplexId> bvhConnectivityList(triangleNumber * 3);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int triangleId = 0; triangleId < triangleNumber; triangleId++) {
    SimplexId vertexTriangleIdA;
    triangulation->getTriangleVertex(triangleId, 0, vertexTriangleIdA);
    SimplexId vertexTriangleIdB;
    triangulation->getTriangleVertex(triangleId, 1, vertexTriangleIdB);
    SimplexId vertexTriangleIdC;
    triangulation->getTriangleVertex(triangleId, 2, vertexTriangleIdC);
    bvhConnectivityList[triangleId * 3 + 0] = vertexTriangleIdA;
    bvhConnectivityList[triangleId * 3 + 1] = vertexTriangleIdB;
    bvhConnectivityList[triangleId * 3 + 2] = vertexTriangleIdC;
  }
  BoundingVolumeHierarchy<SimplexId> bvh(bvhCoordinates.data(),
                                         bvhConnectivityList.data(),
                                         static_cast<size_t>(triangleNumber));

  /*for(int vertexId = 0; vertexId < noVertices; vertexId++) {
    float x = 0, y = 0, z = 0;
    englobanteTriangulation->getVertexPoint(vertexId, x, y, z);
    std::cout << vertexId << " : " << x << " " << y << " " << z << std::endl;
  }
  for(int edgeId = 0; edgeId < englobanteTriangulation->getNumberOfEdges();
      edgeId++) {
    SimplexId vertexIdA;
    englobanteTriangulation->getEdgeVertex(edgeId, 0, vertexIdA);
    // Point A;
    // englobanteTriangulation->getVertexPoint(vertexIdA, A.x, A.y, A.z);
    SimplexId vertexIdB;
    englobanteTriangulation->getEdgeVertex(edgeId, 1, vertexIdB);
    // Point B;
    // englobanteTriangulation->getVertexPoint(vertexIdB, B.x, B.y, B.z);
    std::cout << edgeId << " : " << vertexIdA << " " << vertexIdB << std::endl;
  }*/
  this->printMsg("Build BVH", 1.0, t_bvh.getElapsedTime(), this->threadNumber_);

  // ===================================
  // === Find Segments Intersecting Triangles
  // ===================================
  Timer t_intersect;

  std::vector<bool> vertexIntersection(noVertices, false);
  std::vector<std::vector<unsigned int>> verticesIntersected;

  if(not naiveMethod_) {
    unsigned int noOrigins = yResolution_ * zResolution_
                             + xResolution_ * zResolution_
                             + xResolution_ * yResolution_;
    verticesIntersected.resize(noOrigins);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
    for(unsigned int i = 0; i < noOrigins; ++i) {
      // Get origin vertex id
      int vertexId, dirDimension;
      if(i < yResolution_ * zResolution_) {
        // O_x = {a * r_x | a in N / r_y r_z N}
        vertexId = i * xResolution_;
        dirDimension = 0;
      } else if(i < yResolution_ * zResolution_ + xResolution_ * zResolution_) {
        // O_y = {a * r_x * r_y + b | a in N / r_z N, a in N / r_x N}
        unsigned int ind = i - yResolution_ * zResolution_;
        vertexId = (int)(ind / xResolution_) * xResolution_ * yResolution_
                   + ind % xResolution_;
        dirDimension = 1;
      } else {
        // O_z = {a | a in N / r_x r_y N}
        vertexId
          = i - (yResolution_ * zResolution_ + xResolution_ * zResolution_);
        dirDimension = 2;
      }

      // Use BVH to find triangles on this ray direction
      float ray_origin[3];
      englobanteTriangulation->getVertexPoint(
        vertexId, ray_origin[0], ray_origin[1], ray_origin[2]);
      float ray_dir[3] = {0, 0, 0};
      ray_dir[dirDimension] = spacing_[dirDimension];
      Ray ray(ray_dir, ray_origin);
      std::vector<int> triangles;
      std::vector<float> distances;
      bool wasHit = bvh.intersect(ray, bvhConnectivityList.data(),
                                  bvhCoordinates.data(), triangles, distances);
      // For each edge of this ray check if it is intersecting a triangle
      if(wasHit) {
        unsigned int rayNoVertices
          = (dirDimension == 0
               ? xResolution_
               : (dirDimension == 1 ? yResolution_ : zResolution_));
        std::vector<unsigned int> verticesList(rayNoVertices);
        for(unsigned int j = 0; j < verticesList.size(); ++j) {
          if(dirDimension == 0) {
            verticesList[j] = vertexId + j;
          } else if(dirDimension == 1) {
            verticesList[j] = vertexId + j * xResolution_;
          } else {
            verticesList[j] = vertexId + j * xResolution_ * yResolution_;
          }
        }
        for(auto &distance : distances) {
          unsigned int ind = distance;
          if(ind + 1 >= verticesList.size())
            --ind;
          auto vertexIdA = verticesList[ind];
          auto vertexIdB = verticesList[ind + 1];
          verticesIntersected[i].emplace_back(vertexIdA);
          verticesIntersected[i].emplace_back(vertexIdB);
        }
      }
    }
  } else {
    auto noEdges = englobanteTriangulation->getNumberOfEdges();
    if(not naiveMethodAllEdges_)
      noEdges = (xResolution_ - 1) * yResolution_ * zResolution_
                + (yResolution_ - 1) * xResolution_ * zResolution_
                + (zResolution_ - 1) * xResolution_ * yResolution_;
    verticesIntersected.resize(noEdges);
#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_) schedule(dynamic)
#endif
    for(int edgeId = 0; edgeId < noEdges; edgeId++) {
      // break;
      // Get Points
      SimplexId vertexIdA;
      englobanteTriangulation->getEdgeVertex(edgeId, 0, vertexIdA);
      float ray_origin[3];
      englobanteTriangulation->getVertexPoint(
        vertexIdA, ray_origin[0], ray_origin[1], ray_origin[2]);

      SimplexId vertexIdB;
      englobanteTriangulation->getEdgeVertex(edgeId, 1, vertexIdB);
      float ray_dir[3];
      englobanteTriangulation->getVertexPoint(
        vertexIdB, ray_dir[0], ray_dir[1], ray_dir[2]);

      for(unsigned int i = 0; i < 3; ++i)
        ray_dir[i] -= ray_origin[i];

      // BVH
      Ray ray(ray_dir, ray_origin);
      bool wasHit = false;
      int triIdx;
      float distance;
      wasHit = bvh.intersect(ray, bvhConnectivityList.data(),
                             bvhCoordinates.data(), &triIdx, &distance, true);
      if(wasHit) {
        verticesIntersected[edgeId].emplace_back(vertexIdA);
        verticesIntersected[edgeId].emplace_back(vertexIdB);
      }
    }
  }
  // Can not be parallelized because some vertices can appear more than once
  for(unsigned int i = 0; i < verticesIntersected.size(); ++i) {
    for(auto &vertexId : verticesIntersected[i]) {
      vertexIntersection[vertexId] = true;
      edgeCrossing[vertexId] = 1;
    }
  }
  /*for(unsigned int i = 0; i < vertexIntersection.size(); ++i)
    if(vertexIntersection[i])
      std::cout << i << std::endl;*/
  this->printMsg("Find intersection edges", 1.0, t_intersect.getElapsedTime(),
                 this->threadNumber_);

  // ===================================
  // === Find Outside vertices
  // ===================================
  Timer t_outside;
  // We search for an outside vertex
  SimplexId outsideVertex = std::numeric_limits<SimplexId>::max();
  for(unsigned int i = 0; i < vertexIntersection.size(); i++) {
    if(not vertexIntersection[i]) {
      outsideVertex = i;
      std::cout << "Premier vertex dehors trouvé : " << outsideVertex
                << std::endl;
      break;
    }
  }
  findOutsideVertices(
    outsideVertex, englobanteTriangulation, vertexIntersection, isInterior);
  this->printMsg("Find outside vertices", 1.0, t_outside.getElapsedTime(),
                 this->threadNumber_);

  // ======================================================
  // === Signed Distance Field
  // ======================================================
  // Find the distance between each point of the grid and the nearest point in
  // the domain
  Timer t_distance;
  std::cout << "debut calcul distance " << std::endl;

  std::vector<int> verticesCrossing;
  if(fastMarching_) {
    for(unsigned int i = 0; i < vertexIntersection.size(); ++i)
      if(vertexIntersection[i])
        verticesCrossing.emplace_back(i);
    noVertices = verticesCrossing.size();
  }

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for num_threads(threadNumber_)
#endif
  for(int i = 0; i < noVertices; i++) {
    auto vertexId = i;
    if(fastMarching_)
      vertexId = verticesCrossing[i];

    ttk::SurfaceGeometrySmoother smoother;

    float x = 0, y = 0, z = 0;
    englobanteTriangulation->getVertexPoint(vertexId, x, y, z);

    ttk::SurfaceGeometrySmoother::Point point;
    point[0] = x;
    point[1] = y;
    point[2] = z;

    std::vector<float> dists(triangulation->getNumberOfVertices());
    ttk::SurfaceGeometrySmoother::ProjectionInput projectionInput{
      static_cast<size_t>(vertexId), point,
      smoother.getNearestSurfaceVertex(point, dists, *triangulation)};

    std::vector<SimplexId> visitedTriangles{};
    std::vector<bool> trianglesTested(
      triangulation->getNumberOfTriangles(), false);

    ttk::VisitedMask vm{trianglesTested, visitedTriangles};
    std::vector<float> dists2(triangulation->getNumberOfVertices());
    std::stack<ttk::SimplexId> trianglesToTest;
    bool reverseProjection = false;

    ttk::SurfaceGeometrySmoother::ProjectionResult projectionResult
      = smoother.findProjection(projectionInput, vm, dists2, trianglesToTest,
                                reverseProjection, *englobanteTriangulation,
                                *triangulation);

    ttk::SurfaceGeometrySmoother::Point projectedPoint = projectionResult.pt;

    // We calculate the distance
    std::vector<double> p0({x, y, z});
    std::vector<double> p1(
      {projectedPoint[0], projectedPoint[1], projectedPoint[2]});
    double distance = ttk::Geometry::distance(p0.data(), p1.data(), 3);

    outputScalars[vertexId] = (isInterior[vertexId] ? -1.0 : 1.0) * distance;
  }
  this->printMsg(
    "Distance", 1.0, t_distance.getElapsedTime(), this->threadNumber_);

  if(fastMarching_) {
    Timer t_marching;
    std::cout << "debut fast marching " << std::endl;
    fastMarching(vertexIntersection, outputScalars, isInterior);
    this->printMsg(
      "Fast marching", 1.0, t_marching.getElapsedTime(), this->threadNumber_);
  }

  this->printMsg(
    "Signed distance field", 1.0, t.getElapsedTime(), this->threadNumber_);

  return 0;
}
