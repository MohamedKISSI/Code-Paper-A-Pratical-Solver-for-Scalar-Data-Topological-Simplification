#pragma once

#include <array>
#include <tuple>
#include <vector>

#include <BaseClass.h>

namespace ttk {

  /**
   * @brief Critical Vertex
   */
  struct CriticalVertex {
    /** vertex id in domain */
    ttk::SimplexId id;
    /** critical type */
    ttk::CriticalType type;
    /** scalar field value */
    double sfValue;
    /** 3D coordinates in domain */
    std::array<float, 3> coords;
  };

  /**
   * @brief Persistence Pair
   */
  struct PersistencePair {
    /** pair birth */
    ttk::CriticalVertex birth;
    /** pair death */
    ttk::CriticalVertex death;
    /** pair dimension */
    ttk::SimplexId dim;
    /** to help distinguish homology classes with infinite persistence
        (connected components, topological handles or cavities in the
        domain) */
    bool isFinite;

    /**
     * @brief Order pairs according to their birth value
     */
    bool operator<(const PersistencePair &rhs) const {
      return this->birth.sfValue < rhs.birth.sfValue;
    }

    bool operator==(const PersistencePair& rhs) const {
      bool resultat = true;

      if((this->birth.id != rhs.birth.id) || (this->birth.type != rhs.birth.type) || (this->birth.sfValue != rhs.birth.sfValue) ||
        (this->birth.coords[0] != rhs.birth.coords[0]) || (this->birth.coords[1] != rhs.birth.coords[1]) || (this->birth.coords[2] != rhs.birth.coords[2])){
          resultat = false; 
      }

      if((this->death.id != rhs.death.id) || (this->death.type != rhs.death.type) || (this->death.sfValue != rhs.death.sfValue) ||
        (this->death.coords[0] != rhs.death.coords[0]) || (this->death.coords[1] != rhs.death.coords[1]) || (this->death.coords[2] != rhs.death.coords[2])){
          resultat = false; 
      }
      
      if((this->dim != rhs.dim) || (this->isFinite != rhs.isFinite)){
        resultat = false; 
      }

      return resultat;
    }

    /**
     * @brief Return the topological persistence of the pair
     */
    inline double persistence() const {
      return this->death.sfValue - this->birth.sfValue;
    }
  };

  /**
   * @brief Persistence Diagram type as a vector of Persistence pairs
   */
  using DiagramType = std::vector<PersistencePair>;

  /**
   * @brief Matching between two Persistence Diagram pairs
   */
  using MatchingType = std::tuple<int, /**< id of first matching pair */
                                  int, /**< id of second matching pair */
                                  double /**< matching cost */
                                  >;

}; // namespace ttk
