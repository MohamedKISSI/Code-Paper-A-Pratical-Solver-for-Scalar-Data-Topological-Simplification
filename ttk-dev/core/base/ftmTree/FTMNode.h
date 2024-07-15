/// \ingroup base
/// \class ttk::FTMTree_MT
/// \author Charles Gueunet <charles.gueunet@lip6.fr>
/// \date June 2016.
///
///\brief TTK container representing a node of the FTMTree_MT
//
///\param dataType Data type of the input scalar field (char, float,
/// etc.).

#pragma once

#include <vector>

#include <Debug.h>
#include <functional>

#include "FTMDataTypes.h"

namespace ttk {
  namespace ftm {
    class Node {
      friend class FTMTree_MT;

    private:
      // mesh vertex where this node is
      SimplexId vertexId_;
      // For leaves, linkedNode is the saddle ending the persistence pair
      // For saddle, linked is the leaf starting the persistence pair in which
      // they are
      SimplexId linkedNode_;
      // link with superArc above and below
      std::vector<idSuperArc> vect_downSuperArcList_, vect_upSuperArcList_;

    public:
      // -----------------
      // CONSTRUCTOR
      // -----------------

      // This node will need to receive a vertex id before being printed
      Node() : vertexId_(nullVertex), linkedNode_(nullNodes) {
      }

      Node(SimplexId id, SimplexId linked)
        : vertexId_(id), linkedNode_(linked) {
      }

      // -----------------
      // ACCESSOR
      // ------------------

      // Vertex id

      inline SimplexId getVertexId() const {
        return vertexId_;
      }

      inline void setVertexId(SimplexId vertexId) {
        vertexId_ = vertexId;
      }

      // Linked node

      inline SimplexId getOrigin() const {
        return linkedNode_;
      }

      inline SimplexId getTermination() const {
        return linkedNode_;
      }

      inline void setOrigin(SimplexId linked) {
        linkedNode_ = linked;
      }

      inline void setTermination(SimplexId linked) {
        linkedNode_ = linked;
      }

      // vector arcs

      inline idSuperArc getNumberOfDownSuperArcs() const {
        return vect_downSuperArcList_.size();
      }

      inline idSuperArc getNumberOfUpSuperArcs() const {
        return vect_upSuperArcList_.size();
      }

      inline idSuperArc getNumberOfSuperArcs() const {
        return vect_upSuperArcList_.size() + vect_downSuperArcList_.size();
      }

      inline idSuperArc getDownSuperArcId(idSuperArc neighborId) const {
#ifndef TTK_ENABLE_KAMIKAZE
        if(neighborId >= vect_downSuperArcList_.size()) {
          std::cerr << "[Merge Tree:Node] get down on bad neighbor !";
          std::cerr << std::endl;
          return 0;
        }
#endif
        return vect_downSuperArcList_[neighborId];
      }

      inline idSuperArc getUpSuperArcId(idSuperArc neighborId) const {
#ifndef TTK_ENABLE_KAMIKAZE
        if(neighborId >= vect_upSuperArcList_.size()) {
          std::cerr << "[FTMTree_MT:Node] No SuperArc to access "
                    << static_cast<unsigned>(neighborId);
          std::cerr << std::endl;
        }
#endif
        if(vect_upSuperArcList_.size() == 0) {
          return nullSuperArc;
        }
        return vect_upSuperArcList_[neighborId];
      }

      inline void addDownSuperArcId(idSuperArc downSuperArcId) {
        vect_downSuperArcList_.emplace_back(downSuperArcId);
      }

      inline void addUpSuperArcId(idSuperArc upSuperArcId) {
        vect_upSuperArcList_.emplace_back(upSuperArcId);
      }

      inline idSuperArc clearDownSuperArcs() {
        idSuperArc const s = vect_downSuperArcList_.size();
        vect_downSuperArcList_.clear();
        return s;
      }

      inline idSuperArc clearUpSuperArcs() {
        idSuperArc const s = vect_upSuperArcList_.size();
        vect_upSuperArcList_.clear();
        return s;
      }

      // remove the i^th arc
      inline void removeDownSuperArcPos(idSuperArc i) {
        vect_downSuperArcList_[i] = vect_downSuperArcList_.back();
        vect_downSuperArcList_.pop_back();
      }

      // Find and remove the arc
      inline void removeDownSuperArc(idSuperArc idSa) {
        for(idSuperArc i = 0; i < vect_downSuperArcList_.size(); ++i) {
          if(vect_downSuperArcList_[i] == idSa) {
            vect_downSuperArcList_[i] = vect_downSuperArcList_.back();
            vect_downSuperArcList_.pop_back();

            return;
          }
        }
      }

      // Find and remove the arc
      inline void removeUpSuperArc(idSuperArc idSa) {
        for(idSuperArc i = 0; i < vect_upSuperArcList_.size(); ++i) {
          if(vect_upSuperArcList_[i] == idSa) {
            vect_upSuperArcList_[i] = vect_upSuperArcList_.back();
            vect_upSuperArcList_.pop_back();

            return;
          }
        }
      }

      void sortUpArcs(
        const std::function<bool(const idSuperArc, const idSuperArc)> &comp) {
        sort(vect_upSuperArcList_.begin(), vect_upSuperArcList_.end(), comp);
      }

      void sortDownArcs(
        const std::function<bool(const idSuperArc, const idSuperArc)> &comp) {
        sort(
          vect_downSuperArcList_.begin(), vect_downSuperArcList_.end(), comp);
      }
    };

  } // namespace ftm
} // namespace ttk
