/// \ingroup base
/// \class ttk::FTMTree
/// \author Charles Gueunet <charles.gueunet@lip6.fr>
/// \date Dec 2016.
///
///\brief TTK processing package that efficiently computes the
/// contour tree of scalar data and more
/// (data segmentation, topological simplification,
/// persistence diagrams, persistence curves, etc.).
///
///\param dataType Data type of the input scalar field (char, float,
/// etc.).

#include <iterator>
#include <string>

#include "FTMTree_CT.h"

using namespace std;
using namespace ttk;

using namespace ftm;

FTMTree_CT::FTMTree_CT(const std::shared_ptr<Params> &params,
                       const std::shared_ptr<Scalars> &scalars)
  : FTMTree_MT(params, scalars, TreeType::Contour),
    jt_(params, scalars, TreeType::Join),
    st_(params, scalars, TreeType::Split) {
  this->setDebugMsgPrefix("FTMTree_CT");
}

int FTMTree_CT::combine() {
  Timer stepTime;
  queue<pair<bool, idNode>> growingNodes, remainingNodes;

  const bool DEBUG = false;

  // Reserve
  mt_data_.nodes->reserve(jt_.getNumberOfNodes());
  mt_data_.superArcs->reserve(jt_.getNumberOfSuperArcs() + 2);
  mt_data_.leaves.reserve(jt_.getNumberOfLeaves() + st_.getNumberOfLeaves());

  // Add JT & ST Leaves to growingNodes

  // Add leves to growing nodes
  const auto &nbSTLeaves = st_.getNumberOfLeaves();
  if(nbSTLeaves > 1) {
    for(idNode n = 0; n < nbSTLeaves; ++n) {
      const auto &nId = st_.getLeave(n);
      growingNodes.emplace(false, nId);
    }
  } else {
    move(jt_);
    return 0;
  }

  // count how many leaves can be added, if more than one : ok!
  const auto &nbJTLeaves = jt_.getNumberOfLeaves();
  if(nbJTLeaves > 1) {
    for(idNode n = 0; n < nbJTLeaves; ++n) {
      const auto &nId = jt_.getLeave(n);
      growingNodes.emplace(true, nId);
    }
  } // else can't clone, not same up and down

  if(DEBUG) {
    cout << "growingNodes : " << growingNodes.size()
         << " in : " << stepTime.getElapsedTime() << endl;
  }

  // Warning, have a reserve here, can't make it at the begnining, need build
  // output
  mt_data_.leaves.reserve(jt_.getLeaves().size() + st_.getLeaves().size());
  mt_data_.superArcs->reserve(jt_.getNumberOfSuperArcs());
  mt_data_.nodes->reserve(jt_.getNumberOfNodes());

  if(growingNodes.empty()) {
    cout << "[FTMTree_CT::combine ] Nothing to combine" << endl;
  }

#ifdef TTK_ENABLE_FTM_TREE_DUAL_QUEUE_COMBINE
  do {
    while(!remainingNodes.empty()) {
      bool isJT;
      idNode currentNodeId;
      FTMTree_MT *xt;

      tie(isJT, currentNodeId) = remainingNodes.front();
      remainingNodes.pop();
      if(isJT) {
        // node come from jt
        xt = &jt_;
      } else {
        // node come from st
        xt = &st_;
      }
      if(xt->getNode(currentNodeId)->getNumberOfUpSuperArcs() == 1) {
        growingNodes.emplace(isJT, currentNodeId);
        if(DEBUG) {
          cout << "repush in growing:" << isJT
               << "::" << xt->printNode(currentNodeId) << endl;
        }
      }
    }
#endif

    while(!growingNodes.empty()) {
      idNode currentNodeId;
      bool isJT;

      // INFO QUEUE

      tie(isJT, currentNodeId) = growingNodes.front();
      growingNodes.pop();

      FTMTree_MT *xt = (isJT) ? &jt_ : &st_;
      FTMTree_MT *yt = (isJT) ? &st_ : &jt_;

      // INFO JT / ST

      // i <- Get(Q)
      const Node *currentNode = xt->getNode(currentNodeId);

      if(DEBUG) {
        if(xt == &jt_)
          cout << endl << "JT ";
        else
          cout << endl << "ST ";
        cout << "node : " << currentNode->getVertexId() << endl;
      }

      // "choose a non-root leaf that is not a split in ST" so we ignore such
      // nodes
      if(currentNode->getNumberOfUpSuperArcs() == 0) {
        if(DEBUG) {
          cout << " ignore already processed" << endl;
        }
        continue;
      }

      idNode const correspondingNodeId
        = yt->getCorrespondingNodeId(currentNode->getVertexId());

      if(yt->getNode(correspondingNodeId)->getNumberOfDownSuperArcs() > 1) {
        if(DEBUG) {
          cout << "put remain:" << isJT << "::" << xt->printNode(currentNodeId)
               << endl;
          cout << " which is in yt : " << yt->printNode(correspondingNodeId)
               << endl;
        }
#ifdef TTK_ENABLE_FTM_TREE_DUAL_QUEUE_COMBINE
        remainingNodes.emplace(isJT, currentNodeId);
#else
      growingNodes.emplace(isJT, currentNodeId);
#endif
        continue;
      }

      // NODES IN CT

      idNode node1, node2;
      SimplexId const curVert = currentNode->getVertexId();
      // NODE1
      if(isCorrespondingNode(curVert)) {
        // already a node in the tree
        node1 = getCorrespondingNodeId(curVert);
      } else {
        // create a new node
        node1 = makeNode(currentNode);

        // check if leaf
        if(!currentNode->getNumberOfDownSuperArcs()
           || !currentNode->getNumberOfUpSuperArcs())
          mt_data_.leaves.emplace_back(node1);
      }

      // j <- GetAdj(XT, i)
      idSuperArc const curUpArc = currentNode->getUpSuperArcId(0);
      idNode const parentId = xt->getSuperArc(curUpArc)->getUpNodeId();
      const Node *parentNode = xt->getNode(parentId);

      if(DEBUG) {
        cout << " parent node :" << parentNode->getVertexId() << endl;
      }

      SimplexId const parVert = parentNode->getVertexId();
      // NODE2
      if(isCorrespondingNode(parVert)) {
        // already a node in the tree
        node2 = getCorrespondingNodeId(parVert);
      } else {
        // create a new node
        node2 = makeNode(parentNode);
        if(!parentNode->getNumberOfUpSuperArcs())
          mt_data_.leaves.emplace_back(node2);
      }

      // CREATE ARC

      idSuperArc const processArc = currentNode->getUpSuperArcId(0);

      // create the arc in in the good direction
      // and add it to crossing if needed
      idSuperArc createdArc;
      if(scalars_->isLower(
           currentNode->getVertexId(),
           parentNode->getVertexId())) { // take care of the order
        createdArc = makeSuperArc(node1, node2);
      } else {
        createdArc = makeSuperArc(node2, node1);
      }

      // Segmentation
      if(params_->segm) {
        createCTArcSegmentation(createdArc, isJT, processArc);
      }

      if(DEBUG) {
        cout << "create arc : " << printArc(createdArc) << endl;
      }

      // DEL NODES

      // DelNode(XT, i)
      {
        if(DEBUG) {
          cout << " delete xt (" << (xt == &jt_) << ") ";
          cout << "node :" << xt->printNode(currentNodeId) << endl;
        }

        xt->delNode(currentNodeId);
      }

      // DelNode(YT, i)
      {
        if(DEBUG) {
          cout << " delete yt (" << isJT << ") node :";
          cout << yt->printNode(correspondingNodeId) << endl;
        }

        yt->delNode(correspondingNodeId);
      }

      // PROCESS QUEUE

      if(parentNode->getNumberOfDownSuperArcs() == 0
         && parentNode->getNumberOfUpSuperArcs()) {
        growingNodes.emplace(isJT, parentId);

        if(DEBUG) {
          cout << "will see : " << parentNode->getVertexId() << endl;
        }
      }
    }
#ifdef TTK_ENABLE_FTM_TREE_DUAL_QUEUE_COMBINE
  } while(!remainingNodes.empty());
#endif

  if(DEBUG) {
    printTree2();
  }

  return 0;
}

void FTMTree_CT::createCTArcSegmentation(idSuperArc ctArc,
                                         const bool isJT,
                                         idSuperArc xtArc) {
  const FTMTree_MT *xt = (isJT) ? &jt_ : &st_;

  /*Here we prefer to create lots of small region, each arc having its own
   * segmentation with no overlap instead of having a same vertice in several
   * arc and using vert2tree to decide because we do not want to maintain
   * vert2tree information during the whole computation*/
  const list<Region> &xtRegions = xt->getSuperArc(xtArc)->getRegions();
  for(const Region &reg : xtRegions) {
    segm_it cur = reg.segmentBegin;
    segm_it const end = reg.segmentEnd;
    segm_it tmpBeg = reg.segmentBegin;
    // each element inside this region
    for(; cur != end; ++cur) {
      if(isCorrespondingNull(*cur)) {
        updateCorrespondingArc(*cur, ctArc);
      } else {
        // already set, we finish a region
        if(cur != tmpBeg) {
          getSuperArc(ctArc)->concat(tmpBeg, cur);
        }
        // if several contiguous vertices are discarded
        // cur will be equals to tmpBeg and we will not create empty regions
        tmpBeg = cur + 1;
      }
    }
    // close last region
    if(cur != tmpBeg) {
      getSuperArc(ctArc)->concat(tmpBeg, cur);
    }
  }
}

void FTMTree_CT::finalizeSegmentation() {
  Timer finSegmTime;
  const auto &nbArc = getNumberOfSuperArcs();

#ifdef TTK_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for(idSuperArc i = 0; i < nbArc; i++) {
    getSuperArc(i)->createSegmentation(scalars_.get());
  }

  printTime(finSegmTime, "post-process segm", 4);
}

void FTMTree_CT::insertNodes() {
  vector<idNode> const sortedJTNodes = jt_.sortedNodes(true);
  vector<idNode> const sortedSTNodes = st_.sortedNodes(true);

  for(const idNode &t : sortedSTNodes) {

    SimplexId const vertId = st_.getNode(t)->getVertexId();
    if(jt_.isCorrespondingNode(vertId)) {
      continue;
    }
    jt_.insertNode(st_.getNode(t));
  }

  for(const idNode &t : sortedJTNodes) {

    SimplexId const vertId = jt_.getNode(t)->getVertexId();
    if(st_.isCorrespondingNode(vertId)) {
      continue;
    }
    st_.insertNode(jt_.getNode(t));
  }
}
