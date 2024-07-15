#include <PersistentSimplexPairs.h>

ttk::PersistentSimplexPairs::PersistentSimplexPairs() {
  this->setDebugMsgPrefix("PersistentSimplexPairs");
}

ttk::SimplexId ttk::PersistentSimplexPairs::eliminateBoundaries(
  const Simplex &c,
  VisitedMask &boundary,
  const std::vector<SimplexId> &filtOrder,
  const std::vector<Simplex> &partners) const {

  this->addCellBoundary(c, boundary);

  while(!boundary.visitedIds_.empty()) {
    // youngest cell on boundary
    const auto tau{*std::max_element(
      boundary.visitedIds_.begin(), boundary.visitedIds_.end(),
      [&filtOrder, &c, this](const SimplexId a, const SimplexId b) {
        return filtOrder[this->getCellId(c.dim_ - 1, a)]
               < filtOrder[this->getCellId(c.dim_ - 1, b)];
      })};
    const auto &partnerTau{partners[this->getCellId(c.dim_ - 1, tau)]};
    if(partnerTau.dim_ == -1 || partnerTau.id_ == -1) {
      return tau;
    }
    this->addCellBoundary(partnerTau, boundary);
  }

  return -1;
}

int ttk::PersistentSimplexPairs::pairCells(
  std::vector<PersistencePair> &pairs,
  std::array<std::vector<bool>, 3> &boundaries,
  const std::vector<Simplex> &filtration,
  const std::vector<SimplexId> &filtOrder) const {

  // for VisitedMask
  std::vector<SimplexId> visitedIds{};
  // paired simplices
  std::vector<Simplex> partners(filtration.size());

  Timer tm{};

  this->printMsg("Computing pairs", 0, 0, 1, ttk::debug::LineMode::REPLACE);

  for(size_t i = 0; i < filtration.size(); ++i) {

    const auto &c{filtration[i]};

    // skip vertices
    if(c.dim_ == 0) {
      continue;
    }

    // store the boundary cells
    VisitedMask vm{boundaries[c.dim_ - 1], visitedIds};

    const auto partner = this->eliminateBoundaries(c, vm, filtOrder, partners);
    if(partner != -1) {
      const auto &pc{
        filtration[filtOrder[this->getCellId(c.dim_ - 1, partner)]]};
      partners[c.cellId_] = pc;
      partners[pc.cellId_] = c;

      // only record pairs with non-null persistence
      if(c.vertsOrder_[0] != pc.vertsOrder_[0]) {
        pairs.emplace_back(partner, c.id_, c.dim_ - 1);
      }
    }

    if(filtration.size() > 10 && i % (filtration.size() / 10) == 0) {
      this->printMsg(
        "Computing pairs",
        std::round(10 * i / static_cast<float>(filtration.size())) / 10.0f,
        tm.getElapsedTime(), 1, ttk::debug::LineMode::REPLACE);
    }
  }

  const auto nRegPairs{pairs.size()};

  this->printMsg("Computed " + std::to_string(nRegPairs) + " regular pair"
                   + (nRegPairs > 1 ? "s" : ""),
                 1.0, tm.getElapsedTime(), 1);

  // get infinite pairs
  for(SimplexId i = 0; i < this->nVerts_; ++i) {
    if(partners[i].id_ == -1) {
      pairs.emplace_back(i, -1, 0);
    }
  }
  for(SimplexId i = 0; i < this->nEdges_; ++i) {
    if(partners[i + this->nVerts_].id_ == -1) {
      pairs.emplace_back(i, -1, 1);
    }
  }
  for(SimplexId i = 0; i < this->nTri_; ++i) {
    if(partners[i + this->nVerts_ + this->nEdges_].id_ == -1) {
      pairs.emplace_back(i, -1, 2);
    }
  }
  for(SimplexId i = 0; i < this->nTetra_; ++i) {
    if(partners[i + this->nVerts_ + this->nEdges_ + this->nTri_].id_ == -1) {
      pairs.emplace_back(i, -1, 3);
    }
  }

  const auto nInfPairs{pairs.size() - nRegPairs};

  this->printMsg("Detected " + std::to_string(nInfPairs) + " infinite pair"
                 + (nInfPairs > 1 ? "s" : ""));

  return 0;
}
