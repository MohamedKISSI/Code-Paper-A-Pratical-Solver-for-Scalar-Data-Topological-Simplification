#include <GabowTarjan.h>

#include <iostream>
#include <queue>
#include <vector>

bool ttk::GabowTarjan::DFS(int v) {
  if(v < 0)
    return true;

  // for every adjacent vertex u of v
  for(unsigned int i = 0; i < Connections[v].size(); ++i) {
    const int u = Connections[v][i];
    if(Layers[Pair[u] + 1] == Layers[v + 1] + 1) {
      if(DFS(Pair[u])) {
        Pair[u] = v;
        Pair[v] = u;
        return true;
      }
    }
  }

  Layers[v + 1] = -1;

  return false;
}

bool ttk::GabowTarjan::BFS() {
  std::queue<int> vertexQueue;

  // For every vertex v given by PersistencePairs1
  for(unsigned int v = 0; v < MaxSize; v++) {
    // If its not paired to vertex in PersistencePairs2
    if(Pair[v] < 0) {
      // Set its layer to 0 and put it in the queue
      Layers[v + 1] = 0;
      vertexQueue.push(v);
    } else {
      // Otherwise mark it as matched (set Layer to NIL)
      Layers[v + 1] = -1;
    }
  }

  // Set layer for NIL
  Layers[0] = -1;

  // Search the vertices in the queue
  while(!vertexQueue.empty()) {
    const int v = vertexQueue.front();
    vertexQueue.pop();
    if(Layers[v + 1] > Layers[0]) {
      for(unsigned int i = 0; i < Connections[v].size(); ++i) {
        const int u = Connections[v][i];
        // Check if the vertex has an edge to the match vertex
        if(Layers[Pair[u] + 1] < 0) {
          // Set the layer of the vertex (it can be NIL) which is matched to the
          // matched vertex u
          Layers[Pair[u] + 1] = Layers[v + 1] + 1;
          // If the pairing vertex is not NIL add it into the queue
          if(Pair[u] != -1)
            vertexQueue.push(Pair[u]);
        }
      }
    }
  }

  return Layers[0] != -1;
}

void ttk::GabowTarjan::HopcroftKarp(unsigned int &matching) {
  while(BFS())
    for(unsigned int vertex = 0; vertex < MaxSize; ++vertex) {
      if(Pair[vertex] == -1) {
        if(DFS(vertex))
          ++matching;
      }
    }
}

double ttk::GabowTarjan::Distance() {
  // Clear the pairing
  Pair.clear();
  Pair.assign(2 * MaxSize, -1);

  // Clearing Layers
  Layers.clear();
  Layers.resize(MaxSize + 1);

  // No vertices are matched
  unsigned int matching = 0;
  unsigned int matching0 = 0;

  // Clear the connection matrix and set it to the right size
  Connections.clear();
  Connections.resize(MaxSize);

  // The maximal weight of the edges which are used for the matching
  double currentWeight = 0;

  // First non added edge is an iterator pointing to the first edge
  // in Edges which was added to the Connections
  unsigned int firstNotAddedEdge = 0;
  const unsigned int nbEdges = (unsigned int)Edges.size();

  unsigned int lowerBound = 0;
  unsigned int upperBound = nbEdges;
  unsigned int guessEdge = (lowerBound + nbEdges) / 2;

  // Repeat till all the vertices are matched
  while(matching < MaxSize) {
    // Save initial matching.
    matching0 = matching;
    const unsigned int oldGuessEdge = guessEdge;

    currentWeight = Edges[guessEdge].weight;
    while(Edges[guessEdge].weight == currentWeight && guessEdge < nbEdges)
      ++guessEdge;

    if(guessEdge >= nbEdges) {
      this->printMsg("ran out of edges.");
    }

    // Add the edges with the current weight (distance) to the connection matrix
    while(firstNotAddedEdge >= guessEdge) {
      const int v1 = Edges[firstNotAddedEdge].v1;
      const int v2 = Edges[firstNotAddedEdge].v2;

      std::vector<int> vec1 = Connections[v1];
      vec1.erase(std::remove(vec1.begin(), vec1.end(), v2), vec1.end());
      Connections[v1] = vec1;

      --firstNotAddedEdge;
    }

    // Edges[firstNotAddedEdge].weight == currentWeight && firstNotAddedEdge <
    // nbEdges
    while(firstNotAddedEdge < guessEdge) {
      const int v1 = Edges[firstNotAddedEdge].v1;
      const int v2 = Edges[firstNotAddedEdge].v2;

      Connections[v1].push_back(v2);
      ++firstNotAddedEdge;
    }

    // Clear the pairing
    Pair.clear();
    Pair.assign(2 * MaxSize, -1);
    // Clearing Layers
    Layers.clear();
    Layers.resize(MaxSize + 1);

    // Temporarily augment matching.
    this->printMsg("Guessing for " + std::to_string(guessEdge) + "...");

    matching = 0;
    HopcroftKarp(matching);
    // printCurrentMatching<dataType>();

    if(matching >= MaxSize) {

      // Reset matching.
      matching = matching0;
      upperBound = guessEdge;
      guessEdge = (lowerBound + guessEdge) / 2;

      // Ended binary search.
      if(oldGuessEdge == guessEdge) {
        this->printMsg("Binary search success.");
        return Edges[(guessEdge > 0 ? guessEdge - 1 : guessEdge)].weight;
      }

    } else {

      // Check if we did not run out of edges. This should never happen.
      if(firstNotAddedEdge == nbEdges) {
        this->printMsg("Not enough edges to find the matching!");
        // return -1;
        return currentWeight;
      }

      lowerBound = guessEdge;
      guessEdge = (guessEdge + upperBound) / 2;
      // Increase the value of the current weight
      // currentWeight = Edges[firstNotAddedEdge].weight;

      // Ended binary search.
      if(oldGuessEdge == guessEdge) {
        this->printMsg("Binary search success.");
        return Edges[(guessEdge > 0 ? guessEdge - 1 : guessEdge)].weight;
      }
    }
  }

  return -1;
}

void ttk::GabowTarjan::printCurrentMatching() {
  const int size = 2 * MaxSize;
  std::vector<int> missedPlaces;

  {
    std::stringstream msg;
    msg << "Assignment matrix: " << std::endl;
    for(int i = 0; i < size; ++i) {
      const int k = Pair[i];
      if(k < 0 || k > size)
        missedPlaces.push_back(i);
      for(int j = 0; j < size; ++j) {
        msg << (j == k ? "1 " : "0 ");
      }
      msg << std::endl;
    }
    msg << "/Assignment matrix." << std::endl << std::endl;
    this->printMsg(msg.str(), debug::Priority::VERBOSE);
  }

  {
    std::stringstream msg;
    msg << "Missed:" << std::endl;
    for(unsigned int i = 0; i < missedPlaces.size(); ++i) {
      msg << missedPlaces.at(i) << " ";
    }
    msg << std::endl << std::endl;
    this->printMsg(msg.str());
  }
}

int ttk::GabowTarjan::run(std::vector<MatchingType> &matchings) {
  // Compute distance.
  const double dist = this->Distance();
  this->printMsg("Computed distance " + std::to_string(dist));

  // Fill matchings.
  matchings.clear();

  for(unsigned int i = 0; i < Size1; ++i) {
    if(Pair[i] == -1)
      continue;

    const int j = Pair[i] - MaxSize;
    if(j <= -1 || (j < (int)Size2 && Pair[j + MaxSize] != (int)i)) {
      this->printErr("Hopcroft-Karp built an invalid matching.");
      // return -1;
    }

    if(j >= (int)Size2) {
      matchings.emplace_back(i, j, (*Cptr)[i][Size2]);
    } else {
      matchings.emplace_back(i, j, (*Cptr)[i][j]);
    }
  }

  for(unsigned int j = Size1; j < MaxSize; ++j) {
    if(Pair[j] == -1)
      continue;

    const int i = Pair[j] - MaxSize - Size2;
    if(i > -1 && (i >= (int)Size1 || Pair[i + MaxSize + Size2] != (int)j)) {
      this->printErr("Hopcroft-Karp built an invalid matching.");
      // return -1;
    }

    if(i <= -1) {
      matchings.emplace_back(i, j - Size1, (*Cptr)[Size1][j - Size1]);
    } else {
      // Already added by symmetry.
      matchings.emplace_back(i, j - Size1, (*Cptr)[i][j - Size1]);
    }
  }

  return 0;
}
