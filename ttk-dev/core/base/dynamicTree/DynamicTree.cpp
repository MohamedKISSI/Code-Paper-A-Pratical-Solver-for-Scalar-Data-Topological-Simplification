#include <DynamicTree.h>

#include <sstream>

// DynamicTree ----------------------------------

std::string ttk::DynamicTree::print() const {
  std::stringstream res;

  for(const auto &node : nodes_) {
    res << "id: " << &node - &nodes_[0];
    if(node.parent_) {
      res << ", parent: " << node.parent_ - &nodes_[0];
    } else {
      res << ", parent: X";
    }
    res << " find root: " << node.findRoot();
    res << " root: " << node.findRoot() - &nodes_[0];
    res << std::endl;
  }
  return res.str();
}

std::string ttk::DynamicTree::printNbCC() const {
  std::stringstream res;
  std::vector<DynTreeNode *> roots;
  roots.reserve(nodes_.size());
  for(const auto &n : nodes_) {
    roots.emplace_back(n.findRoot());
  }
  std::sort(roots.begin(), roots.end());
  const auto it = std::unique(roots.begin(), roots.end());
  roots.erase(it, roots.end());
  res << "nb nodes " << nodes_.size() << std::endl;
  res << "nb cc " << roots.size() << std::endl;
  return res.str();
}

// DynTreeNode ----------------------------------

void ttk::DynTreeNode::evert() {
  if(!parent_)
    return;

  DynTreeNode *curNode = this;
  DynTreeNode *parentNode = curNode->parent_;
  DynTreeNode *gParentNode = parentNode->parent_;

  curNode->parent_ = nullptr;

  // Reverse all the node until the root
  while(true) {
    parentNode->parent_ = curNode;

    curNode = parentNode;
    parentNode = gParentNode;

    if(gParentNode) {
      gParentNode = gParentNode->parent_;
    } else {
      break;
    }
  }
}

ttk::DynTreeNode *ttk::DynTreeNode::findRoot() const {
  DynTreeNode *curNode = const_cast<DynTreeNode *>(this);
  while(curNode->parent_) {
    curNode = curNode->parent_;
  }
  return curNode;
}

bool ttk::DynTreeNode::insertEdge(DynTreeNode *const n) {
  evert();
  auto nRoot = n->findRoot();

  if(nRoot != this) {
    // The two nodes are in two different trees
    parent_ = n;
    return true;
  }

  // here the nodes are in the same tree
  return false;
}
