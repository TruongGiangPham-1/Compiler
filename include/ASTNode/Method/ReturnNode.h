#pragma once
#include "ASTNode/ASTNode.h"

class ReturnNode : public ASTNode {
public:
  // can be nullptr
  ReturnNode(int line) : ASTNode(line) {};
  std::shared_ptr<ASTNode> returnExpr;
  std::shared_ptr<ASTNode> getReturnExpr() {return children[0];};
};
