#include "ASTNode/ASTNode.h"
#include "ASTNode/Block/BlockNode.h"

class CallableNode : public ASTNode  {
  std::vector<ASTNode> args;
  BlockNode body;
};
