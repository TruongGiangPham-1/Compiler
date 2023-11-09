#pragma once
#include "ExprNode.h"
#include "ASTNode/Type/TypeNode.h"


class CastNode : public ExprNode {
public:
  CastNode(int line) : ExprNode(line) {} ;

  std::shared_ptr<ExprNode> getExpr() {return std::dynamic_pointer_cast<ExprNode>(children[1]);}
  std::shared_ptr<TypeNode> getType() {return std::dynamic_pointer_cast<TypeNode>(children[0]);}
};
