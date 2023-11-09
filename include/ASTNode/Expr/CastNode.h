#pragma once
#include "ExprNode.h"

class CastNode : public ExprNode {
public:
  CastNode(int line) : ExprNode(line) {} ;

};
