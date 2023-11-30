//
// Created by Joshua Ji on 2023-11-28.
//

#ifndef GAZPREABASE_STDINPUTNODE_H
#define GAZPREABASE_STDINPUTNODE_H

#include "ExprNode.h"

// To make the stream_state function work well with our current code for function calls,
// it is easier to create a StdInput Node representing the token std_input that gets passed into stream_state
// It provides no functionality other than to be a placeholder for the token std_input

// this is only added in the ASTBuilder, so it is guaranteed to only be used in the stream_state function

class StdInputNode : public ExprNode {
public:
    StdInputNode(int line) : ExprNode(line) {};
};


#endif //GAZPREABASE_STDINPUTNODE_H
