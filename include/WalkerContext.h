//
// Created by Joshua Ji on 2023-12-07.
//

#ifndef GAZPREABASE_WALKERCONTEXT_H
#define GAZPREABASE_WALKERCONTEXT_H

// additional context as to what we're currently visiting

enum class WALKER_CONTEXT {
    FUNCTION,
    DECL_BODY, // inside `type qualifier ID = ***`
    VECTOR_LITERAL, // inside a VectorNode
    NONE,
};

#endif //GAZPREABASE_WALKERCONTEXT_H
