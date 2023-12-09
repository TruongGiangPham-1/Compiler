//
// Created by Joshua Ji on 2023-12-07.
//

#ifndef GAZPREABASE_WALKERCONTEXT_H
#define GAZPREABASE_WALKERCONTEXT_H

// additional context as to what we're currently visiting

enum class WALKER_CONTEXT {
    FUNCTION, // func def
    PROCEDURE, // procedure def
    DECL_BODY, // inside `type qualifier ID = ***`
    ASSIGN_BODY, // inside `a = ***`
    VECTOR_LITERAL, // inside a VectorNode
    STREAM_OUT,
    BINOP, // inside any binary operation
    INPUT_ARGS, // f(***) argument inside a func/procedure call
    RETURN_STMT,
    CONDITIONAL_EXPR, // e.g. `if (***) else...` or `loop while (***)`
    ITERATOR_DOMAIN,
    NONE,
};

#endif // GAZPREABASE_WALKERCONTEXT_H
