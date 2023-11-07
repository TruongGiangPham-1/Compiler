#pragma once
#include "ASTNode/ASTNode.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "ScopedSymbol.h"

class ProcedureCallNode: public ASTNode {
public:
    std::shared_ptr<Symbol> procCallName;  // only used for calling user defined function
    std::shared_ptr<FunctionSymbol> procedureRef;  // symbol to the function definition that it is calling

    ProcedureCallNode(int loc): ASTNode(loc) {};
};
