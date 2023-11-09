//
// Created by truong on 08/11/23.
//

#ifndef GAZPREABASE_SWAP_H
#define GAZPREABASE_SWAP_H

#include "ASTWalker.h"
#include "SymbolTable.h"
#include "BuiltInTypeSymbol.h"
#include "CompileTimeExceptions.h"
#include "ScopedSymbol.h"
#include "AdvanceType.h"

namespace gazprea {
    class Swap : public ASTWalker {
    public:
        std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ASTNode>>> map;
        Swap(std::shared_ptr<std::unordered_map<std::string, std::shared_ptr<ASTNode>>>map): map(map) {};

        std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;
        std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;
    };
}
#endif //GAZPREABASE_SWAP_H
