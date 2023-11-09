//
// Created by truong on 08/11/23.
//
#include "../include/Swap.h"


namespace  gazprea{

    std::any Swap::visitFunction(std::shared_ptr<FunctionNode> tree) {
        return 0;
    }

    std::any Swap::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
        if (tree->body) {
            auto cast = std::dynamic_pointer_cast<ASTNode>(tree);
            auto find = map->find(tree->nameSym->getName());
            if (find != map->end() ) {
                throw SyntaxError(tree->loc(), "redefintion of procedure");
            }
            map->emplace(tree->nameSym->getName(), cast);
        }
        return 0;
    }

}
