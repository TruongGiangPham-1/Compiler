//
// Created by Joshua Ji on 2023-12-08.
//

#ifndef GAZPREABASE_ALIASERRORWALKER_H
#define GAZPREABASE_ALIASERRORWALKER_H

#include "ASTWalker.h"

namespace gazprea {
    class AliasErrorWalker : public ASTWalker {
    private:
        bool isProcedure(const std::shared_ptr<CallNode>& tree);
        // whenever we encounter a procedure call
        // we go through the variable args and add the mlirName of the id into the
        std::set<std::string> mlirNames;

        bool insideProcedureArgs;
    public:
        AliasErrorWalker();

        std::any visitCall(std::shared_ptr<CallNode> tree) override;
        std::any visitID(std::shared_ptr<IDNode> tree) override;
    };
}

#endif //GAZPREABASE_ALIASERRORWALKER_H
