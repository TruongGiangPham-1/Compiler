//
// Created by Joshua Ji on 2023-12-08.
//

#ifndef GAZPREABASE_PROCEDURECALLARGWALKER_H
#define GAZPREABASE_PROCEDURECALLARGWALKER_H

#include "ASTWalker.h"

namespace gazprea {
    class AliasCount {
    public:
        int constReferences; // const reference count
        bool mutReferenced; // true if the variable has been referenced at least once mutably
        explicit AliasCount(bool mut);
        void incrementConstRef();
    };

    class ProcedureCallArgWalker : public ASTWalker {
    private:
        // whenever we encounter a procedure call
        // we go through the variable args and add the mlirName of the id into the
        std::map<std::string, std::shared_ptr<AliasCount>> mlirNames;

        // given a symbol and some other metadata
        // check if the symbol is already aliased in the current procedure argument list
        void checkAliasing(std::shared_ptr<Symbol> sym, int loc, std::string varName);

        // -1 if we are not in a procedure call
        std::shared_ptr<ProcedureSymbol> procSymbol;
        int procedureArgIdx;
    public:
        ProcedureCallArgWalker();

        std::any visitCall(std::shared_ptr<CallNode> tree) override;
        std::any visitID(std::shared_ptr<IDNode> tree) override;

        // valid mutable arguments include IDs, matrix/vector indexes and tuple indexes
        bool validMutArg(std::shared_ptr<ASTNode> tree);
    };
}

#endif //GAZPREABASE_PROCEDURECALLARGWALKER_H
