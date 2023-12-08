//
// Created by Joshua Ji on 2023-12-08.
//

#ifndef GAZPREABASE_ALIASERRORWALKER_H
#define GAZPREABASE_ALIASERRORWALKER_H

#include "ASTWalker.h"

namespace gazprea {
    class AliasCount {
    public:
        int constReferences; // const reference count
        bool mutReferenced; // true if the variable has been referenced at least once mutably
        explicit AliasCount(bool mut);
        void incrementConstRef();
    };

    class AliasErrorWalker : public ASTWalker {
    private:
        // whenever we encounter a procedure call
        // we go through the variable args and add the mlirName of the id into the
        std::map<std::string, std::shared_ptr<AliasCount>> mlirNames;

        // -1 if we are not in a procedure call
        std::shared_ptr<ProcedureSymbol> procSymbol;
        int procedureArgIdx;
    public:
        AliasErrorWalker();

        // common code for visitID, visitIndex, visitTupleIndex
        // i can't name things for the life of me
        void handleSymbolAlias(std::shared_ptr<Symbol> sym, int loc, std::string varName);

        std::any visitCall(std::shared_ptr<CallNode> tree) override;
        std::any visitID(std::shared_ptr<IDNode> tree) override;
//        std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
//        std::any visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) override;
    };
}

#endif //GAZPREABASE_ALIASERRORWALKER_H
