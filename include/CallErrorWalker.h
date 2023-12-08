//
// Created by Joshua Ji on 2023-12-07.
//

// CallErrorWalker is another pass that runs after def and ref
// the main goal is to catch CallErrors, such as procedure values being used in wrong circumstances

#ifndef GAZPREABASE_CALLERRORWALKER_H
#define GAZPREABASE_CALLERRORWALKER_H

#include "ContextedWalker.h"
#include "SyntaxWalker.h"
#include "SymbolTable.h"

namespace gazprea {
    class CallErrorWalker : public ContextedWalker {
    private:
        // WALKER_CONTEXT gives us more info as to what we're currently visiting
        // it's a vector so it's easy to push/pop as we enter into new contexts
        std::vector<WALKER_CONTEXT> contexts;
        bool inContext(WALKER_CONTEXT context);

        std::string debugContext();
        static std::string contextToString(WALKER_CONTEXT context);

        std::shared_ptr<SymbolTable> symTab;
    public:
        CallErrorWalker(std::shared_ptr<SymbolTable> symTab);
    };
}


#endif //GAZPREABASE_CALLERRORWALKER_H
