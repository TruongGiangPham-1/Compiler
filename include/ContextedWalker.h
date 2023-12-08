//
// Created by Joshua Ji on 2023-12-07.
//

#ifndef GAZPREABASE_CONTEXTEDWALKER_H
#define GAZPREABASE_CONTEXTEDWALKER_H

#include "ASTWalker.h"
#include "WalkerContext.h"

// Walkers that need the WALKER_CONTEXT enum
// e.g. SyntaxWalker and CallErrorWalker

namespace gazprea {
    class ContextedWalker : public ASTWalker {
    protected:
        // WALKER_CONTEXT gives us more info as to what we're currently visiting
        // it's a vector so it's easy to push/pop as we enter into new contexts
        std::vector<WALKER_CONTEXT> contexts;
        bool inContext(WALKER_CONTEXT context);
        bool directlyInContext(WALKER_CONTEXT context);

        std::string debugContext();
        static std::string contextToString(WALKER_CONTEXT context);

        int countContextDepth(WALKER_CONTEXT ctx);
    public:
        ContextedWalker();
    };
}


#endif //GAZPREABASE_CONTEXTEDWALKER_H
