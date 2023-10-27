#pragma once

#include "ASTNode.h"
#include "Scope.h"
#include "SymbolTable.h"
#include "BuiltInTypeSymbol.h"

namespace gazprea {
    class ASTWalker {
    public:
        ASTWalker() {}

        virtual std::any visit(ASTNode *t) {
            visitChildren(t);
            return 0;
        }

        virtual void visitChildren(ASTNode *t) {
//            for (auto child: t->children) visit(child);
        }
    };

    class DefRef : public ASTWalker {
    private:
        SymbolTable *symtab;
        Scope *currentScope;

    public:
        DefRef(SymbolTable *symtab, ASTNode *root);
        std::any visit(ASTNode *t) override;
        void visitChildren(ASTNode *t) override;

        // custom
        BuiltInTypeSymbol* resolveType(ASTNode *t);

        // tokens in our grammar
        void visitVAR_DECL(ASTNode *t);
        void visitASSIGN(ASTNode *t);
        void visitLOOPCONDITIONAL(ASTNode *t);
        void visitFILTERGENERATOR(ASTNode *t);
        void visitID(ASTNode *t);
    };
}
