#pragma once

#include "AST.h"
#include "Scope.h"
#include "SymbolTable.h"
#include "BuiltInTypeSymbol.h"

namespace gazprea {
    class ASTWalker {
    public:
        ASTWalker() {}

        virtual std::any visit(AST *t) {
            visitChildren(t);
            return 0;
        }

        virtual void visitChildren(AST *t) {
//            for (auto child: t->children) visit(child);
        }
    };

    class DefRef : public ASTWalker {
    private:
        SymbolTable *symtab;
        Scope *currentScope;

    public:
        DefRef(SymbolTable *symtab, AST *root);
        std::any visit(AST *t) override;
        void visitChildren(AST *t) override;

        // custom
        BuiltInTypeSymbol* resolveType(AST *t);

        // tokens in our grammar
        void visitVAR_DECL(AST *t);
        void visitASSIGN(AST *t);
        void visitLOOPCONDITIONAL(AST *t);
        void visitFILTERGENERATOR(AST *t);
        void visitID(AST *t);
    };
}
