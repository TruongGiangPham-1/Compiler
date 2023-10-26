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
            for (auto child: t->children) visit(child);
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

    class ComputeType : public ASTWalker {
    private:
        size_t ancestorCount;
        SymbolTable *symtab;
    public:
        ComputeType(SymbolTable *symtab);
        std::any visit(AST *t) override;
        void visitChildren(AST *t) override;

        // tokens in our grammar
        void visitEXPRESSION(AST* t);
        void visitINT(AST* t);
        void visitVectorConstructor(AST* t);
        void visitINDEX(AST* t);
        void visitPARENTHESES(AST* t);
        void visitID(AST* t);
        void visitBinaryOp(AST* t);
    };
}
