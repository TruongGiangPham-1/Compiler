#pragma once
#include "ASTNode/ASTNode.h"
#include "ASTWalker.h"
#include "SymbolTable.h"
#include "Type.h"
#include <memory>

namespace gazprea {
    class PromotedType {
    public:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;
        static std::string booleanResult[7][7]; //XOR AND NOT OR
        static std::string arithmeticResult[7][7]; //+ - / * ^ % **
        static std::string comparisonResult[7][7]; //>= <= > <
        static std::string equalityResult[7][7]; //==, !=
        static std::string promotionTable[7][7];
        static std::string castTable[4][4];

        const int boolIndex = 0;
        const int charIndex = 1;
        const int integerIndex = 2;
        const int realIndex = 3;
        const int tupleIndex = 4;
        const int identityIndex = 5;
        const int nullIndex = 6;

        std::shared_ptr<Type> getType(std::string table[7][7], std::shared_ptr<ASTNode> lhs, std::shared_ptr<ASTNode> rhs, std::shared_ptr<ASTNode> t);
        int getTypeIndex(const std::string type);
        void promoteVectorElements(std::shared_ptr<Type>promoteTo, std::shared_ptr<ASTNode> exprNode);
        void updateVectorNodeEvaluatedType(std::shared_ptr<Type>assignType, std::shared_ptr<ASTNode> exprNode);
        void promoteIdentityAndNull(std::shared_ptr<Type>promoteTo, std::shared_ptr<ASTNode>identityNode);

        PromotedType(std::shared_ptr<SymbolTable> symtab);
        ~PromotedType();
    };


    class TypeWalker : public ASTWalker {
    private:
        std::shared_ptr<SymbolTable> symtab;
        std::shared_ptr<Scope> currentScope;
        std::shared_ptr<PromotedType> promotedType;

    public:
        TypeWalker(std::shared_ptr<SymbolTable> symtab, std::shared_ptr<PromotedType> promotedType);
        ~TypeWalker();

        std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
        std::any visitDecl(std::shared_ptr<DeclNode> tree) override;

        std::any visitID(std::shared_ptr<IDNode> tree) override;
        std::any visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) override;
        std::any visitInt(std::shared_ptr<IntNode> tree) override;
        std::any visitReal(std::shared_ptr<RealNode> tree) override;
        std::any visitTuple(std::shared_ptr<TupleNode> tree) override;
        std::any visitChar(std::shared_ptr<CharNode> tree) override;
        std::any visitBool(std::shared_ptr<BoolNode> tree) override;
        std::any visitVector(std::shared_ptr<VectorNode> tree) override;
        std::any visitMatrix(std::shared_ptr<MatrixNode> tree) override;

        std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
        std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
        std::any visitUnaryArith(std::shared_ptr<UnaryArithNode>tree) override;

        // streams
        std::any visitStreamIn(std::shared_ptr<StreamIn> tree) override;
        std::any visitStreamOut(std::shared_ptr<StreamOut> tree) override;

        // === Null and identity
        std::any visitNull(std::shared_ptr<NullNode> tree) override;
        std::any visitIdentity(std::shared_ptr<IdentityNode> tree) override;

        // Return CFG unsupported right now :(

        std::any visitCall(std::shared_ptr<CallNode> tree) override; // Procedure Call, function and procedure call in expr
        std::any visitTypedef(std::shared_ptr<TypeDefNode> tree) override;
        std::any visitCast(std::shared_ptr<CastNode> tree) override;

        std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree);
        std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree);
        std::string typeEnumToString(TYPE t);




    };
}
