#include "ASTBuilder.h"
#include "ASTNode/ASTNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/TypeNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/PrintNode.h"
#include "ASTNode/Expr/IDNode.h"
#include "ASTNode/Expr/IntNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Block/LoopNode.h"
#include "ASTNode/Block/ConditionalNode.h"

namespace gazprea {
    class ASTWalker {
    protected:
        std::any walkChildren(std::shared_ptr<ASTNode> tree);

    public:
        ASTWalker() {};

        std::any walk(std::shared_ptr<ASTNode> tree);
        virtual std::any visitArith(std::shared_ptr<ArithNode> tree);
        virtual std::any visitAssign(std::shared_ptr<AssignNode> tree);
        virtual std::any visitBaseVector(std::shared_ptr<BaseVectorNode> tree);
        // MISNOMER ALERT: this is for any node containing a nested scope (i.e. nested conditionals, loops)
        // this might not call at all
        virtual std::any visitBlock(std::shared_ptr<BlockNode> tree);
        virtual std::any visitComp(std::shared_ptr<CmpNode> tree);
        virtual std::any visitDecl(std::shared_ptr<DeclNode> tree);
        virtual std::any visitFilter(std::shared_ptr<FilterNode> tree);
        virtual std::any visitGenerator(std::shared_ptr<GeneratorNode> tree);
        virtual std::any visitID(std::shared_ptr<IDNode> tree);
        virtual std::any visitIfStatement(std::shared_ptr<ConditionalNode> tree);
        virtual std::any visitIndex(std::shared_ptr<IndexNode> tree);
        virtual std::any visitInt(std::shared_ptr<IntNode> tree);
        virtual std::any visitLoop(std::shared_ptr<LoopNode> tree);
        virtual std::any visitPrint(std::shared_ptr<PrintNode> tree);
        virtual std::any visitRangeVector(std::shared_ptr<RangeVecNode> tree);
        virtual std::any visitType(std::shared_ptr<TypeNode> tree);
    };
}
