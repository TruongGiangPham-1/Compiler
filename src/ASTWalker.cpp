#include "ASTWalker.h"

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

//#define DEBUG

namespace gazprea {
    std::any ASTWalker::walkChildren(std::shared_ptr<ASTNode> tree) {
        for (auto child: tree->children) {
#ifdef DEBUG
            std::cout << "Visiting a child" << std::endl;
#endif
            walk(child);
        }
        return 0;
    }


    std::any ASTWalker::walk(std::shared_ptr<ASTNode> tree) {
        // ==========
        // Top-level AST Nodes
        // ==========
        if (std::dynamic_pointer_cast<AssignNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Assign" << std::endl;
#endif // DEBUG
            return this->visitAssign(std::dynamic_pointer_cast<AssignNode>(tree));

        } else if (std::dynamic_pointer_cast<DeclNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Declaration" << std::endl;
#endif // DEBUG
            return this->visitDecl(std::dynamic_pointer_cast<DeclNode>(tree));

        } else if (std::dynamic_pointer_cast<PrintNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit print" << std::endl;
#endif // DEBUG
            return this->visitPrint(std::dynamic_pointer_cast<PrintNode>(tree));

        } else if (std::dynamic_pointer_cast<TypeNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit type " << tree->toString() << std::endl;
#endif // DEBUG
            return this->visitType(std::dynamic_pointer_cast<TypeNode>(tree));

        }

        // ==========
        // EXPRESSION AST NODES
        // ==========
        else if (std::dynamic_pointer_cast<IDNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit ID" << std::endl;
#endif // DEBUG
            return this->visitID(std::dynamic_pointer_cast<IDNode>(tree));

        } else if (std::dynamic_pointer_cast<IntNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Int" << std::endl;
#endif // DEBUG
            return this->visitInt(std::dynamic_pointer_cast<IntNode>(tree));

        // ======
        // Expr/Binary
        // ======
        } else if (std::dynamic_pointer_cast<ArithNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Arith" << std::endl;
#endif // DEBUG
            return this->visitArith(std::dynamic_pointer_cast<ArithNode>(tree));

        } else if (std::dynamic_pointer_cast<CmpNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Cmp" << std::endl;
#endif // DEBUG
            return this->visitCmp(std::dynamic_pointer_cast<CmpNode>(tree));

        } else if (std::dynamic_pointer_cast<IndexNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Index" << std::endl;
#endif // DEBUG
            return this->visitIndex(std::dynamic_pointer_cast<IndexNode>(tree));

        // ======
        // Expr/Vector
        // ======
        } else if (std::dynamic_pointer_cast<FilterNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Filter" << std::endl;
#endif // DEBUG
            return this->visitFilter(std::dynamic_pointer_cast<FilterNode>(tree));

        } else if (std::dynamic_pointer_cast<GeneratorNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Generator" << std::endl;
#endif // DEBUG
            return this->visitGenerator(std::dynamic_pointer_cast<GeneratorNode>(tree));

        } else if (std::dynamic_pointer_cast<RangeVecNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit RangeVec" << std::endl;
#endif // DEBUG
            return this->visitRangeVec(std::dynamic_pointer_cast<RangeVecNode>(tree));
        }

        // =================
        // BLOCK AST NODES
        // =================
        if (std::dynamic_pointer_cast<ConditionalNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Conditional" << std::endl;
#endif // DEBUG
            return this->visitConditional(std::dynamic_pointer_cast<ConditionalNode>(tree));

        } else if (std::dynamic_pointer_cast<LoopNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Loop" << std::endl;
#endif // DEBUG
            return this->visitLoop(std::dynamic_pointer_cast<LoopNode>(tree));

        } else if (std::dynamic_pointer_cast<FunctionForwardNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit functionForwardNode" << std::endl;
#endif // DEBUG
        } else if (std::dynamic_pointer_cast<FunctionSingleNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit functionSingleNode" << std::endl;
#endif // DEBUG

        } else if (std::dynamic_pointer_cast<FunctionBlockNode>(tree)) {

#ifdef DEBUG
            std::cout << "about to visit FunctionBlockNode" << std::endl;
#endif // DEBUG

        }

        // NIL node
#ifdef DEBUG
        std::cout << "about to visit NIL" << std::endl;
#endif // DEBUG
        return this->walkChildren(tree);
    }

    // Top level AST nodes
    std::any ASTWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitPrint(std::shared_ptr<PrintNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitType(std::shared_ptr<TypeNode> tree) {
        return this->walkChildren(tree);
    }

    // Expr
    std::any ASTWalker::visitID(std::shared_ptr<IDNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitInt(std::shared_ptr<IntNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitArith(std::shared_ptr<ArithNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitCmp(std::shared_ptr<CmpNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitIndex(std::shared_ptr<IndexNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitFilter(std::shared_ptr<FilterNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree) {
        return this->walkChildren(tree);
    }

    // Block
    std::any ASTWalker::visitConditional(std::shared_ptr<ConditionalNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitLoop(std::shared_ptr<LoopNode> tree) {
        return this->walkChildren(tree);
    }
    // FUNCTION
    std::any ASTWalker::visitFunctionForward(std::shared_ptr<FunctionForwardNode> tree) {
        return  this->walkChildren(tree);
    }
    std::any ASTWalker::visitFunctionSingle(std::shared_ptr<FunctionSingleNode> tree) {
        return  this->walkChildren(tree);
    }
    std::any ASTWalker::visitFunctionBlock(std::shared_ptr<FunctionBlockNode> tree) {
        return  this->walkChildren(tree);
    }
}