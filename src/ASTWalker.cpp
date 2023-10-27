#include "ASTWalker.h"

#include "ASTWalker.h"
 #define DEBUG

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

    // WARNING: we have to ensure the order in which we dynamic_cast
    // is a bottom-up approach wrt the ASTNode hierarchy
    // i.e. we have to cast the most specific type first
    // I still need to do this for the rest of the nodes
    // TODO: Maybe there's a way to automate this?
    std::any ASTWalker::walk(std::shared_ptr<ASTNode> tree) {
        if (std::dynamic_pointer_cast<ArithNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Arithmetic" << std::endl;
#endif // DEBUG
            return this->visitArith(std::dynamic_pointer_cast<ArithNode>(tree));

        } else if (std::dynamic_pointer_cast<AssignNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Assign" << std::endl;
#endif // DEBUG

            return this->visitAssign(std::dynamic_pointer_cast<AssignNode>(tree));

        } else if (std::dynamic_pointer_cast<BlockNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Block" << std::endl;
#endif // DEBUG

            return this->visitBlock(std::dynamic_pointer_cast<BlockNode>(tree));

        } else if (std::dynamic_pointer_cast<CmpNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Comparison" << std::endl;
#endif // DEBUG

            return this->visitComp(std::dynamic_pointer_cast<CmpNode>(tree));

        } else if (std::dynamic_pointer_cast<DeclNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Declaration" << std::endl;
#endif // DEBUG

            return this->visitDecl(std::dynamic_pointer_cast<DeclNode>(tree));

        } else if (std::dynamic_pointer_cast<FilterNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit filter" << std::endl;
#endif // DEBUG

            return this->visitFilter(std::dynamic_pointer_cast<FilterNode>(tree));

        } else if (std::dynamic_pointer_cast<GeneratorNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit generator" << std::endl;
#endif // DEBUG

            return this->visitGenerator(std::dynamic_pointer_cast<GeneratorNode>(tree));

        } else if (std::dynamic_pointer_cast<IDNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit ID" << std::endl;
#endif // DEBUG

            return this->visitID(std::dynamic_pointer_cast<IDNode>(tree));

        } else if (std::dynamic_pointer_cast<ConditionalNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit If statement" << std::endl;
#endif // DEBUG

            return this->visitIfStatement(std::dynamic_pointer_cast<ConditionalNode>(tree));

        } else if (std::dynamic_pointer_cast<IndexNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Index" << std::endl;
#endif // DEBUG

            return this->visitIndex(std::dynamic_pointer_cast<IndexNode>(tree));

        } else if (std::dynamic_pointer_cast<IntNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Integer" << std::endl;
#endif // DEBUG

            return this->visitInt(std::dynamic_pointer_cast<IntNode>(tree));

        } else if (std::dynamic_pointer_cast<LoopNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit loop" << std::endl;
#endif // DEBUG

            return this->visitLoop(std::dynamic_pointer_cast<LoopNode>(tree));
        } else if (std::dynamic_pointer_cast<PrintNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit print" << std::endl;
#endif // DEBUG

            return this->visitPrint(std::dynamic_pointer_cast<PrintNode>(tree));
        } else if (std::dynamic_pointer_cast<RangeVecNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Range Vector" << std::endl;
#endif // DEBUG

            return this->visitRangeVector(
                    std::dynamic_pointer_cast<RangeVecNode>(tree));
        } else if (std::dynamic_pointer_cast<BaseVectorNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit base vector" << std::endl;
#endif // DEBUG

            return this->visitBaseVector(
                    std::dynamic_pointer_cast<BaseVectorNode>(tree));
        } else if (std::dynamic_pointer_cast<TypeNode>(
                tree)) { // don't need to visit type nodes
#ifdef DEBUG
            std::cout << "about to visit type " << tree->toString() << std::endl;
#endif // DEBUG

            return this->visitType(
                    std::dynamic_pointer_cast<BaseVectorNode>(tree));
        } else {
#ifdef DEBUG
            std::cout << "about to visit NIL" << std::endl;
#endif // DEBUG

            return this->walkChildren(tree);
        }

        return 0;
    }

    std::any ASTWalker::visitArith(std::shared_ptr<ArithNode> node) {

        return this->walkChildren(node);
    }

    std::any ASTWalker::visitAssign(std::shared_ptr<AssignNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitBaseVector(std::shared_ptr<BaseVectorNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitBlock(std::shared_ptr<BlockNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitComp(std::shared_ptr<CmpNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitDecl(std::shared_ptr<DeclNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitFilter(std::shared_ptr<FilterNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitGenerator(std::shared_ptr<GeneratorNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitID(std::shared_ptr<IDNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitIfStatement(std::shared_ptr<ConditionalNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitIndex(std::shared_ptr<IndexNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitInt(std::shared_ptr<IntNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitLoop(std::shared_ptr<LoopNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitPrint(std::shared_ptr<PrintNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitRangeVector(std::shared_ptr<RangeVecNode> node) {
        return this->walkChildren(node);
    }

    std::any ASTWalker::visitType(std::shared_ptr<TypeNode> node) {
        return this->walkChildren(node);
    }
}