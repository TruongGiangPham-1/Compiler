#include "ASTWalker.h"

#include "ASTNode/ASTNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/Expr/CastNode.h"
#include "ASTNode/Expr/Literal/BoolNode.h"
#include "ASTNode/Method/ReturnNode.h"
#include "ASTNode/Type/TypeNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/Stream/StreamOut.h"

#include "ASTNode/Expr/Literal/IDNode.h"
#include "ASTNode/Expr/Literal/IntNode.h"
#include "ASTNode/Expr/Literal/RealNode.h"
#include "ASTNode/Expr/Literal/TupleNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
#include "ASTNode/Expr/ExprListNode.h"


#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Loop/LoopNode.h"
#include "ASTNode/Block/ConditionalNode.h"

//#define DEBUG

using namespace gazprea;
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

        } else if (std::dynamic_pointer_cast<StreamOut>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit print" << std::endl;
#endif // DEBUG
            return this->visitStreamOut(std::dynamic_pointer_cast<StreamOut>(tree));
        } else if (std::dynamic_pointer_cast<StreamIn>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit read" << std::endl;
#endif // DEBUG
            return this->visitStreamIn(std::dynamic_pointer_cast<StreamIn>(tree));

        } else if (std::dynamic_pointer_cast<TypeNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit type " << tree->toString() << std::endl;
#endif // DEBUG
            return this->visitType(std::dynamic_pointer_cast<TypeNode>(tree));

        } else if (std::dynamic_pointer_cast<TypeDefNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit typedef " << std::endl;
#endif // DEBUG
            return this->visitTypedef(std::dynamic_pointer_cast<TypeDefNode>(tree));
        }

        // ==========
        // EXPRESSION AST NODES
        // ==========
        else if (std::dynamic_pointer_cast<ExprListNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit expression list " << std::endl;
#endif // DEBUG
            return this->visitExpressionList(std::dynamic_pointer_cast<ExprListNode>(tree));
        }

        else if (std::dynamic_pointer_cast<TupleIndexNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit tupleIndex " << std::endl;
#endif // DEBUG
            return this->visitTupleIndex(std::dynamic_pointer_cast<TupleIndexNode>(tree));
        }

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

        } else if (std::dynamic_pointer_cast<BoolNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Bool" << std::endl;
#endif // DEBUG
            return this->visitBool(std::dynamic_pointer_cast<BoolNode>(tree));

        } else if (std::dynamic_pointer_cast<RealNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Real" << std::endl;
#endif // DEBUG
            return this->visitReal(std::dynamic_pointer_cast<RealNode>(tree));

        } else if (std::dynamic_pointer_cast<TupleNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Tuple" << std::endl;
#endif // DEBUG
            return this->visitTuple(std::dynamic_pointer_cast<TupleNode>(tree));

        } else if (std::dynamic_pointer_cast<CharNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Char" << std::endl;
#endif // DEBUG
            return this->visitChar(std::dynamic_pointer_cast<CharNode>(tree));
        } else if (std::dynamic_pointer_cast<StringNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit String" << std::endl;
#endif // DEBUG
            return this->visitString(std::dynamic_pointer_cast<StringNode>(tree));

        }else if (std::dynamic_pointer_cast<BinaryArithNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Arith" << std::endl;
#endif // DEBUG
            return this->visitArith(std::dynamic_pointer_cast<BinaryArithNode>(tree));
        }

        else if (std::dynamic_pointer_cast<ConcatNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit concat" << std::endl;
#endif // DEBUG
           return this->visitConcat(std::dynamic_pointer_cast<ConcatNode>(tree));
        }
        else if (std::dynamic_pointer_cast<VectorNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit literal Vector" << std::endl;
#endif // DEBUG
            return this->visitVector(std::dynamic_pointer_cast<VectorNode>(tree));
        } else if (std::dynamic_pointer_cast<StdInputNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit std input token (should only be in stream_state function)" << std::endl;
#endif // DEBUG
            return this->visitStdInputNode(std::dynamic_pointer_cast<StdInputNode>(tree));

        } else if (std::dynamic_pointer_cast<MatrixNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit literal Matrix" << std::endl;
#endif // DEBUG
            //return this->visitMatrix(std::dynamic_pointer_cast<MatrixNode>(tree));
        }  else if (std::dynamic_pointer_cast<GeneratorNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit generator" << std::endl;
#endif // DEBUG
            return this->visitGenerator(std::dynamic_pointer_cast<GeneratorNode>(tree));
        }
        else if (std::dynamic_pointer_cast<BinaryCmpNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Cmp" << std::endl;
#endif // DEBUG
            return this->visitCmp(std::dynamic_pointer_cast<BinaryCmpNode>(tree));

        } else if (std::dynamic_pointer_cast<UnaryArithNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit unary Arith node" << std::endl;
#endif // DEBUG
            return this->visitUnaryArith(std::dynamic_pointer_cast<UnaryArithNode>(tree));

        } else if (std::dynamic_pointer_cast<CastNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit cast node" << std::endl;
#endif // DEBUG
            return this->visitCast(std::dynamic_pointer_cast<CastNode>(tree));

        } else if (std::dynamic_pointer_cast<IndexNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Index" << std::endl;
#endif // DEBUG
            return this->visitIndex(std::dynamic_pointer_cast<IndexNode>(tree));

        // ======
        // Expr/Vector
        // ======
        }
        else if (std::dynamic_pointer_cast<FilterNode>(tree)) {
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
        // CONTROL NODES
        // =================
        if (std::dynamic_pointer_cast<ConditionalNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Conditional" << std::endl;
#endif // DEBUG
            return this->visitConditional(std::dynamic_pointer_cast<ConditionalNode>(tree));

        } else if (std::dynamic_pointer_cast<InfiniteLoopNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Infinite Loop" << std::endl;
#endif // DEBUG
            return this->visitInfiniteLoop(std::dynamic_pointer_cast<InfiniteLoopNode>(tree));
        } else if (std::dynamic_pointer_cast<PredicatedLoopNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Predicated Loop" << std::endl;
#endif // DEBUG
            return this->visitPredicatedLoop(std::dynamic_pointer_cast<PredicatedLoopNode>(tree));
        } else if (std::dynamic_pointer_cast<PostPredicatedLoopNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Post Predicated Loop" << std::endl;
#endif // DEBUG
            return this->visitPostPredicatedLoop(std::dynamic_pointer_cast<PostPredicatedLoopNode>(tree));
        } else if (std::dynamic_pointer_cast<BreakNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Break" << std::endl;
#endif // DEBUG
            return this->visitBreak(std::dynamic_pointer_cast<BreakNode>(tree));
        } else if (std::dynamic_pointer_cast<ContinueNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Continue" << std::endl;
#endif // DEBUG
            return this->visitContinue(std::dynamic_pointer_cast<ContinueNode>(tree));
        } else if (std::dynamic_pointer_cast<BlockNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit Block" << std::endl;
#endif // DEBUG
            return this->visitBlock(std::dynamic_pointer_cast<BlockNode>(tree));
        } else if (std::dynamic_pointer_cast<ProcedureNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit procedure" << std::endl;
#endif // DEBUG
            return this->visitProcedure(std::dynamic_pointer_cast<ProcedureNode>(tree));
        } else if (std::dynamic_pointer_cast<ArgNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit arg" << std::endl;
#endif // DEBUG
            return this->visitParameter(std::dynamic_pointer_cast<ArgNode>(tree));

        } else if (std::dynamic_pointer_cast<FunctionNode>(tree))  {
#ifdef DEBUG
            std::cout << "about to visit function" << std::endl;
#endif // DEBUG
            return this->visitFunction(std::dynamic_pointer_cast<FunctionNode>(tree));

        } else if (std::dynamic_pointer_cast<CallNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit call" << std::endl;
#endif // DEBUG
           return this->visitCall(std::dynamic_pointer_cast<CallNode>(tree));
      } else if (std::dynamic_pointer_cast<ReturnNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit return" << std::endl;
#endif // DEBUG
            return this->visitReturn(std::dynamic_pointer_cast<ReturnNode>(tree));

      } else if (std::dynamic_pointer_cast<IdentityNode>(tree)) {
#ifdef DEBUG
            std::cout << "about to visit identity" << std::endl;
#endif // DEBUG
            return this->visitIdentity(std::dynamic_pointer_cast<IdentityNode>(tree));
    } else if (std::dynamic_pointer_cast<NullNode>(tree)) {

#ifdef DEBUG
            std::cout << "about to visit null" << std::endl;
#endif // DEBUG
            return this->visitNull(std::dynamic_pointer_cast<NullNode>(tree));
    } else {
          // NIL node
#ifdef DEBUG
          std::cout << "about to visit NIL" << std::endl;
#endif // DEBUG
          return this->walkChildren(tree);
      }
  }

    // Top level AST nodes
    std::any ASTWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitStreamOut(std::shared_ptr<StreamOut> tree) {
        return this->walkChildren(tree);
    }

    std::any ASTWalker::visitStreamIn(std::shared_ptr<StreamIn> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitType(std::shared_ptr<TypeNode> tree) {
        return this->walkChildren(tree);
    }

    // Expr
    std::any ASTWalker::visitExpressionList(std::shared_ptr<ExprListNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitID(std::shared_ptr<IDNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitInt(std::shared_ptr<IntNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitBool(std::shared_ptr<BoolNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitReal(std::shared_ptr<RealNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitTuple(std::shared_ptr<TupleNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitChar(std::shared_ptr<CharNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitString(std::shared_ptr<StringNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitVector(std::shared_ptr<VectorNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitStdInputNode(std::shared_ptr<StdInputNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitArith(std::shared_ptr<BinaryArithNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree) {
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
        for (auto conditional : tree->conditions) {
          walkChildren(conditional);
        }

        for (auto body : tree->bodies) {
          walkChildren(body);
        }

        return this->walkChildren(tree);
    }

    std::any ASTWalker::visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) {
        return this->walkChildren(tree);
    }

    std::any ASTWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
        return this->walkChildren(tree);
    }

    std::any ASTWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
        return this->walkChildren(tree);
    }

    std::any ASTWalker::visitBreak(std::shared_ptr<BreakNode> tree) {
        return 0;
    }

    std::any ASTWalker::visitContinue(std::shared_ptr<ContinueNode> tree) {
        return 0;
    }
    // FUNCTION
    //
    std::any ASTWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree) {
        for (auto arg : tree->orderedArgs) {
          walk(arg);
        }       
        //return this->walkChildren(tree);
        if (tree->body) {
            walk(tree->body);
        }
        return 0;

    }

    std::any ASTWalker::visitFunction(std::shared_ptr<FunctionNode> tree) {
        for (auto arg : tree->orderedArgs) {
          walk(arg);
        }

        //return this->walkChildren(tree);
        if (tree->body) {
            walk(tree->body);
        } else if (tree->expr) {
            walk(tree->expr);
        }
        return 0;

    }

    std::any ASTWalker::visitBlock(std::shared_ptr<BlockNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitParameter(std::shared_ptr<ArgNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitUnaryArith(std::shared_ptr<UnaryArithNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitCall(std::shared_ptr<CallNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitReturn(std::shared_ptr<ReturnNode> tree) {
        walk(tree->getReturnExpr());
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitTypedef(std::shared_ptr<TypeDefNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitCast(std::shared_ptr<CastNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitNull(std::shared_ptr<NullNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitIdentity(std::shared_ptr<IdentityNode> tree) {
        return this->walkChildren(tree);
    }
    std::any ASTWalker::visitConcat(std::shared_ptr<ConcatNode> tree) {
        return this->walkChildren(tree);
    }

}
