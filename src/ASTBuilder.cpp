#include "ASTBuilder.h"
#include "ASTNode/ASTNode.h"
#include "ASTNode/AssignNode.h"
#include "ASTNode/DeclNode.h"
#include "ASTNode/Stream/StreamOut.h"
#include "ASTNode/Expr/IDNode.h"
#include "ASTNode/Expr/IntNode.h"
#include "ASTNode/Expr/NullNode.h"
#include "ASTNode/Expr/Binary/BinaryExpr.h"
#include "ASTNode/Expr/Vector/RangeVecNode.h"
#include "ASTNode/Expr/Vector/GeneratorNode.h"
#include "ASTNode/Expr/Vector/FilterNode.h"
#include "ASTNode/Expr/Unary/UnaryExpr.h"
#include "ASTNode/Block/LoopNode.h"
#include "ASTNode/Block/ConditionalNode.h"
#include "ASTNode/Block/FunctionNode.h"
#include "ASTNode/Type/TypeNode.h"
#include "ASTNode/Type/VectorTypeNode.h"
#include "ASTNode/Type/StringTypeNode.h"
#include "ASTNode/Type/MatrixTypeNode.h"
#include "FunctionCallTypes/FuncCallType.h"
#include "ASTNode/FunctionCallNode.h"
#include "ASTNode/Type/TupleTypeNode.h"

#define DEBUG

namespace gazprea {
    std::any ASTBuilder::visitFile(GazpreaParser::FileContext *ctx) {
#ifdef DEBUG
        std::cout << "INIT BUILDER: VISITING FILE" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<ASTNode>();
        int size = ctx->children.size();
        int c = 0;
        for ( auto statement : ctx->children) {
            c++;
            if (c == size) break;
            t->addChild(visit(statement));
        }
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    // STREAMS
    std::any ASTBuilder::visitOutputStream(GazpreaParser::OutputStreamContext *ctx) {
#ifdef DEBUG
        std::cout << "visitOutputStream" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<StreamOut>(ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    // DECLARATION STUFF
    std::any ASTBuilder::visitInferred_size(GazpreaParser::Inferred_sizeContext *ctx) {
        /*
         * "Inferred size" declarations are declarations where the type's size is not known at compile time
         * This includes vectors (integer[*]) and matrices (integer[*][*])
         */
#ifdef DEBUG
        std::cout << "visitInferred_size (vardecl inferred size) type " << ctx->getStart()->getType() << ": "
                  << ctx->inferred_sized_type()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<DeclNode> t = std::make_shared<DeclNode>(ctx->getStart()->getLine(), sym);

        // Qualifier
        if (ctx->qualifier()) {
            t->qualifier = std::any_cast<QUALIFIER>(visit(ctx->qualifier()));
        } else {
            t->qualifier = QUALIFIER::NONE;
        }

        // Type
        t->addChild(visit(ctx->inferred_sized_type()));

        // expression (it is always present when inferred)
        t->addChild(visit(ctx->expression()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitSized(GazpreaParser::SizedContext *ctx) {
        /*
         * "Sized" declarations are declarations where the type's size is known ar compile time
         * note that this includes types where there is no size (e.g. integer, boolean)
         */
#ifdef DEBUG
        std::cout << "visitSized (vardecl sized) type " << ctx->getStart()->getType() << ": "
                  << ctx->ID()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<DeclNode> t = std::make_shared<DeclNode>(ctx->getStart()->getLine(), sym);

        // Qualifier
        if (ctx->qualifier()) {
            t->qualifier = std::any_cast<QUALIFIER>(visit(ctx->qualifier()));
        } else {
            t->qualifier = QUALIFIER::NONE;
        }

        // Type
        t->addChild(visit(ctx->known_sized_type()));

        // check if expression is present
        if (ctx->expression()) {
#ifdef DEBUG
            std::cout << "\tAdding non-null expression to decl" << std::endl;
#endif
            t->addChild(visit(ctx->expression()));
        } else {
            std::shared_ptr<ASTNode> nullNode = std::make_shared<NullNode>(ctx->getStart()->getLine());
            t->addChild(nullNode);
#ifdef DEBUG
            std::cout << "\tAdding null to empty decl" << std::endl;
#endif
        }

        return std::dynamic_pointer_cast<ASTNode>(t);
    }


    // TYPE STUFF
    std::any ASTBuilder::visitQualifier(GazpreaParser::QualifierContext *ctx) {
        if (ctx->RESERVED_CONST()) {
            return QUALIFIER::CONST;
        } else if (ctx->RESERVED_VAR()) {
            return QUALIFIER::VAR;
        } else {
            std::cout << "ERROR: unknown qualifier" << ctx->getText() << std::endl;
            throw std::runtime_error("unknown qualifier " + ctx->getText());
        }
    }

    std::any ASTBuilder::visitBuilt_in_type(GazpreaParser::Built_in_typeContext *ctx) {
        /*
         * I chose to not distinguish between the RESERVED and ID types
         * since when we do the Def and Ref passes, i think we will resolve them in the same way
         */
#ifdef DEBUG
        std::cout << "visitBuiltin Type " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<TypeNode>(ctx->getStart()->getLine(), sym);

        return t;
    }

    std::any ASTBuilder::visitVector(GazpreaParser::VectorContext *ctx) {
        /*
         * <innerType>[*]
         */
#ifdef DEBUG
        std::cout << "visitVector (inferred size vector type)" << ctx->getText() << std::endl;
#endif
        auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->built_in_type()));
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<VectorTypeNode> t = std::make_shared<VectorTypeNode>(ctx->getStart()->getLine(), sym, innerType);

        t->typeEnum = TYPE::VECTOR;

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitVector_type(GazpreaParser::Vector_typeContext *ctx) {
        /*
         * <innerType>[expr] (known size vector type)
         */
#ifdef DEBUG
        std::cout << "visitVector_type (known size)" << ctx->getText() << std::endl;
#endif
        auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->built_in_type()));
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<VectorTypeNode> t = std::make_shared<VectorTypeNode>(ctx->getStart()->getLine(), sym, innerType);

        t->typeEnum = TYPE::VECTOR;
        t->size = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitString(GazpreaParser::StringContext *ctx) {
        /*
         * string[*] (inferred size)
         */
#ifdef DEBUG
        std::cout << "visitString (inferred size)" << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<StringTypeNode> t = std::make_shared<StringTypeNode>(ctx->getStart()->getLine(), sym);

        t->typeEnum = TYPE::STRING;

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitString_type(GazpreaParser::String_typeContext *ctx) {
        /*
         * string[expr] (known size)
         */
#ifdef DEBUG
        std::cout << "visitString_type (known size)" << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<StringTypeNode> t = std::make_shared<StringTypeNode>(ctx->getStart()->getLine(), sym);

        t->typeEnum = TYPE::STRING;
        t->size = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitMatrixFirst(GazpreaParser::MatrixFirstContext *ctx) {
        /*
         * matrix[*, expr]
         */
#ifdef DEBUG
        std::cout << "visitMatrixFirst [*, expr] " << ctx->getText() << std::endl;
#endif
        auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->built_in_type()));
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<MatrixTypeNode> t = std::make_shared<MatrixTypeNode>(ctx->getStart()->getLine(), sym, innerType);

        t->typeEnum = TYPE::MATRIX;

        // get right size
        t->sizeRight = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));

        std::cout << "Done matrix" << std::endl;
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitMatrixSecond(GazpreaParser::MatrixSecondContext *ctx) {
        /*
         * matrix[expr, *]
         */
#ifdef DEBUG
        std::cout << "visitMatrixSecond [*, expr] " << ctx->getText() << std::endl;
#endif
        auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->built_in_type()));
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<MatrixTypeNode> t = std::make_shared<MatrixTypeNode>(ctx->getStart()->getLine(), sym, innerType);

        t->typeEnum = TYPE::MATRIX;

        // get right size
        t->sizeLeft = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitMatrix(GazpreaParser::MatrixContext *ctx) {
        /*
         * matrix[*, *]
         */
#ifdef DEBUG
        std::cout << "visitMatrix [*, *] " << ctx->getText() << std::endl;
#endif
        auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->built_in_type()));
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<MatrixTypeNode> t = std::make_shared<MatrixTypeNode>(ctx->getStart()->getLine(), sym, innerType);

        t->typeEnum = TYPE::MATRIX;

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitMatrix_type(GazpreaParser::Matrix_typeContext *ctx) {
        /*
         * matrix[*, *]
         */
#ifdef DEBUG
        std::cout << "visitMatrix [expr, expr] " << ctx->getText() << std::endl;
#endif
        auto innerType = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->built_in_type()));
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<MatrixTypeNode> t = std::make_shared<MatrixTypeNode>(ctx->getStart()->getLine(), sym, innerType);

        t->typeEnum = TYPE::MATRIX;
        t->sizeLeft = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression(0)));
        t->sizeRight = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitTuple_type(GazpreaParser::Tuple_typeContext *ctx) {
        /*
         * tuple[<type>, <type>, ...]
         */
#ifdef DEBUG
        std::cout << "visitTuple_type " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->getText());
        std::shared_ptr<TupleTypeNode> t = std::make_shared<TupleTypeNode>(ctx->getStart()->getLine(), sym);

        for (auto tupleElement : ctx->tuple_type_element()) {
            std::cout << "tuple type: " << tupleElement->getText() << std::endl;
            auto typeCtx = tupleElement->tuple_allowed_type();
            auto idCtx = tupleElement->ID();
            std::string idName = "";
            if (idCtx) {
                idName = idCtx->getSymbol()->getText();
            }
            t->innerTypes.push_back(std::make_pair(idName, std::any_cast<std::shared_ptr<ASTNode>>(visit(typeCtx))));
        }

        t->typeEnum = TYPE::TUPLE;
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    // EXPR


    std::any ASTBuilder::visitExpression(GazpreaParser::ExpressionContext *ctx) {
#ifdef DEBUG
        std::cout << "visitExpression (parent) " << ctx->getText() << std::endl;
#endif
        // just return the inner expression
        // the parent expression is just to help the grammar, so it's not needed here
        return visit(ctx->expr());
    }

    std::any ASTBuilder::visitIdentity(GazpreaParser::IdentityContext *ctx) {
#ifdef DEBUG
        std::cout << "visitIdentity " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<NullNode>(ctx->getStart()->getLine());

        return t;
    }

    std::any ASTBuilder::visitNull(GazpreaParser::NullContext *ctx) {
#ifdef DEBUG
        std::cout << "visitNull " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<NullNode>(ctx->getStart()->getLine());

        return t;
    }


    std::any ASTBuilder::visitLiteralID(GazpreaParser::LiteralIDContext *ctx) {
#ifdef DEBUG
        std::cout << "visitID " << ctx->ID()->getSymbol()->getText() << std::endl;
#endif
        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
        std::shared_ptr<ASTNode> t = std::make_shared<IDNode>(ctx->getStart()->getLine(), sym);

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitLiteralInt(GazpreaParser::LiteralIntContext *ctx) {
#ifdef DEBUG
        std::cout << "visitInt " << ctx->getText() << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<IntNode>(ctx->getStart()->getLine(),std::stoi(ctx->getText()));

        return t;
    }



    std::any ASTBuilder::visitMath(GazpreaParser::MathContext *ctx) {
#ifdef DEBUG
        std::cout << "visitMath, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<ArithNode> t = std::make_shared<ArithNode>(ctx->getStart()->getLine());

        switch (ctx->op->getType()) {
            case GazpreaParser::EXP:
                t->op = BINOP::EXP;
                break;
            case GazpreaParser::MULT:
                t->op = BINOP::MULT;
                break;
            case GazpreaParser::DIV:
                t->op = BINOP::DIV;
                break;
            case GazpreaParser::ADD:
                t->op = BINOP::ADD;
                break;
            case GazpreaParser::SUB:
                t->op = BINOP::SUB;
                break;
            case GazpreaParser::REM:
                t->op = BINOP::REM;
                break;
            case GazpreaParser::CONCAT:
                t->op = BINOP::CONCAT;
                break;
            case GazpreaParser::DOT_PRODUCT:
                t->op = BINOP::DOT_PROD;
                break;
        }
        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        // casting upward to an ASTNode
        // we want to use the .op attribute, so we don't want to cast it upward when initializing
        // like in the case of most other nodes
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitCmp(GazpreaParser::CmpContext *ctx) {
#ifdef DEBUG
        std::cout << "visitCmp, op = " << ctx->op->getText() << std::endl;
#endif
        std::shared_ptr<CmpNode> t = std::make_shared<CmpNode>(ctx->getStart()->getLine());

        switch (ctx->op->getType()) {
            case GazpreaParser::LT:
                t->op = BINOP::LTHAN;
                break;
            case GazpreaParser::GT:
                t->op = BINOP::GTHAN;
                break;
            case GazpreaParser::LE:
                t->op = BINOP::LEQ;
                break;
            case GazpreaParser::GE:
                t->op = BINOP::GEQ;
                break;
            case GazpreaParser::EQ:
                t->op = BINOP::EQUAL;
                break;
            case GazpreaParser::NEQ:
                t->op = BINOP::NEQUAL;
                break;
        }

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        // casting upward to an ASTNode
        // we want to use the .op attribute, so we don't want to cast it upward when initializing
        // like in the case of most other nodes
        return std::dynamic_pointer_cast<ASTNode>(t);
    }


    std::any ASTBuilder::visitAssign(GazpreaParser::AssignContext *ctx) {
//#ifdef DEBUG
//        std::cout << "visitAssign " << ctx->getStart()->getType() << ": "
//                  << ctx->ID()->getText() << std::endl;
//#endif
//        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
//        std::shared_ptr<AssignNode> t = std::make_shared<AssignNode>(ctx->getStart()->getLine(), sym);
//
//        t->addChild(visit(ctx->expression()));

//        return std::dynamic_pointer_cast<ASTNode>(t);
        return nullptr;
    }

    std::any ASTBuilder::visitCond(GazpreaParser::CondContext *ctx) {
#ifdef DEBUG
        std::cout << "visitCond " << std::endl;
#endif
        std::shared_ptr<ConditionalNode> t = std::make_shared<ConditionalNode>(ctx->getStart()->getLine());
        //t->condition = std::any_cast<std::shared_ptr<ASTNode>>(visit(ctx->expression()));
        //for (auto statement: ctx->statement()) {
        //    t->addChild(visit(statement));
        //}

        //return std::dynamic_pointer_cast<ASTNode>(t);
        return nullptr;
    }

    std::any ASTBuilder::visitIndex(GazpreaParser::IndexContext *ctx) {
#ifdef DEBUG
        std::cout << "visitIndex" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<IndexNode>(ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitRange(GazpreaParser::RangeContext *ctx) {
#ifdef DEBUG
        std::cout << "visitRange" << std::endl;
#endif
        std::shared_ptr<ASTNode> t = std::make_shared<RangeVecNode>(ctx->getStart()->getLine());

        t->addChild(visit(ctx->expr(0)));
        t->addChild(visit(ctx->expr(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitGenerator(GazpreaParser::GeneratorContext *ctx) {
#ifdef DEBUG
        std::cout << "visitGenerator" << std::endl;
#endif
        std::string domainVar = ctx->ID(0)->getSymbol()->getText();
        std::shared_ptr<ASTNode> t = std::make_shared<GeneratorNode>(domainVar, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitFilter(GazpreaParser::FilterContext *ctx) {
#ifdef DEBUG
        std::cout << "visitFilter" << std::endl;
#endif
        std::string domainVar = ctx->ID()->getSymbol()->getText();
        std::shared_ptr<ASTNode> t = std::make_shared<FilterNode>(domainVar, ctx->getStart()->getLine());

        t->addChild(visit(ctx->expression(0)));
        t->addChild(visit(ctx->expression(1)));

        return std::dynamic_pointer_cast<ASTNode>(t);
    }



    std::any ASTBuilder::visitFunctionSingle(GazpreaParser::FunctionSingleContext *ctx) {
        std::cout << "visiting function Single\n";
        // ctx->ID(0) is always the function name, kind of mystery indexing
        std::shared_ptr<Symbol> funcNameSym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
        std::shared_ptr<FunctionSingleNode> t = std::make_shared<FunctionSingleNode>(ctx->getStart()->getLine(), funcNameSym);

        auto typesArray = ctx->type();
        auto argIDArray = ctx->ID();
        // TODO: add the retType node(doesnt work yet without visitType)
        //t->addChild(visit(typesArray[typesArray.size() - 1])); // the last type is always the return type?

        // iterate thru all the orderedArg
        // ctx->ID() is an array of arguments id, skip ID(0) because thats the function name
        // type of the argument ID(i) is at ctx->type(i - 1)
        for (long unsigned int i = 1; i < argIDArray.size(); i++) {
            // TODO: cant visit type yet since its not yet done
            //auto argTypeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(typesArray[i - 1]));
            auto  sym = std::make_shared<Symbol>(argIDArray[i]->getSymbol()->getText());
            auto argIDnode = std::make_shared<IDNode>(argIDArray[i]->getSymbol()->getLine(), sym);
            // TODO:  set the type of the ID node once visit Type is implemented
            //argIDnode->type = std::any_cast<std::shared_ptr<Type>>(visit(typesArray[i - 1]));

            t->orderedArgs.push_back(argIDnode);

        }
        t->addChild(visit(ctx->expression()));
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitFunctionBlock(GazpreaParser::FunctionBlockContext *ctx) {
        std::cout << "visiting function block\n";
        // ctx->ID(0) is always the function name, kind of mystery indexing
        std::shared_ptr<Symbol> funcNameSym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
        std::shared_ptr<FunctionBlockNode> t = std::make_shared<FunctionBlockNode>(ctx->getStart()->getLine(), funcNameSym);

        auto typesArray = ctx->type();
        auto argIDArray = ctx->ID();
        // TODO: add the retType node(doesnt work yet without type walker)
        //t->addChild(visit(typesArray[typesArray.size() - 1])); // the last type is always the return type?

        // iterate thru all the orderedArg
        // ctx->ID() is an array of arguments id, skip ID(0) because thats the function name
        // type of the argument ID(i) is at ctx->type(i - 1)
        for (long unsigned int i = 1; i < argIDArray.size(); i++) {
            // TODO: cant visit type yet since its not yet done
            //auto argTypeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(typesArray[i - 1]));
            auto  sym = std::make_shared<Symbol>(argIDArray[i]->getSymbol()->getText());
            auto argIDnode = std::make_shared<IDNode>(argIDArray[i]->getSymbol()->getLine(), sym);
            // TODO:  set the type of the ID node once visit Type is implemented
            //argIDnode->type = std::any_cast<std::shared_ptr<Type>>(visit(typesArray[i - 1]));

            t->orderedArgs.push_back(argIDnode);

        }
        // TODO: add block node when visitBlock is implemented
        //t->addChild(visit(ctx->block()));
        return std::dynamic_pointer_cast<ASTNode>(t);

    }

    // func decl node, holds retType,
    std::any ASTBuilder::visitFunctionForward(GazpreaParser::FunctionForwardContext *ctx) {
        std::cout << "visiting function forwward\n";
        // ctx->ID(0) is always the function name, kind of mystery indexing
        std::shared_ptr<Symbol> funcNameSym = std::make_shared<Symbol>(ctx->ID(0)->getSymbol()->getText());
        std::shared_ptr<FunctionForwardNode> t = std::make_shared<FunctionForwardNode>(ctx->getStart()->getLine(), funcNameSym);

        auto typesArray = ctx->type();
        auto argIDArray = ctx->ID();
        // TODO: add the retType node(doesnt work yet without type walker)
        //t->addChild(visit(typesArray[typesArray.size() - 1])); // the last type is always the return type?

        // iterate thru all the orderedArg
        // ctx->ID() is an array of arguments id, skip ID(0) because thats the function name
        // type of the argument ID(i) is at ctx->type(i - 1)
        for (long unsigned int i = 1; i < argIDArray.size(); i++) {
            // TODO: cant visit type yet since its not yet done
            //auto argTypeNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(typesArray[i - 1]));
            auto  sym = std::make_shared<Symbol>(argIDArray[i]->getSymbol()->getText());
            auto argIDnode = std::make_shared<IDNode>(argIDArray[i]->getSymbol()->getLine(), sym);
            // TODO:  set the type of the ID node once visit Type is implemented
            //argIDnode->type = std::any_cast<std::shared_ptr<Type>>(visit(typesArray[i - 1]));

            t->orderedArgs.push_back(argIDnode);

        }
        // doesnt have the block expression

        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitFunction_call(GazpreaParser::Function_callContext *ctx) {
        std::shared_ptr<FunctionCallNode> t;
        if (ctx->RESERVED_COLUMNS()) {

        } else if (ctx->RESERVED_FORMAT()) {

        } else if (ctx->RESERVED_LENGTH()) {

        } else if (ctx->RESERVED_REVERSE()) {

        } else if (ctx->RESERVED_ROWS()) {

        } else if (ctx->RESERVED_STD_INPUT()) {

        } else if (ctx->RESERVED_STREAM_STATE()) {

        } else if (ctx->ID()) {
            t = std::make_shared<FunctionCallNode>(ctx->ID()->getSymbol()->getLine(), FUNCTYPE::FUNC_NORMAL);
            std::shared_ptr<Symbol> funcCallSym = std::make_shared<Symbol>(ctx->ID()->getSymbol()->getText());
            t->funcCallName = funcCallSym;
        }
        return std::dynamic_pointer_cast<ASTNode>(t);
    }

    std::any ASTBuilder::visitFuncCall(GazpreaParser::FuncCallContext *ctx) {
        std::shared_ptr<ASTNode> funcNode;
        assert(ctx->children.size() == 1);
        for (auto child: ctx->children) {
            // should only have one child
            funcNode = std::any_cast<std::shared_ptr<ASTNode>>(visit(child));
        }
        return funcNode;
    }
}
