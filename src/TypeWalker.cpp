#include "TypeWalker.h"
#define DEBUG

namespace gazprea {

    PromotedType::PromotedType(std::shared_ptr<SymbolTable> symtab) : symtab(symtab), currentScope(symtab->globalScope) {}
    PromotedType::~PromotedType() {}

    std::string PromotedType::booleanResult[5][5] = {
/*                      boolean   character    integer  real  tuple */
/*boolean*/  {"boolean",  "",         "",         "",       "" },
/*character*/{"",         "",         "",         "",       "" },
/*integer*/  {"",         "",         "",         "",       "" },
/*real*/     {"",         "",         "",         "",       "" },
/*tuple*/    {"",         "",         "",         "",       "" }
    };


    std::string PromotedType::arithmeticResult[5][5] = {
/*                      boolean   character    integer  real  tuple */
/*boolean*/  {"",         "",         "",         "",       "" },
/*character*/{"",         "",         "",         "",       "" },
/*integer*/  {"",         "",         "integer",  "real",   "" },
/*real*/     {"",         "",         "real",     "real",   "" },
/*tuple*/    {"",         "",         "",         "",       "" }
    };

    std::string PromotedType::comparisonResult[5][5] = {
/*                      boolean   character    integer  real  tuple */
/*boolean*/  {"",         "",         "",         "",       "" },
/*character*/{"",         "",         "",         "",       "" },
/*integer*/  {"",         "",         "boolean",  "boolean","" },
/*real*/     {"",         "",         "boolean",  "boolean","" },
/*tuple*/    {"",         "",         "",         "",       "" }
    };

    std::string PromotedType::equalityResult[5][5] = {
/*                      boolean   character    integer  real  tuple */
/*boolean*/  {"boolean",  "",         "",         "",        "" },
/*character*/{"",         "boolean",  "",         "",        "" },
/*integer*/  {"",         "",         "boolean",  "boolean", "" },
/*real*/     {"",         "",         "boolean",  "boolean", "" },
/*tuple*/    {"",         "",         "",         "",        "boolean" }
    };

    std::string PromotedType::promotionTable[5][5] = {
/*                      boolean   character    integer  real  tuple */
/*boolean*/      {"boolean",  "",         "",         "",        "" },
/*character*/    {"",         "boolean",  "",         "",        "" },
/*integer*/      {"",         "",         "",         "real",    "" },
/*real*/         {"",         "",         "",         "",        "" },
/*tuple*/        {"",         "",         "",         "",        "" }
// TODO: Add identity and null support promotion when Def Ref is done.
    };

    std::shared_ptr<Type> PromotedType::getType(std::string table[5][5], std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right, std::shared_ptr<ASTNode> t) {
        if (left->evaluatedType == nullptr || right->evaluatedType == nullptr) {
            return nullptr;
        }
        // TODO: identity and null handling
        auto leftIndex = this->getTypeIndex(left->evaluatedType->getName());
        auto rightIndex = this->getTypeIndex(right->evaluatedType->getName());
        std::string resultTypeString = table[leftIndex][rightIndex];
        if (resultTypeString.empty()) {
            if (table == promotionTable) {
                throw TypeError(t->loc(), "Cannot implicitly promote " + left->evaluatedType->getName() + " to " + right->evaluatedType->getName());
            }
            else {
                throw TypeError(t->loc(), "Cannot perform operation between " + left->evaluatedType->getName() + " and " + right->evaluatedType->getName());
            }
        }

        auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));

        #ifdef DEBUG
                std::cout << "type promotions between " <<  left->evaluatedType->getName() << ", " << right->evaluatedType->getName() << "\n";
                std::cout << "result: " <<  resultType->getName() << "\n";
        #endif
        assert(resultType);
        return resultType;
    }

    int PromotedType::getTypeIndex(const std::string type) {
        if (type == "boolean") {
            return this->boolIndex;
        } else if (type == "character") {
            return this->charIndex;
        } else if (type == "integer") {
            return this->integerIndex;
        } else if (type == "real") {
            return this->realIndex;
        } else if (type == "tuple") {
            return this->tupleIndex;
        } else {
            throw std::runtime_error("Unknown type");
        }
    }

    TypeWalker::TypeWalker(std::shared_ptr<SymbolTable> symtab, std::shared_ptr<PromotedType> promotedType) : symtab(symtab), currentScope(symtab->globalScope), promotedType(promotedType) {}
    TypeWalker::~TypeWalker() {}

    std::any TypeWalker::visitInt(std::shared_ptr<IntNode> tree) {
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(currentScope->resolveType("integer"));
        return nullptr;
    }

    std::any TypeWalker::visitReal(std::shared_ptr<RealNode> tree) {
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(currentScope->resolveType("real"));
        return nullptr;
    }

    std::any TypeWalker::visitChar(std::shared_ptr<CharNode> tree) {
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(currentScope->resolveType("character"));
        return nullptr;
    }

    std::any TypeWalker::visitBool(std::shared_ptr<BoolNode> tree) {
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(currentScope->resolveType("boolean"));
        return nullptr;
    }

    std::any TypeWalker::visitArith(std::shared_ptr<BinaryArithNode> tree) {
        walkChildren(tree);
        auto lhsType = tree->getLHS()->evaluatedType;
        auto rhsType = tree->getRHS()->evaluatedType;
        auto op = tree->op;
        switch(op) {
            case BINOP::MULT:
            case BINOP::DIV:
            case BINOP::ADD:
            case BINOP::EXP:
            case BINOP::SUB:
            case BINOP::REM:
                tree->evaluatedType = promotedType->getType(promotedType->arithmeticResult, tree->getLHS(), tree->getRHS(), tree);
                break;
            case BINOP::XOR:
            case BINOP::OR:
            case BINOP::AND:
                tree->evaluatedType = promotedType->getType(promotedType->booleanResult, tree->getLHS(), tree->getRHS(), tree);
                break;
        }
        return nullptr;
    }

    std::any TypeWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree) {
        walkChildren(tree);
        auto lhsType = tree->getLHS()->evaluatedType;
        auto rhsType = tree->getRHS()->evaluatedType;
        auto op = tree->op;
        switch(op) {
            case BINOP::LTHAN:
            case BINOP::GTHAN:
            case BINOP::LEQ:
            case BINOP::GEQ:
                tree->evaluatedType = promotedType->getType(promotedType->comparisonResult, tree->getLHS(), tree->getRHS(), tree);
                break;
            case BINOP::EQUAL:
            case BINOP::NEQUAL:
                tree->evaluatedType = promotedType->getType(promotedType->equalityResult, tree->getLHS(), tree->getRHS(), tree);
                break;
        }
        return nullptr;
    }

    std::any TypeWalker::visitUnaryArith(std::shared_ptr<UnaryArithNode> tree) {
        walkChildren(tree);
        tree->evaluatedType = tree->getExpr()->evaluatedType;
        return nullptr;
    }

    std::any TypeWalker::visitID(std::shared_ptr<IDNode> tree) {
        if(tree->sym == nullptr) {
            throw SymbolError(tree->loc(), "Unidentified Symbol referenced!");
        }
        tree->evaluatedType = tree->sym->typeSym;
        return nullptr;
    }

    std::any TypeWalker::visitAssign(std::shared_ptr<AssignNode> tree) {
        walkChildren(tree);
        auto rhsType = tree->getRvalue()->evaluatedType;
        auto lhsCount = tree->getLvalue()->children.size();
        auto exprList = std::dynamic_pointer_cast<ExprListNode>(tree->getLvalue());

        if (lhsCount == 1) {
            if(std::dynamic_pointer_cast<IDNode>(exprList->children[0])) {
                auto lvalue = std::dynamic_pointer_cast<IDNode>(exprList->children[0]);
                auto symbol = lvalue->sym;
                if (symbol != nullptr and symbol->qualifier == QUALIFIER::CONST) {
                    throw AssignError(tree->loc(), "Cannot assign to const");
                }
            }
            // TODO else tupleIndex

            if (rhsType != nullptr) {
                if (tree->getLvalue()->children[0]->evaluatedType == nullptr)
                    return nullptr;
                // TODO tuple handling and identity, null handling
                if(std::dynamic_pointer_cast<IDNode>(exprList->children[0])) {
                    auto lvalue = std::dynamic_pointer_cast<IDNode>(exprList->children[0]);

                    if (tree->getRvalue()->evaluatedType->getName() != tree->getLvalue()->children[0]->evaluatedType->getName())
                        tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getRvalue(), lvalue, tree);
                    else
                        tree->evaluatedType = tree->getRvalue()->evaluatedType;
                }
                // TODO else tupleIndex
            }
        }
        //TODO tuple unpack / assignment
        return nullptr;
    }
}
