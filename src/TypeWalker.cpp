#include "TypeWalker.h"
#include "BuiltInTypeSymbol.h"
#define DEBUG

namespace gazprea {

    PromotedType::PromotedType() {}
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

    std::shared_ptr<Type> TypeWalker::getType(std::string table[5][5], std::shared_ptr<Type> left, std::shared_ptr<Type> right, std::shared_ptr<ASTNode> t) {
        auto leftIndex = this->getTypeIndex(left->getName());
        auto rightIndex = this->getTypeIndex(right->getName());
        // TODO: identity and null handling
        std::string resultTypeString = table[leftIndex][rightIndex];
        if (resultTypeString.empty()) {
            throw TypeError(t->loc(), "Cannot perform operation between " + left->getName() + " and " + right->getName());
        }

        auto resultType = std::make_shared<BuiltInTypeSymbol>(resultTypeString); // maybe need to resolve this instead?

        #ifdef DEBUG
                std::cout << "type promotions between " <<  left->getName() << ", " << right->getName() << "\n";
                std::cout << "result: " <<  resultType->getName() << "\n";
        #endif
        return resultType;
    }

    int TypeWalker::getTypeIndex(const std::string type) {
        if (type == "boolean") {
            return this->boolIndex;
        } else if (type == "character") {
            return this->charIndex;
        } else if (type == "int") {
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

    std::any TypeWalker::visitID(std::shared_ptr<IDNode> tree) {
        if (tree->sym == nullptr) {
            throw SymbolError(tree->loc(), "Undefined Symbol " + tree->getName() + " Referenced");
        }

    }










}