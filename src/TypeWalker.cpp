#include "TypeWalker.h"
#include "ASTNode/Type/TupleTypeNode.h"
//#define DEBUG

// until we get more typecheck done

namespace gazprea {

    PromotedType::PromotedType(std::shared_ptr<SymbolTable> symtab) : symtab(symtab), currentScope(symtab->globalScope) {}
    PromotedType::~PromotedType() {}

    std::string PromotedType::booleanResult[7][7] = {
/*                      boolean   character    integer  real  tuple identity null:*/
/*boolean*/  {"boolean",  "",         "",         "",       "",  "boolean", ""},
/*character*/{"",         "",         "",         "",       "",  "character", ""},
/*integer*/  {"",         "",         "",         "",       "" , "integer", ""},
/*real*/     {"",         "",         "",         "",       "" , "real", ""},
/*tuple*/    {"",         "",         "",         "",       "" , "tuple", ""},
/*identity*/ {"boolean",  "character", "integer", "real",  "tuple" , "", ""},
/*null*/     {"",  "", "", "", "" , "", ""}
    };

    std::string PromotedType::castTable[4][4] = {
/* from\to                     boolean   character    integer  real  */
/*boolean*/  {"boolean",  "character",   "integer",  "real"},
/*character*/{"boolean",  "character",   "integer",  "real"},
/*integer*/  {"boolean",  "character",   "integer",  "real"},
/*real*/     {"",         "",            "integer",  "real"},
    };

    std::string PromotedType::arithmeticResult[7][7] = {
/*                      boolean   character    integer  real  tuple identity null*/
/*boolean*/  {"",         "",         "",         "",       "" , "boolean", ""},
/*character*/{"",         "",         "",         "",       "" , "character", ""},
/*integer*/  {"",         "",         "integer",  "real",   "" , "integer", ""},
/*real*/     {"",         "",         "real",     "real",   "" , "real", ""},
/*tuple*/    {"",         "",         "",         "",       "" , "tuple", ""},
/*identity*/ {"boolean",  "character", "integer", "real",  "tuple" , "", ""},
/*null*/         {"",  "", "", "", "" , "", ""}
    };

    std::string PromotedType::comparisonResult[7][7] = {
/*                      boolean   character    integer  real  tuple identity null*/
/*boolean*/  {"",         "",         "",         "",       "" , "boolean", ""},
/*character*/{"",         "",         "",         "",       "" , "character", ""},
/*integer*/  {"",         "",         "boolean",  "boolean","" , "integer", ""},
/*real*/     {"",         "",         "boolean",  "boolean","" , "real", ""},
/*tuple*/    {"",         "",         "",         "",       "" , "tuple", ""},
/*identity*/ {"boolean",  "character", "integer", "real",  "tuple" , "", ""},
/*null*/         {"",  "", "", "", "" , "", ""}
    };

    std::string PromotedType::equalityResult[7][7] = {
/*                      boolean   character    integer  real  tuple identity null*/
/*boolean*/  {"boolean",  "",         "",         "",        "" ,       "boolean", ""},
/*character*/{"",         "boolean",  "",         "",        "" ,       "character", ""},
/*integer*/  {"",         "",         "boolean",  "boolean", "" ,       "integer", ""},
/*real*/     {"",         "",         "boolean",  "boolean", "" ,       "real", ""},
/*tuple*/    {"",         "",         "",         "",        "boolean", "tuple", ""},
/*identity*/ {"boolean",  "character", "integer", "real", "tuple",      "", ""},
/*null*/         {"",  "", "", "", "" , "", ""}
    };

    std::string PromotedType::promotionTable[7][7] = {
/*                      boolean   character    integer  real  tuple identity null*/
/*boolean*/      {"boolean",  "",         "",         "",        "",   "boolean", ""},
/*character*/    {"",         "character","",         "",        "",   "character", ""},
/*integer*/      {"",         "",         "integer",  "real",    "",   "integer", ""},
/*real*/         {"",         "",         "",         "real",    "",   "real", ""},
/*tuple*/        {"",         "",         "",         "",        "",   "tuple", ""},
/*identity*/     {"boolean",  "character", "integer", "real", "tuple", "", ""},
/*null*/         {"",  "", "", "", "" , "", ""}
// TODO: Add identity and null support promotion when Def Ref is done.
    };

    std::shared_ptr<Type> PromotedType::getType(std::string table[7][7], std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right, std::shared_ptr<ASTNode> t) {
        if (left->evaluatedType == nullptr || right->evaluatedType == nullptr) {
            return nullptr;
        }
        // TODO: identity and null handling
        auto leftIndex = this->getTypeIndex(left->evaluatedType->getBaseTypeEnumName());
        auto rightIndex = this->getTypeIndex(right->evaluatedType->getBaseTypeEnumName());
        std::string resultTypeString = table[leftIndex][rightIndex];
        if (resultTypeString.empty()) {
            if (table == promotionTable) {
                throw TypeError(t->loc(), "Cannot implicitly promote " + left->evaluatedType->getBaseTypeEnumName() + " to " + right->evaluatedType->getBaseTypeEnumName());
            }
            else {
                throw TypeError(t->loc(), "Cannot perform operation between " + left->evaluatedType->getBaseTypeEnumName() + " and " + right->evaluatedType->getBaseTypeEnumName());
            }
        }

        auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));

        #ifdef DEBUG
                std::cout << "type promotions between " <<  left->evaluatedType->getBaseTypeEnumName() << ", " << right->evaluatedType->getBaseTypeEnumName() << "\n";
                std::cout << "result: " <<  resultType->getBaseTypeEnumName() << "\n";
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
        } else if (type == "identity") {
            return this->identityIndex;
        } else if (type == "null") {
            return this->nullIndex;
        } else {
                throw std::runtime_error("Unknown type");
        }
    }
    void PromotedType::promoteVectorElements(std::shared_ptr<Type> promoteTo, std::shared_ptr<ASTNode> exprNode) {
        // only care about vector expr node
        if (exprNode->evaluatedType->vectorOrMatrixEnum != TYPE::NONE) {
            return;
        }
        assert(exprNode->evaluatedType->getBaseTypeEnumName() == "vector");
        auto exprNodeCast = std::dynamic_pointer_cast<VectorNode>(exprNode);
        // promote each vector elements
        for (auto &child: exprNodeCast->getElements()) {
            auto rhsIndex = getTypeIndex(child->evaluatedType->getBaseTypeEnumName());
            auto lhsIndex = getTypeIndex(promoteTo->getBaseTypeEnumName());
            auto promoteTypeString = promotionTable[rhsIndex][lhsIndex];
            if (promoteTypeString.empty()) throw  TypeError(exprNode->loc(), "cannot promote vector element");
            auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(promoteTypeString));
#ifdef DEBUG
            std::cout << "promoted vector element " << child->evaluatedType->getBaseTypeEnumName() << "to " << promoteTo->getBaseTypeEnumName() << "\n";
#endif
            child->evaluatedType = resultType;  // set each vector element node to its promoted type
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

    std::any TypeWalker::visitIdentity(std::shared_ptr<IdentityNode> tree) {
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(currentScope->resolveType("identity"));
        return nullptr;
    }
    std::any TypeWalker::visitNull(std::shared_ptr<NullNode> tree) {
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(currentScope->resolveType("null"));
        return nullptr;
    }

    std::any TypeWalker::visitTuple(std::shared_ptr<TupleNode> tree) {
        for (auto expr: tree->val) {
            walk(expr);
        }

        std::shared_ptr<Symbol> sym = std::make_shared<Symbol>("_");
        auto tupleType = std::dynamic_pointer_cast<Type>(std::make_shared<AdvanceType>("tuple"));

        for (auto tupleVal : tree->val) {
            auto childType = tupleVal->evaluatedType;
            tupleType->tupleChildType.push_back(std::make_pair("", childType));
        }
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(tupleType);
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
                if (lhsType->getBaseTypeEnumName() == "identity") tree->getLHS()->evaluatedType = tree->evaluatedType;  // promote LHS
                if (rhsType->getBaseTypeEnumName() == "identity") tree->getRHS()->evaluatedType = tree->evaluatedType;  // promote RHS
                break;
            case BINOP::XOR:
            case BINOP::OR:
            case BINOP::AND:
                tree->evaluatedType = promotedType->getType(promotedType->booleanResult, tree->getLHS(), tree->getRHS(), tree);
                if (lhsType->getBaseTypeEnumName() == "identity") tree->getLHS()->evaluatedType = tree->evaluatedType;  // promote LHS
                if (rhsType->getBaseTypeEnumName() == "identity") tree->getRHS()->evaluatedType = tree->evaluatedType;  // promote RHS
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

    std::any TypeWalker::visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) {
        walkChildren(tree);
        auto tupleId = std::dynamic_pointer_cast<IDNode>(tree->children[0]);
        if (std::dynamic_pointer_cast<IDNode>(tree->children[1])) {
            auto index = std::dynamic_pointer_cast<IDNode>(tree->children[1])->getName();
            for (auto c: tupleId->evaluatedType->tupleChildType) {
                if (index == c.first) {
                    tree->evaluatedType = std::dynamic_pointer_cast<Type>(c.second);
                    break;
                }
            }
            if (!tree->evaluatedType) {
                throw SymbolError(tree->loc(), "Undefined tuple index referenced");
            }
        }
        else if (std::dynamic_pointer_cast<IntNode>(tree->children[1])) {
            auto index = std::dynamic_pointer_cast<IntNode>(tree->children[1])->getVal();
            if (index < 1 || index > tupleId->evaluatedType->tupleChildType.size()) {
                throw SymbolError(tree->loc(), "Out of bound tuple index referenced");
            }
            tree->evaluatedType = std::dynamic_pointer_cast<Type>(tupleId->evaluatedType->tupleChildType[index-1].second);
        }
        return nullptr;
    }

    std::any TypeWalker::visitDecl(std::shared_ptr<DeclNode> tree) {
        walkChildren(tree);
        if (!tree->getTypeNode()) {
            tree->sym->typeSym = tree->getExprNode()->evaluatedType;
            if (tree->getExprNode()->evaluatedType->baseTypeEnum == TYPE::IDENTITY)  {
               throw TypeError(tree->loc(), "cannot have identity when type is not defined");
            }
            tree->evaluatedType = tree->getExprNode()->evaluatedType;
            return nullptr;
        }
        if(!tree->getExprNode()) {
            // has a type node //  add a null node :) TODO: this identity and null handling is different for matrices and vector
            std::shared_ptr<ASTNode> nullNode = std::make_shared<NullNode>(tree->loc());
            tree->addChild(nullNode);
            walk(nullNode);  // walk null node to popualte the type

            auto lType = tree->sym->typeSym;
            if (lType == nullptr) {
                throw SyntaxError(tree->loc(), "Declaration is missing expression to infer type.");
            }
            tree->evaluatedType = lType;  //
            tree->getExprNode()->evaluatedType = lType;  // set identity/null node type to this type for promotion
            return nullptr;
        }

        auto lType = tree->sym->typeSym;
        auto rType = tree->getExprNode()->evaluatedType;

        if (rType == nullptr) {
            return nullptr;
        }

        // I think this is already handled in ref pass
        if (lType == nullptr) {
            throw SyntaxError(tree->loc(), "Declaration is missing expression to infer type.");
        }

        if (lType->getBaseTypeEnumName() == "tuple") {
            auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->getTypeNode());
            for (size_t i = 0; i < tupleNode->innerTypes.size(); i++) {
                auto leftTypeString = std::dynamic_pointer_cast<TypeNode>(tupleNode->innerTypes[i].second)->getTypeName();

                if (leftTypeString == "tuple") {
                    throw TypeError(tree->loc(), "Cannot have tuple as a tuple member");
                }
            }
        }

        if (lType->getBaseTypeEnumName() == "tuple" && rType->getBaseTypeEnumName() == "tuple") {
            auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->getTypeNode());
            if (tupleNode->innerTypes.size() != rType->tupleChildType.size()) {
                throw TypeError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
            }
            for (size_t i = 0; i < rType->tupleChildType.size(); i++) {
                auto leftTypeString = std::dynamic_pointer_cast<TypeNode>(tupleNode->innerTypes[i].second)->getTypeName();
                auto rightTypeString = rType->tupleChildType[i].second->getBaseTypeEnumName();

                if (leftTypeString != rightTypeString) {
                    auto leftIndex = promotedType->getTypeIndex(rightTypeString);
                    auto rightIndex = promotedType->getTypeIndex(leftTypeString);
                    std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
                    if (resultTypeString.empty()) {
                        throw TypeError(tree->loc(), "Cannot implicitly promote " + rightTypeString + " to " + leftTypeString);
                    }
                }
            }
            tree->evaluatedType = rType;
        } else if (rType->getBaseTypeEnumName() == "null" || rType ->getBaseTypeEnumName() == "identity") {  // if it null then we just set it to ltype
            tree->evaluatedType = lType;  //
            tree->getExprNode()->evaluatedType = lType;  // set identity/null node type to this type for promotion
        } else if (lType->vectorOrMatrixEnum == TYPE::VECTOR) {
            // promote all RHS vector element to ltype if exprNode is a vectorNode
            promotedType->promoteVectorElements(lType, tree->getExprNode());
            tree->getExprNode()->evaluatedType->baseTypeEnum = lType->baseTypeEnum;  // set the LHS vector literal type. int?real?
            tree->getExprNode()->evaluatedType->setName(lType->getBaseTypeEnumName());  // set the string evaluated type
            tree->evaluatedType = tree->getExprNode()->evaluatedType;  // copy the vectorLiteral's type into this node
        }
        else if (lType->getBaseTypeEnumName() != rType->getBaseTypeEnumName()) {
            auto leftIndex = promotedType->getTypeIndex(rType->getBaseTypeEnumName());
            auto rightIndex = promotedType->getTypeIndex(lType->getBaseTypeEnumName());
            std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
            if (resultTypeString.empty()) {
                throw TypeError(tree->loc(), "Cannot implicitly promote " + rType->getBaseTypeEnumName() + " to " + lType->getBaseTypeEnumName());
            }
            auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));
            tree->evaluatedType = resultType;
        }
        else {  // normal implicit promotion
            tree->evaluatedType = rType;
        }
        return nullptr;
    }
    std::any TypeWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) {
        walk(tree->getCondition());
        if (tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::IDENTITY || tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::NULL_) {
            throw TypeError(tree->loc(), "idenitty in loop cond");
        }
        return nullptr;
    }
    std::any TypeWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) {
        walk(tree->getCondition());
        if (tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::IDENTITY || tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::NULL_) {
            throw TypeError(tree->loc(), "idenitty in loop cond");
        }
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

            else if(std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0])) {
                auto lvalue = std::dynamic_pointer_cast<IDNode>(std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0])->children[0]);
                auto symbol = lvalue->sym;
                if (symbol != nullptr and symbol->qualifier == QUALIFIER::CONST) {
                    throw AssignError(tree->loc(), "Cannot assign to const");
                }
            }

            else
                throw SyntaxError(tree->loc(), "vector and matrix indexing not implemented yet/ incorrect lvalue");


            if (rhsType != nullptr) {
                if (tree->getLvalue()->children[0]->evaluatedType == nullptr) {
                    tree->evaluatedType = rhsType;
                    return nullptr;
                }
                // TODO identity and null handling
                if(std::dynamic_pointer_cast<IDNode>(exprList->children[0])) {
                    auto lvalue = std::dynamic_pointer_cast<IDNode>(exprList->children[0]);

                    if (lvalue->evaluatedType->getBaseTypeEnumName() == "tuple" and rhsType->getBaseTypeEnumName() == "tuple") {
                        if (lvalue->evaluatedType->tupleChildType.size() != rhsType->tupleChildType.size()) {
                            throw TypeError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
                        }
                        for (size_t i = 0; i < rhsType->tupleChildType.size(); i++) {
                            auto leftTypeString = lvalue->evaluatedType->tupleChildType[i].second->getBaseTypeEnumName();
                            auto rightTypeString = rhsType->tupleChildType[i].second->getBaseTypeEnumName();

                            if (leftTypeString != rightTypeString) {
                                auto leftIndex = promotedType->getTypeIndex(rightTypeString);
                                auto rightIndex = promotedType->getTypeIndex(leftTypeString);
                                std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
                                if (resultTypeString.empty()) {
                                    throw TypeError(tree->loc(), "Cannot implicitly promote " + rightTypeString + " to " + leftTypeString);
                                }
                            }
                        }
                        tree->evaluatedType = lvalue->evaluatedType;
                    } else if (rhsType->getBaseTypeEnumName() == "null" || rhsType->getBaseTypeEnumName() == "identity") {  // if it null then we just set it to ltype
                        tree->evaluatedType = lvalue->evaluatedType;  //
                        tree->getRvalue()->evaluatedType = lvalue->evaluatedType;  // set identity/null node type to this type for promotion
                    }

                    if (tree->getRvalue()->evaluatedType->getBaseTypeEnumName() != tree->getLvalue()->children[0]->evaluatedType->getBaseTypeEnumName())
                        tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getRvalue(), lvalue, tree);
                    else
                        tree->evaluatedType = tree->getRvalue()->evaluatedType;
                }
                else if (std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0])) {
                    auto tupleIndexNode = std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0]);
                    if (rhsType->getBaseTypeEnumName() == "tuple") {
                        throw TypeError(tree->loc(), "Cannot assign tuple to tuple index");
                    }
                    if (tree->getRvalue()->evaluatedType->getBaseTypeEnumName() != tupleIndexNode->evaluatedType->getBaseTypeEnumName())
                        tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getRvalue(), tupleIndexNode, tree);
                    else
                        tree->evaluatedType = tree->getRvalue()->evaluatedType;
                }
                else
                    throw SyntaxError(tree->loc(), "vector and matrix indexing not implemented yet/ incorrect lvalue");
            }
        }
        else {
            if (rhsType->getBaseTypeEnumName() != "tuple")
                throw TypeError(tree->loc(), "Tuple Unpacking requires a tuple as the r-value.");
            if (lhsCount != rhsType->tupleChildType.size())
                throw AssignError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
            for (size_t i = 0; i < rhsType->tupleChildType.size(); i++) {

                std::string leftTypeString;
                if (std::dynamic_pointer_cast<IDNode>(exprList->children[i])) {
                    auto lvalue = std::dynamic_pointer_cast<IDNode>(exprList->children[i]);
                    leftTypeString = lvalue->evaluatedType->getBaseTypeEnumName();
                }
                else if (std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[i])) {
                    auto lvalue = std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[i]);
                    leftTypeString = lvalue->evaluatedType->getBaseTypeEnumName();
                }

                if (leftTypeString.empty()) {
                    throw SyntaxError(tree->loc(), "vector and matrix indexing not implemented yet/ incorrect lvalue");
                }

                auto rightTypeString = rhsType->tupleChildType[i].second->getBaseTypeEnumName();

                if (leftTypeString != rightTypeString) {
                    auto leftIndex = promotedType->getTypeIndex(rightTypeString);
                    auto rightIndex = promotedType->getTypeIndex(leftTypeString);
                    std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
                    if (resultTypeString.empty()) {
                        throw TypeError(tree->loc(), "Cannot implicitly promote " + rightTypeString + " to " + leftTypeString);
                    }
                }
            }
        }
        return nullptr;
    }

    std::any TypeWalker::visitStreamOut(std::shared_ptr<StreamOut> tree) {
        // streamOut supports the following types:
        // - char, integer, real, boolean
        // - vector, string, matrix (part 2)
        // basically, NOT tuples
        std::vector<TYPE> allowedTypes = {TYPE::CHAR, TYPE::INTEGER, TYPE::REAL, TYPE::BOOLEAN,  TYPE::STRING};

        walkChildren(tree);
        auto exprType = tree->getExpr()->evaluatedType;
        if (exprType != nullptr) {
            if (std::find(allowedTypes.begin(), allowedTypes.end(), exprType->baseTypeEnum) == allowedTypes.end()) {
                throw TypeError(tree->loc(), "Cannot stream out a " + typeEnumToString(exprType->baseTypeEnum));
            } else if (exprType->vectorOrMatrixEnum == TYPE::VECTOR) {
               // check if vector type is in

            }
        } else {
            throw TypeError(tree->loc(), "Cannot stream out unknown type");
        }
        return nullptr;
    }

    std::any TypeWalker::visitStreamIn(std::shared_ptr<StreamIn> tree) {
        // streamIn supports reading into the following types:
        // - char, integer, real, boolean
        // NOT any other types
        std::vector<TYPE> allowedTypes = {TYPE::CHAR, TYPE::INTEGER, TYPE::REAL, TYPE::BOOLEAN};
        walkChildren(tree);
        auto exprType = tree->getExpr()->evaluatedType;
        if (exprType != nullptr) {
            if (std::find(allowedTypes.begin(), allowedTypes.end(), exprType->baseTypeEnum) == allowedTypes.end()) {
                throw TypeError(tree->loc(), "Cannot stream in a " + typeEnumToString(exprType->baseTypeEnum));
            }
        } else {
            throw TypeError(tree->loc(), "Cannot stream out unknown type");
        }

        if(!std::dynamic_pointer_cast<IDNode>(tree->getExpr()) && !std::dynamic_pointer_cast<TupleIndexNode>(tree->getExpr())) {
            throw SyntaxError(tree->loc(), "vector and matrix indexing not implemented yet/ incorrect lvalue");
        }

        return nullptr;
    }

    std::any TypeWalker::visitCall(std::shared_ptr<CallNode> tree) {
        if (tree->procCall) {
            walkChildren(tree);
            tree->evaluatedType = nullptr;
        }
        else {
            //must be an expression then
            tree->evaluatedType = tree->MethodRef->typeSym;
        }
        return nullptr;
    }

    std::any TypeWalker::visitCast(std::shared_ptr<CastNode> tree) {
        walkChildren(tree);
        auto toType = symtab->resolveTypeUser(tree->children[0]);
        auto exprType = tree->children[1]->evaluatedType;
        if (toType->getBaseTypeEnumName() == "tuple" && exprType->getBaseTypeEnumName() == "tuple") {
            if (toType->tupleChildType.size() != exprType->tupleChildType.size()) {
                throw TypeError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
            }
            for (size_t i = 0; i < exprType->tupleChildType.size(); i++) {
                auto leftTypeString = exprType->tupleChildType[i].second->getBaseTypeEnumName();
                auto rightTypeString = toType->tupleChildType[i].second->getBaseTypeEnumName();

                if (leftTypeString != rightTypeString) {
                    auto leftIndex = promotedType->getTypeIndex(leftTypeString);
                    auto rightIndex = promotedType->getTypeIndex(rightTypeString);
                    std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
                    if (resultTypeString.empty()) {
                        throw TypeError(tree->loc(), "Cannot implicitly promote " + leftTypeString + " to " + rightTypeString);
                    }
                }
            }
            tree->evaluatedType = toType; // tuple Type
        }
        else if (toType->getBaseTypeEnumName() == "tuple" || toType->getBaseTypeEnumName() == "tuple" ) {
            throw TypeError(tree->loc(), "only tuple to tuple casting is permitted");
        }
        else {
            auto leftIndex = promotedType->getTypeIndex(exprType->getBaseTypeEnumName());
            auto rightIndex = promotedType->getTypeIndex(toType->getBaseTypeEnumName());
            std::string resultTypeString = promotedType->castTable[leftIndex][rightIndex];
            if (resultTypeString.empty()) {
                throw TypeError(tree->loc(), "Cannot cast " + exprType->getBaseTypeEnumName() + " to " + toType->getBaseTypeEnumName());
            }
            auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));
            tree->evaluatedType = resultType; // base Type
        }
        return nullptr;
    }

    std::any TypeWalker::visitTypedef(std::shared_ptr<TypeDefNode> tree) {
        auto typeNode = tree->children[0];
        tree->evaluatedType = symtab->resolveTypeUser(typeNode);
        return nullptr;
    }

    std::any TypeWalker::visitVector(std::shared_ptr<VectorNode> tree) {
        // innertype(evaluatedType->baseTypeEnum) will be set by the declaration node
        for (auto &exprNode: tree->getElements()) {
            walk(exprNode);  // set the evaluated type of each expr
        }
        tree->evaluatedType = std::make_shared<AdvanceType>(tree->getElement(0)->evaluatedType->getBaseTypeEnumName());  // this string name will be modified in visitDecl
        tree->evaluatedType->vectorOrMatrixEnum = TYPE::VECTOR;
        tree->evaluatedType->baseTypeEnum = TYPE::NONE; // this will be set by visitDecl when we promote all RHS
        tree->evaluatedType->dims.push_back(tree->getSize());  // the size of this vector
        return nullptr;
    }
    std::any TypeWalker::visitMatrix(std::shared_ptr<MatrixNode> tree) {
        // innertype(evaluatedType->baseTypeEnum) will be set by the declaration node
        tree->evaluatedType = std::make_shared<AdvanceType>("matrix");
        tree->evaluatedType->vectorOrMatrixEnum = TYPE::MATRIX;
        return nullptr;
    }

    std::string TypeWalker::typeEnumToString(TYPE t) {
        switch (t) {
            case TYPE::BOOLEAN:
                return "boolean";
            case TYPE::CHAR:
                return "character";
            case TYPE::INTEGER:
                return "integer";
            case TYPE::REAL:
                return "real";
            case TYPE::STRING:
                return "string";
            case TYPE::VECTOR:
                return "vector";
            case TYPE::MATRIX:
                return "matrix";
            case TYPE::TUPLE:
                return "tuple";
            case TYPE::NONE:
                return "none";
        }

    }


}
