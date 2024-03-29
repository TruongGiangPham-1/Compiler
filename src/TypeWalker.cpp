#include "TypeWalker.h"
#include "ASTNode/Type/TupleTypeNode.h"
// #define DEBUG

// until we get more typecheck done

namespace gazprea {

PromotedType::PromotedType(std::shared_ptr<SymbolTable> symtab)
    : symtab(symtab)
    , currentScope(symtab->globalScope)
{
}
PromotedType::~PromotedType() { }

std::string PromotedType::booleanResult[7][7] = {
    /*                      boolean   character    integer  real  tuple identity null:*/
    /*boolean*/ { "boolean", "", "", "", "", "boolean", "boolean" },
    /*character*/ { "", "", "", "", "", "", "" },
    /*integer*/ { "", "", "", "", "", "", "" },
    /*real*/ { "", "", "", "", "", "", "" },
    /*tuple*/ { "", "", "", "", "", "", "" },
    /*identity*/ { "boolean", "", "", "", "", "", "" },
    /*null*/ { "boolean", "", "", "", "", "", "" }
};

std::string PromotedType::castTable[7][7] = {
    /* from\to                     boolean   character    integer  real  */
    /*boolean*/ { "boolean", "character", "integer", "real", "", "", "" },
    /*character*/ { "boolean", "character", "integer", "real", "", "", "" },
    /*integer*/ { "boolean", "character", "integer", "real", "", "", "" },
    /*real*/ { "", "", "integer", "real", "", "", "" },
    /*tuple*/ { "", "", "", "", "", "", "" },
    /*identity*/ { "boolean", "character", "integer", "real", "", "", "" },
    /*null*/ { "boolean", "character", "integer", "real", "", "", "" },
};

std::string PromotedType::arithmeticResult[7][7] = {
    /*                      boolean   character    integer  real  tuple identity null*/
    /*boolean*/ { "", "", "", "", "", "", "" },
    /*character*/ { "", "", "", "", "", "", "" },
    /*integer*/ { "", "", "integer", "real", "", "integer", "integer" },
    /*real*/ { "", "", "real", "real", "", "real", "real" },
    /*tuple*/ { "", "", "", "", "", "", "" },
    /*identity*/ { "", "", "integer", "real", "", "", "" },
    /*null*/ { "", "", "integer", "real", "", "", "" }
};

std::string PromotedType::comparisonResult[7][7] = {
    /*                      boolean   character    integer  real  tuple identity null*/
    /*boolean*/ { "", "", "", "", "", "", "" },
    /*character*/ { "", "", "", "", "", "", "" },
    /*integer*/ { "", "", "boolean", "boolean", "", "boolean", "boolean" },
    /*real*/ { "", "", "boolean", "boolean", "", "boolean", "boolean" },
    /*tuple*/ { "", "", "", "", "", "", "" },
    /*identity*/ { "", "", "boolean", "boolean", "", "", "" },
    /*null*/ { "", "", "boolean", "boolean", "", "", "" }
};

std::string PromotedType::equalityResult[7][7] = {
    /*                      boolean   character    integer  real  tuple identity null*/
    /*boolean*/ { "boolean", "", "", "", "", "boolean", "boolean" },
    /*character*/ { "", "boolean", "", "", "", "boolean", "boolean" },
    /*integer*/ { "", "", "boolean", "boolean", "", "boolean", "boolean" },
    /*real*/ { "", "", "boolean", "boolean", "", "boolean", "boolean" },
    /*tuple*/ { "", "", "", "", "", "", "" },
    /*identity*/ { "boolean", "boolean", "boolean", "boolean", "", "", "" },
    /*null*/ { "boolean", "boolean", "boolean", "boolean", "", "", "" }
};

std::string PromotedType::promotionTable[7][7] = {
    /*                      boolean   character    integer  real  tuple identity null*/
    /*boolean*/ { "boolean", "", "", "", "", "boolean", "boolean" },
    /*character*/ { "", "character", "", "", "", "character", "character" },
    /*integer*/ { "", "", "integer", "real", "", "integer", "integer" },
    /*real*/ { "", "", "", "real", "", "real", "real" },
    /*tuple*/ { "", "", "", "", "tuple", "tuple", "tuple" },
    /*identity*/ { "boolean", "character", "integer", "real", "tuple", "", "" },
    /*null*/ { "boolean", "character", "integer", "real", "tuple", "", "" }
    // TODO: Add identity and null support promotion when Def Ref is done.
};
void PromotedType::populateInnerTypes(std::shared_ptr<Type> type, std::shared_ptr<VectorNode> tree)
{
    // given a vector nodes, just simply add to innerType array in type
    type->vectorInnerTypes.clear();
    for (auto& child : tree->getElements()) {
        type->vectorInnerTypes.push_back(getTypeCopy(child->evaluatedType));
    }
}

void PromotedType::printTypeClass(std::shared_ptr<Type> type)
{
    if (isVector(type)) {
        std::cout << type->getBaseTypeEnumName() << " vector\n";
    } else if (isMatrix(type)) {
        std::cout << type->getBaseTypeEnumName() << " matrix\n";
    } else {
        if (type->baseTypeEnum == TUPLE) {
            std::cout << "printint tuple\n";
            for (auto& c : type->tupleChildType) {
                printTypeClass(c.second);
            }
        }
        if (isScalar(type)) {
            std::cout << type->getBaseTypeEnumName() << " base\n";
        } else if (isEmptyArrayLiteral(type)) {
            std::cout << " empty\n";
        }
    }
}
// just creates Vector Type or Matrix TYpe  given base type
std::shared_ptr<Type> PromotedType::createArrayType(std::string baseTypeStr, TYPE vectORMatrix)
{
    std::shared_ptr<Type> arrayType = nullptr;
    switch (vectORMatrix) {
    case VECTOR: {
        arrayType = getTypeCopy(symtab->globalScope->resolveType(baseTypeStr));
        auto innerType = getTypeCopy(symtab->globalScope->resolveType(baseTypeStr));
        arrayType->vectorInnerTypes.push_back(innerType);
        arrayType->vectorOrMatrixEnum = VECTOR;
        break;
    }
    case MATRIX: {
        arrayType = getTypeCopy(symtab->globalScope->resolveType(baseTypeStr));
        auto innerType = getTypeCopy(symtab->globalScope->resolveType(baseTypeStr));
        innerType->vectorOrMatrixEnum = VECTOR;
        auto innerType2 = getTypeCopy(symtab->globalScope->resolveType(baseTypeStr));
        innerType->vectorInnerTypes.push_back(innerType2);
        arrayType->vectorInnerTypes.push_back(innerType);
        arrayType->vectorOrMatrixEnum = VECTOR;
        break;
    }
    }

    return arrayType;
}
void PromotedType::possiblyPaddMatrix(std::shared_ptr<VectorNode> tree)
{
    // given a node, possibly padd them with null node
    int isMatrix = 0;
    for (auto& child : tree->getElements()) {
        if (child->evaluatedType->vectorOrMatrixEnum == VECTOR) {
            isMatrix = 1;
        }
    }
    if (!isMatrix) { // case: this is not a vector so we dont need to pad
        return;
    }
    int maxSizeRow = INT32_MIN;
    for (auto& child : tree->getElements()) {
        maxSizeRow = std::max(maxSizeRow, child->evaluatedType->dims[0]);
    }
    for (auto& child : tree->getElements()) {
        int howMuch = maxSizeRow - child->evaluatedType->dims[0];
        if (std::dynamic_pointer_cast<VectorNode>(child)) {
            addNullNodesToVector(maxSizeRow, std::dynamic_pointer_cast<VectorNode>(child)); //
        } else {
            // its an ID node
        }
    }
    return;
}
void PromotedType::addNullNodesToVector(int howMuch, std::shared_ptr<VectorNode> tree)
{
    // used when there are matrix rows that are shorter size the
    while (howMuch--) {
        std::shared_ptr<NullNode> nullNode = std::make_shared<NullNode>(tree->loc());
        tree->addChild(std::dynamic_pointer_cast<ASTNode>(nullNode));
    }

    return;
}
void PromotedType::dotProductErrorCheck(std::shared_ptr<Type> l, std::shared_ptr<Type> r, int line)
{
    if (l->vectorOrMatrixEnum == NONE || r->vectorOrMatrixEnum == NONE) {
        throw TypeError(line, "scaler cannot be dot producted");
    }
    if ((isMatrix(l) && isMatrix(r))
        || (isVector(r) && isVector(l))) {

    } else {
        throw TypeError(line, "dot product can only be done between matrix or between vectors");
    }
}

std::shared_ptr<Type> PromotedType::getTypeCopy(std::shared_ptr<Type> type)
{
    // returns a copy of the type
    auto newtype = std::make_shared<AdvanceType>(type->getBaseTypeEnumName());
    newtype->baseTypeEnum = type->baseTypeEnum;
    if (type->vectorOrMatrixEnum != NONE) {
        newtype->vectorOrMatrixEnum = type->vectorOrMatrixEnum;
    }
    newtype->isString = type->isString;
    if (!type->dims.empty()) {
        newtype->dims = type->dims;
    }
    for (auto& innerType : type->vectorInnerTypes) {
        newtype->vectorInnerTypes.push_back(getTypeCopy(innerType));
    }
    for (auto& tupleType : type->tupleChildType) {
        newtype->tupleChildType.push_back(std::make_pair(tupleType.first, getTypeCopy(tupleType.second)));
    }
    return newtype;
}
std::shared_ptr<Type> PromotedType::getType(std::string table[7][7], std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right, std::shared_ptr<ASTNode> t)
{
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
        } else {
            throw TypeError(t->loc(), "Cannot perform operation between " + left->evaluatedType->getBaseTypeEnumName() + " and " + right->evaluatedType->getBaseTypeEnumName());
        }
    }

    auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));
    // HERE vector should have identical types because possiblyPromoteBinop have make sure both vectors are same types.
    if (left->evaluatedType->vectorOrMatrixEnum == VECTOR && right->evaluatedType->vectorOrMatrixEnum == VECTOR && std::dynamic_pointer_cast<BinaryCmpNode>(t) == nullptr) { // skip this is if we are doing cmpNode since we want binop tobe nonVec

        assert(right->evaluatedType->vectorOrMatrixEnum == VECTOR);
        auto typeCopyl = getTypeCopy(left->evaluatedType);
        auto typeCopyr = getTypeCopy(right->evaluatedType);
        if (typeCopyl->vectorInnerTypes.size() > typeCopyr->vectorInnerTypes.size()) {
            return typeCopyl;
        } else
            return typeCopyr; // return copy of the type
    }
#ifdef DEBUG
    std::cout << "type promotions between " << left->evaluatedType->getBaseTypeEnumName() << ", " << right->evaluatedType->getBaseTypeEnumName() << "\n";
    std::cout << "result: " << resultType->getBaseTypeEnumName() << "\n";
#endif
    assert(resultType);
    return resultType;
}

std::string PromotedType::getPromotedTypeString(std::string table[7][7], std::shared_ptr<Type> left, std::shared_ptr<Type> right)
{
    auto leftIndex = this->getTypeIndex(left->getBaseTypeEnumName());
    auto rightIndex = this->getTypeIndex(right->getBaseTypeEnumName());
    std::string resultTypeString = table[leftIndex][rightIndex];
    return resultTypeString;
}

int PromotedType::getTypeIndex(const std::string type)
{
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
void PromotedType::promoteLiteralToArray(std::shared_ptr<Type> promoteTo, std::shared_ptr<ASTNode> literalNode, std::string table[7][7])
{
    if (literalNode->evaluatedType->baseTypeEnum == TUPLE)
        throw TypeError(literalNode->loc(), "cannot promote tuple to array");
    if (literalNode->evaluatedType->vectorOrMatrixEnum == VECTOR) {
        return;
    }
    if (promoteTo->vectorOrMatrixEnum == VECTOR) {
        auto s = getPromotedTypeString(table, literalNode->evaluatedType, promoteTo);
        if (s.empty()) {
            throw TypeError(literalNode->loc(), "cannot implicitly promote: typewalk line 215");
        }
        auto typeCopy = getTypeCopy(promoteTo);
        literalNode->evaluatedType = typeCopy; // to make sure it gets its own copy
        literalNode->evaluatedType->dims.clear();
        literalNode->evaluatedType->dims.push_back(1); // size 1 vector
    } else {
    }
}
/*
 * given left and right binop node
 * try to promote one side with another, vice vcerssa
 * eg: left =int vector , right =  real vector, i will promote left to a real vector
 * TODO: i only implement this for vector binops for far. so future ill try to generalize this to all type?
 */
void PromotedType::possiblyPromoteBinop(std::shared_ptr<ASTNode> left, std::shared_ptr<ASTNode> right, std::string table[7][7])
{

    if ((isMatrix(left->evaluatedType) && isVector(right->evaluatedType)) || (isMatrix(right->evaluatedType) && isVector(left->evaluatedType))) {
        // cannot have matrix to vector operation
        throw TypeError(left->loc(), "matrix and vector operation invalid");
    }
    if (isEmptyArrayLiteral(left->evaluatedType) || isEmptyArrayLiteral(right->evaluatedType)) {
        throw TypeError(left->loc(), "cannot use empty array in binop");
    }
    auto LtoRpromotion = getPromotedTypeString(table, left->evaluatedType, right->evaluatedType);
    auto RtoLpromotion = getPromotedTypeString(table, right->evaluatedType, left->evaluatedType);
    if (LtoRpromotion.empty() && RtoLpromotion.empty())
        throw TypeError(left->loc(), "invalid vectors type binop");
    std::shared_ptr<Type> dominantType;
    std::shared_ptr<ASTNode> promoteNode;
    if (left->evaluatedType->baseTypeEnum != TYPE::IDENTITY && left->evaluatedType->baseTypeEnum != TYPE::NULL_ && right->evaluatedType->baseTypeEnum != TYPE::IDENTITY && right->evaluatedType->baseTypeEnum != TYPE::NULL_) {
        dominantType = !LtoRpromotion.empty() ? right->evaluatedType : left->evaluatedType; // l to r promotion, so r has dominant type
        promoteNode = !LtoRpromotion.empty() ? left : right; // l to r promotion valid so promote left node, vice versa
    } else {
        if (left->evaluatedType->baseTypeEnum == TYPE::IDENTITY || left->evaluatedType->baseTypeEnum == TYPE::NULL_) {
            dominantType = right->evaluatedType;
            promoteNode = left;
        } else if (right->evaluatedType->baseTypeEnum == TYPE::IDENTITY || right->evaluatedType->baseTypeEnum == TYPE::NULL_) {
            dominantType = left->evaluatedType;
            promoteNode = right;
        }
    }
    // vector handling
    if (isVector(left->evaluatedType) && isVector(right->evaluatedType)) {
        // if (left->evaluatedType->dims[0] != right->evaluatedType->dims[0]) throw SizeError(left->loc(), "incompatible size binop");

        promoteVectorElements(dominantType, promoteNode, table);
        updateVectorNodeEvaluatedType(dominantType, promoteNode);
    } else if (isMatrix(left->evaluatedType) && isMatrix(right->evaluatedType)) {
        promoteVectorElements(dominantType, promoteNode, table);
        updateVectorNodeEvaluatedType(dominantType, promoteNode);
    } else if (isMatrix(left->evaluatedType) && isVector(right->evaluatedType)) {
        possiblyPromoteToVectorOrMatrix(left->evaluatedType, right->evaluatedType, left->loc()); // update right->Evaltype to matrix
        promoteVectorElements(dominantType, promoteNode, table);
        updateVectorNodeEvaluatedType(dominantType, promoteNode);
    } else if (isMatrix(right->evaluatedType) && isVector(left->evaluatedType)) {
        possiblyPromoteToVectorOrMatrix(right->evaluatedType, left->evaluatedType, left->loc()); // update left->evaltype ot matrix
        auto r = left->evaluatedType;
        promoteVectorElements(dominantType, promoteNode, table);
        updateVectorNodeEvaluatedType(dominantType, promoteNode);
    } else if (isMatrix(left->evaluatedType) && right->evaluatedType->vectorOrMatrixEnum == NONE) {
        possiblyPromoteToVectorOrMatrix(left->evaluatedType, right->evaluatedType, left->loc()); // update left->evaltype ot matrix
        promoteVectorElements(dominantType, promoteNode, table);
        updateVectorNodeEvaluatedType(dominantType, promoteNode);
    } else if (isMatrix(right->evaluatedType) && left->evaluatedType->vectorOrMatrixEnum == NONE) {
        possiblyPromoteToVectorOrMatrix(right->evaluatedType, left->evaluatedType, left->loc()); // update left->evaltype ot matrix
        promoteVectorElements(dominantType, promoteNode, table);
        updateVectorNodeEvaluatedType(dominantType, promoteNode);
        auto l = left->evaluatedType;

    } else if (isVector(left->evaluatedType) && right->evaluatedType->vectorOrMatrixEnum == NONE) {
        // possiblyPromoteToVectorOrMatrix(left->evaluatedType, right->evaluatedType, left->loc());  // update left->evaltype ot matrix
        right->evaluatedType = createArrayType(right->evaluatedType->getBaseTypeEnumName(), VECTOR);
        promoteVectorElements(dominantType, promoteNode, table);
        promoteLiteralToArray(left->evaluatedType, right, table);
    } else if (isVector(right->evaluatedType) && left->evaluatedType->vectorOrMatrixEnum == NONE) {
        // none vector
        // possiblyPromoteToVectorOrMatrix(right->evaluatedType, left->evaluatedType, left->loc());  // update left->evaltype ot matrix
        left->evaluatedType = createArrayType(left->evaluatedType->getBaseTypeEnumName(), VECTOR);
        promoteVectorElements(dominantType, promoteNode, table);
        promoteLiteralToArray(right->evaluatedType, left, table);
    } else {
        // case everything else. like base type
        promoteNode->evaluatedType = dominantType;
    }
}
void PromotedType::promoteIdentityAndNull(std::shared_ptr<Type> promoteTo, std::shared_ptr<ASTNode> identityNode)
{
    if (promoteTo->vectorOrMatrixEnum == TYPE::VECTOR) {
        identityNode->evaluatedType = getTypeCopy(promoteTo);
        if (std::dynamic_pointer_cast<VectorNode>(identityNode)) {
            for (auto childExpr : std::dynamic_pointer_cast<VectorNode>(identityNode)->getElements()) {
                childExpr->evaluatedType = getTypeCopy(identityNode->evaluatedType->vectorInnerTypes[0]);
            }
            return;
        }

        else if ((isVector(identityNode->evaluatedType) || isMatrix(identityNode->evaluatedType)) && (identityNode->evaluatedType->baseTypeEnum != TYPE::IDENTITY && identityNode->evaluatedType->baseTypeEnum != TYPE::NULL_)) {
            for (int i = 0; i < identityNode->evaluatedType->vectorInnerTypes.size(); i++) {
                identityNode->evaluatedType->vectorInnerTypes[i] = promoteTo->vectorInnerTypes[0];
            }
            return;
        }
    } else if (promoteTo->vectorOrMatrixEnum == TYPE::MATRIX) {

    } else {
        identityNode->evaluatedType = getTypeCopy(promoteTo);
    }
    return;
}
int PromotedType::isMatrix(std::shared_ptr<Type> type)
{
    int t = !type->vectorInnerTypes.empty() && type->vectorOrMatrixEnum == VECTOR && type->vectorInnerTypes[0]->vectorOrMatrixEnum == VECTOR;
    return t;
}
int PromotedType::isVector(std::shared_ptr<Type> type)
{
    int t = !type->vectorInnerTypes.empty() && type->vectorOrMatrixEnum == VECTOR && type->vectorInnerTypes[0]->vectorOrMatrixEnum == NONE;
    return t;
}
int PromotedType::isScalar(std::shared_ptr<Type> type)
{
    int t = type->vectorOrMatrixEnum == NONE && type->baseTypeEnum != TUPLE && type->baseTypeEnum != STRING;
    return t;
}
int PromotedType::isEmptyArrayLiteral(std::shared_ptr<Type> type)
{
    int t = type->vectorOrMatrixEnum == VECTOR && type->baseTypeEnum == NONE;
    return t;
}

void PromotedType::possiblyPromoteToVectorOrMatrix(std::shared_ptr<Type> promoteTo,
    std::shared_ptr<Type> promotedType, int line)
{
    // promoteTo should be a matrix type
    // if promteTo is matrix and promotedType is vector, promote vector->matrix
    // if promoteTo is matrix and promotedType is scalar, prmote scalar->matrix
    int vecToMatrixPromo = isMatrix(promoteTo) && isVector(promotedType);
    int scalarToMatrixPromo = isMatrix(promoteTo)
        && promotedType->vectorOrMatrixEnum == NONE;
    int scalarToVectorPromo = isVector(promoteTo) && promotedType->vectorOrMatrixEnum == NONE;
    if (vecToMatrixPromo) {
        /*
         *   [int, int] -> [intvect, intvect]
         *
         */
        auto itself = getTypeCopy(promotedType); // copy itself
        for (int i = 0; i < promotedType->vectorInnerTypes.size(); i++) {
            promotedType->vectorInnerTypes[i] = getTypeCopy(itself);
        }
        promotedType->dims.push_back(promotedType->vectorInnerTypes.size());
        // should create a square matrix
        assert(promotedType->dims[0] == promotedType->dims[1] == 1);
        return;
    }
    if (scalarToMatrixPromo) {
        /*
         * int -> [int] -> [[int]]
         *
         */
        auto itself = getTypeCopy(promotedType); // copy itself
        itself->vectorInnerTypes.push_back(getTypeCopy(itself));
        itself->vectorOrMatrixEnum = VECTOR;
        promotedType->vectorInnerTypes.push_back(itself);
        promotedType->vectorOrMatrixEnum = VECTOR;
        promotedType->dims.push_back(1); // should be (1, 1) matrix
        promotedType->dims.push_back(1); // should be (1, 1) matrix
        assert(promotedType->dims[0] == 1 && promotedType->dims[1] == 1);
        return;
    }
    if (scalarToVectorPromo) {
        if (promotedType->baseTypeEnum == TUPLE)
            throw TypeError(line, "cannot promote tuple to array");
        // just create size 1 vector
        auto itself = getTypeCopy(promotedType); // copy itself
        promotedType->vectorInnerTypes.push_back(itself);
        promotedType->vectorOrMatrixEnum = VECTOR;
        promotedType->dims.clear();
        promotedType->dims.push_back(1);
    }

    // if (promoteTo->vectorInnerTypes[0]->size() != promotedType->vectorInnerTypes.size()) {
    //     // TODO: .size() might be tricky if promted type used to be identity/null because it can
    //     throw TypeError(1, "cannot promote to matrix");
    // }
    /*
     * Note to myself.
     * Matrix[3, 2] a = [1, 2], i cannot promote rhs to [[1, 2], [1, 2], [1, 2]] since i dont know the size, im just making [[1, 2], [1, 2]] for now
     *
     */
    return;
}
std::shared_ptr<Type> PromotedType::promoteVectorTypeObj(std::shared_ptr<Type> promoteTo, std::shared_ptr<Type> promotedType, int line, std::string table[7][7])
{
    // symmetric to promoteVectorElements, but do it on Type obj instead of ASTNode
    auto promotedTypeCop = getTypeCopy(promotedType);
    if (promotedTypeCop->vectorInnerTypes.empty()) {
        // basecase
        assert(promotedTypeCop->vectorOrMatrixEnum == NONE);
        auto str = getPromotedTypeString(table, promotedTypeCop, promoteTo);
        if (str.empty())
            throw TypeError(line, "cannot promote vector element");
        promotedTypeCop->baseTypeEnum = promoteTo->baseTypeEnum;
        return promotedTypeCop;
    }
    for (int i = 0; i < promotedTypeCop->vectorInnerTypes.size(); i++) {
        promotedTypeCop->vectorInnerTypes[i] = promoteVectorTypeObj(promoteTo, promotedTypeCop->vectorInnerTypes[i], line, table);
    }
    promotedTypeCop->baseTypeEnum = promoteTo->baseTypeEnum;
    promotedTypeCop->isString = promoteTo->isString;
    return promotedTypeCop;
}

void PromotedType::promoteTupleElements(std::shared_ptr<Type> promoteTo, std::shared_ptr<ASTNode> exprNode, std::string table[7][7])
{
    if (std::dynamic_pointer_cast<TupleNode>(exprNode)) {
        auto tupleCast = std::dynamic_pointer_cast<TupleNode>(exprNode);
        int i = 0;
        for (auto& kv : tupleCast->getVal()) { // each tupleNode's child
            auto lType = promoteTo->tupleChildType[i].second;
            auto rType = kv->evaluatedType;
            if (lType->vectorOrMatrixEnum == VECTOR) {
                if (isMatrix(lType) && isVector(rType)) {
                    throw TypeError(exprNode->loc(), "cannot promote vectorNode to matrix");
                }
                if (isScalar(rType)) {
                    promoteLiteralToArray(lType, kv, table);
                } else if (isEmptyArrayLiteral(rType)) {
                    // empty array literal. simply promote to ltype
                    // promotedType->emptyArrayErrorCheck(tree->getTypeNode());
                    kv->evaluatedType = getTypeCopy(lType);
                } else {
                    promoteVectorElements(promoteTo->tupleChildType[i].second, kv, table);
                    updateVectorNodeEvaluatedType(promoteTo->tupleChildType[i].second, kv);
                }
                exprNode->evaluatedType->tupleChildType[i].second = getTypeCopy(kv->evaluatedType);

            } else if (isScalar(kv->evaluatedType)) {
                auto promo = getPromotedTypeString(table, kv->evaluatedType, promoteTo->tupleChildType[i].second);
                if (promo.empty())
                    throw TypeError(exprNode->loc(), "cannot promote tuple element");
                kv->evaluatedType = getTypeCopy(currentScope->resolveType(promo));
                exprNode->evaluatedType->tupleChildType[i].second = getTypeCopy(kv->evaluatedType);
            }
            i += 1;
        }
    } else {
        // other nodes
        int i = 0;
        for (auto& type : exprNode->evaluatedType->tupleChildType) {
            if (type.second->vectorOrMatrixEnum == VECTOR) {
                auto promotedVtype = promoteVectorTypeObj(promoteTo->tupleChildType[i].second, type.second, exprNode->loc(), table);
                exprNode->evaluatedType->tupleChildType[i].second = getTypeCopy(promotedVtype);
            } else if (isScalar(type.second)) {
                auto promo = getPromotedTypeString(table, type.second, promoteTo->tupleChildType[i].second);
                if (promo.empty())
                    throw TypeError(exprNode->loc(), "cannot promote tuple element");
                type.second = getTypeCopy(currentScope->resolveType(promo));
                exprNode->evaluatedType->tupleChildType[i].second = getTypeCopy(type.second);
            }
            i += 1;
        }
    }
}

// promote all evaluatedType of a vector tree node
void PromotedType::promoteVectorElements(std::shared_ptr<Type> promoteTo, std::shared_ptr<ASTNode> exprNode, std::string table[7][7])
{
    if (exprNode->evaluatedType->baseTypeEnum == TYPE::IDENTITY || exprNode->evaluatedType->baseTypeEnum == TYPE::NULL_) {
        promoteIdentityAndNull(promoteTo, exprNode);
        return;
    }

    // assert(exprNode->evaluatedType->vectorOrMatrixEnum == TYPE::VECTOR);  // remove this when im implementing matrix
    if (exprNode->evaluatedType->vectorOrMatrixEnum == NONE) {
        // this is a vector element node. one of base case of recursion
        auto rhsIndex = getTypeIndex(exprNode->evaluatedType->getBaseTypeEnumName());
        auto lhsIndex = getTypeIndex(promoteTo->getBaseTypeEnumName());
        auto promoteTypeString = table[rhsIndex][lhsIndex];
        if (promoteTypeString.empty())
            throw TypeError(exprNode->loc(), "cannot promote vector element");
        auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(promoteTypeString));
#ifdef DEBUG
        // std::cout << "promoted vector element " << child->evaluatedType->getBaseTypeEnumName() << "to " << promoteTo->getBaseTypeEnumName() << "\n";
#endif
        exprNode->evaluatedType = resultType; // set each vector element node to its promoted type
        return;
    }
    // recursively promote inner vector nodes
    auto vectNodeCast = std::dynamic_pointer_cast<VectorNode>(exprNode);
    if (vectNodeCast == nullptr) {
        // this is not a vector literal. but could be other vector nodes
        if (std::dynamic_pointer_cast<IDNode>(exprNode) || std::dynamic_pointer_cast<StrideNode>(exprNode) || std::dynamic_pointer_cast<ConcatNode>(exprNode)
            || std::dynamic_pointer_cast<RangeVecNode>(exprNode) || std::dynamic_pointer_cast<GeneratorNode>(exprNode) || std::dynamic_pointer_cast<ExprNode>(exprNode) || std::dynamic_pointer_cast<StringNode>(exprNode)) {
            //
            auto sizeVec = exprNode->evaluatedType->dims; // vector of size
            auto promoteTypeString = getPromotedTypeString(table, exprNode->evaluatedType, promoteTo);
            if (promoteTypeString.empty())
                throw TypeError(exprNode->loc(), "cannot promote vector element");
            // promote all the inner types
            auto promotedType = promoteVectorTypeObj(promoteTo, exprNode->evaluatedType, exprNode->loc(), table); // promote the old evaluted type
            // exprNode->evaluatedType = getTypeCopy(promoteTo);  // create copy of the pormoted type
            exprNode->evaluatedType = promotedType; // promteType promtes all innerchildToo
            exprNode->evaluatedType->vectorOrMatrixEnum = VECTOR; // assign correct attribute
            exprNode->evaluatedType->dims.clear();
            exprNode->evaluatedType->dims = sizeVec; // reassign size

        } else {
            // case: exprNode that is not an IDnode AND it is not a vectorNode
            throw SyntaxError(exprNode->loc(), "invalid matrix/vector element");
        }
        return;
    }
    // promote each vector elements
    for (auto& child : vectNodeCast->getElements()) {
        if (isEmptyArrayLiteral(child->evaluatedType)) {
            auto vectorType = createArrayType(promoteTo->getBaseTypeEnumName(), VECTOR);
            child->evaluatedType = vectorType;
            continue;
        }

        if (child->evaluatedType->vectorOrMatrixEnum == VECTOR) {
            // this means that vector is recursive
            if (child->evaluatedType->baseTypeEnum == VECTOR)
                throw SyntaxError(exprNode->loc(), "invalid matrix");

            auto sizeVec = child->evaluatedType->dims;
            auto promotedType = promoteVectorTypeObj(promoteTo, child->evaluatedType, exprNode->loc(), table); // promote the old evaluted type
            promoteVectorElements(promoteTo, child, table); // promote the children first
            // child->evaluatedType = getTypeCopy(promoteTo);
            child->evaluatedType = promotedType;
            child->evaluatedType->dims.clear();
            child->evaluatedType->dims = sizeVec; // just reasign old size

        } else { // child is not a vector
            auto rhsIndex = getTypeIndex(child->evaluatedType->getBaseTypeEnumName());
            auto lhsIndex = getTypeIndex(promoteTo->getBaseTypeEnumName());
            auto promoteTypeString = table[rhsIndex][lhsIndex];
            if (promoteTypeString.empty())
                throw TypeError(exprNode->loc(), "cannot promote vector element");
            auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(promoteTypeString));
            // child->evaluatedType = resultType;  // set each vector element node to its promoted type
            child->evaluatedType = promoteVectorTypeObj(promoteTo, child->evaluatedType, exprNode->loc(), table);
        }
    }
    // update the root node's evaluated type
    vectNodeCast->evaluatedType->vectorInnerTypes.clear();
    for (int i = 0; i < vectNodeCast->getSize(); i++) {
        // vectNodeCast->evaluatedType->vectorInnerTypes[i] = getTypeCopy(vectNodeCast->getElement(i)->evaluatedType);
        vectNodeCast->evaluatedType->vectorInnerTypes.push_back(getTypeCopy(vectNodeCast->getElement(i)->evaluatedType));
    }
}
/*
 *  update a node's evaluated type by copying over attributes that matters. do not modify the type->dims(which was set in visitVector)
 */
void PromotedType::updateVectorNodeEvaluatedType(std::shared_ptr<Type> assignType, std::shared_ptr<ASTNode> exprNode)
{
    exprNode->evaluatedType->baseTypeEnum = assignType->baseTypeEnum; // set the LHS vector literal type. int?real?
    exprNode->evaluatedType->vectorOrMatrixEnum = assignType->vectorOrMatrixEnum;
    exprNode->evaluatedType->isString = assignType->isString;
    exprNode->evaluatedType->setName(assignType->getBaseTypeEnumName()); // set the string evaluated type
}

std::shared_ptr<Type> PromotedType::getDominantTypeFromVector(std::shared_ptr<VectorNode> tree)
{
    std::shared_ptr<Type> bestType = nullptr;
    for (int i = 0; i < tree->getSize(); i++) {
        auto promoteTo = tree->getElement(i)->evaluatedType;
        if (isEmptyArrayLiteral(promoteTo))
            continue; // skip if its empty litereal
        int isDominant = 1; // gonna chekc if all the other element can promote to this type
        for (int j = 0; j < tree->getSize(); j++) {
            if (i == j)
                continue;
            auto promoteFrom = tree->getElement(j)->evaluatedType;
            if (isEmptyArrayLiteral(promoteFrom))
                continue; // skip if its empty
            if (promoteFrom->baseTypeEnum == TYPE::NULL_ or promoteFrom->baseTypeEnum == TYPE::IDENTITY)
                continue;
            if (getPromotedTypeString(promotionTable, promoteFrom, promoteTo).empty()) {
                //
                isDominant = 0;
            }
        }
        if (isDominant) {
            // this is the dominant type
            bestType = promoteTo;
        }
    }
    if (bestType == nullptr) {
        throw TypeError(tree->loc(), "invalid vector literal type, failed promotion");
    }
    if (bestType->vectorOrMatrixEnum == NONE) {
        // scalar
        auto vectorType = getTypeCopy(bestType);
    }
    auto copy = getTypeCopy(bestType); // if vectorNode is vector, return besttype as matrix
    copy->vectorInnerTypes.push_back(bestType);
    return copy;
}

void PromotedType::assertVector(std::shared_ptr<ASTNode> tree)
{
    if (tree->evaluatedType->vectorOrMatrixEnum == NONE) {
        throw TypeError(tree->loc(), "must be vector");
    }
}

TypeWalker::TypeWalker(std::shared_ptr<SymbolTable> symtab, std::shared_ptr<PromotedType> promotedType)
    : symtab(symtab)
    , currentScope(symtab->globalScope)
    , promotedType(promotedType)
{
}
TypeWalker::~TypeWalker() { }

std::any TypeWalker::visitInt(std::shared_ptr<IntNode> tree)
{
    tree->evaluatedType = promotedType->getTypeCopy(std::dynamic_pointer_cast<Type>(currentScope->resolveType("integer")));
    return nullptr;
}

std::any TypeWalker::visitReal(std::shared_ptr<RealNode> tree)
{
    tree->evaluatedType = promotedType->getTypeCopy(std::dynamic_pointer_cast<Type>(currentScope->resolveType("real")));
    return nullptr;
}

std::any TypeWalker::visitChar(std::shared_ptr<CharNode> tree)
{
    tree->evaluatedType = promotedType->getTypeCopy(std::dynamic_pointer_cast<Type>(currentScope->resolveType("character")));
    return nullptr;
}

std::any TypeWalker::visitBool(std::shared_ptr<BoolNode> tree)
{
    tree->evaluatedType = promotedType->getTypeCopy(std::dynamic_pointer_cast<Type>(currentScope->resolveType("boolean")));
    return nullptr;
}

std::any TypeWalker::visitIdentity(std::shared_ptr<IdentityNode> tree)
{
    tree->evaluatedType = promotedType->getTypeCopy(std::dynamic_pointer_cast<Type>(currentScope->resolveType("identity")));
    return nullptr;
}
std::any TypeWalker::visitNull(std::shared_ptr<NullNode> tree)
{
    tree->evaluatedType = promotedType->getTypeCopy(std::dynamic_pointer_cast<Type>(currentScope->resolveType("null")));
    return nullptr;
}

std::any TypeWalker::visitString(std::shared_ptr<VectorNode> tree)
{
    // and vectorInnerType of character
    for (auto& exprNode : tree->getElements()) {
        walk(exprNode); // set the evaluated type of each expr
    }
    tree->evaluatedType = std::make_shared<AdvanceType>(""); // just initialize it
    tree->evaluatedType->vectorOrMatrixEnum = TYPE::VECTOR;
    if (tree->getElements().empty()) {
        // we will represent empty vector literal if it is vector but doesnt have a type
        tree->evaluatedType->baseTypeEnum = TYPE::CHAR;
        tree->isEmpty = true;
        return nullptr;
    }
    tree->evaluatedType->baseTypeEnum = TYPE::CHAR;
    tree->evaluatedType->vectorInnerTypes.clear();
    // add the inner types to type class
    tree->evaluatedType->isString = true;
    // Populating Inner Types of String "vector" with char types
    tree->evaluatedType->vectorInnerTypes.push_back(promotedType->getTypeCopy(currentScope->resolveType("character")));
    tree->evaluatedType->dims.push_back(tree->getSize()); // the row size of this vector
    return nullptr;
}

std::any TypeWalker::visitTuple(std::shared_ptr<TupleNode> tree)
{
    for (auto expr : tree->val) {
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

std::any TypeWalker::visitArith(std::shared_ptr<BinaryArithNode> tree)
{
    walkChildren(tree);
    auto lhsType = tree->getLHS()->evaluatedType;
    auto rhsType = tree->getRHS()->evaluatedType;
    auto op = tree->op;
    switch (op) {
    case BINOP::MULT:
    case BINOP::DIV:
    case BINOP::ADD:
    case BINOP::EXP:
    case BINOP::SUB:
    case BINOP::REM:
        promotedType->possiblyPromoteBinop(tree->getLHS(), tree->getRHS(), promotedType->promotionTable); //  right now, only handles vectors. make sure rhs and lhs vectors are same type.promote if neccesary
        tree->evaluatedType = promotedType->getType(promotedType->arithmeticResult, tree->getLHS(), tree->getRHS(), tree);
        if (lhsType->getBaseTypeEnumName() == "identity")
            tree->getLHS()->evaluatedType = tree->evaluatedType; // promote LHS
        if (rhsType->getBaseTypeEnumName() == "identity")
            tree->getRHS()->evaluatedType = tree->evaluatedType; // promote RHS
        break;
    case BINOP::XOR:
    case BINOP::OR:
    case BINOP::AND:
        promotedType->possiblyPromoteBinop(tree->getLHS(), tree->getRHS(), promotedType->promotionTable); //  right now, only handles vectors. make sure rhs and lhs vectors are same type.promote if neccesary
        tree->evaluatedType = promotedType->getType(promotedType->booleanResult, tree->getLHS(), tree->getRHS(), tree);
        if (lhsType->getBaseTypeEnumName() == "identity")
            tree->getLHS()->evaluatedType = tree->evaluatedType; // promote LHS
        if (rhsType->getBaseTypeEnumName() == "identity")
            tree->getRHS()->evaluatedType = tree->evaluatedType; // promote RHS
        break;
    case BINOP::DOT_PROD:
        promotedType->dotProductErrorCheck(lhsType, rhsType, tree->loc());
        promotedType->possiblyPromoteBinop(tree->getLHS(), tree->getRHS(), promotedType->promotionTable); //  right now, only handles vectors. make sure rhs and lhs vectors are same type.promote if neccesary

        tree->evaluatedType = promotedType->getType(promotedType->arithmeticResult, tree->getLHS(), tree->getRHS(), tree);
        if (promotedType->isVector(lhsType) && promotedType->isVector(rhsType)) {
            // vector to vector inner product eval type should be scalar
            tree->evaluatedType = promotedType->getTypeCopy(symtab->globalScope->resolveType(tree->getLHS()->evaluatedType->getBaseTypeEnumName()));
        }
        break;
    }
#ifdef DEBUG
    promotedType->printTypeClass(tree->evaluatedType);
#endif
    return nullptr;
}

std::any TypeWalker::visitConcat(std::shared_ptr<ConcatNode> tree)
{
    // separate switch cuz concat can be difrent size
    // need to handle literal promote
    walkChildren(tree);

    // CASE: both lhs and rhs is none vector
    auto l = tree->getLHS();
    auto r = tree->getRHS();

    if (promotedType->isMatrix(l->evaluatedType) || promotedType->isMatrix(r->evaluatedType)) {
        throw TypeError(tree->loc(), "cannot concat Matrices");
    }

    // empty vector
    if (promotedType->isEmptyArrayLiteral(l->evaluatedType) && promotedType->isVector(r->evaluatedType)) {
        tree->evaluatedType = promotedType->getTypeCopy(r->evaluatedType);
        return nullptr;
    } else if (promotedType->isEmptyArrayLiteral(r->evaluatedType) && promotedType->isVector(l->evaluatedType)) {
        tree->evaluatedType = promotedType->getTypeCopy(l->evaluatedType);
        return nullptr;
    } else if (promotedType->isEmptyArrayLiteral(l->evaluatedType) && promotedType->isScalar(r->evaluatedType)) {
        auto vectorType = promotedType->createArrayType(r->evaluatedType->getBaseTypeEnumName(), VECTOR);
        promotedType->promoteLiteralToArray(vectorType, tree->getRHS(), promotedType->promotionTable);
        tree->evaluatedType = promotedType->getTypeCopy(tree->getRHS()->evaluatedType);
        return nullptr;
    } else if (promotedType->isEmptyArrayLiteral(r->evaluatedType) && promotedType->isScalar(l->evaluatedType)) {
        auto vectorType = promotedType->createArrayType(l->evaluatedType->getBaseTypeEnumName(), VECTOR);
        promotedType->promoteLiteralToArray(vectorType, tree->getLHS(), promotedType->promotionTable);
        tree->evaluatedType = promotedType->getTypeCopy(tree->getLHS()->evaluatedType);
        return nullptr;
    } else if (promotedType->isEmptyArrayLiteral(l->evaluatedType) && promotedType->isEmptyArrayLiteral(r->evaluatedType)) {
        // both empty, then just return empty vector
        tree->evaluatedType = promotedType->getTypeCopy(tree->getLHS()->evaluatedType);
        return nullptr;
    }
    //--------------------------------
    promotedType->possiblyPromoteBinop(tree->getLHS(), tree->getRHS(), promotedType->promotionTable); //   make sure rhs and lhs are same type.promote if neccesary
    assert(tree->getLHS()->evaluatedType->baseTypeEnum == tree->getRHS()->evaluatedType->baseTypeEnum);
    tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getLHS(), tree->getRHS(), tree);

    if (promotedType->isScalar(tree->getLHS()->evaluatedType) && promotedType->isScalar(tree->getRHS()->evaluatedType)) {
        // concat between 2 non vector. we have to promote them all to vector
        auto vectorType = std::make_shared<AdvanceType>(tree->getLHS()->evaluatedType->getBaseTypeEnumName());
        vectorType->vectorOrMatrixEnum = VECTOR;
        promotedType->promoteLiteralToArray(vectorType, tree->getLHS(), promotedType->promotionTable); // promote both of em to vector
        promotedType->promoteLiteralToArray(vectorType, tree->getRHS(), promotedType->promotionTable);
        auto typeCopy = promotedType->getTypeCopy(tree->getLHS()->evaluatedType); // create a copy
        tree->evaluatedType = typeCopy; // re assign evaluatoin type
    }
    // size of concat is size of the sum of both side
    if (!tree->getLHS()->evaluatedType->dims.empty() && !tree->getRHS()->evaluatedType->dims.empty()) {
        tree->evaluatedType->dims.clear();
        tree->evaluatedType->dims.push_back(tree->getLHS()->evaluatedType->dims[0] + tree->getRHS()->evaluatedType->dims[0]); // update dim
    }
    return nullptr;
}

std::any TypeWalker::visitStride(std::shared_ptr<StrideNode> tree)
{
    walkChildren(tree);
    auto l = tree->getLHS();
    auto r = tree->getRHS();

    if (r->evaluatedType->vectorOrMatrixEnum != NONE || r->evaluatedType->baseTypeEnum == TUPLE) {
        throw TypeError(tree->loc(), "stride amount must be a scalar");
    } else if (!promotedType->isVector(l->evaluatedType)) {
        throw TypeError(tree->loc(), "stride lhs must be a vector");
    } else if (r->evaluatedType->baseTypeEnum != INTEGER) {
        throw TypeError(tree->loc(), "stride rhs must be an integer");
    }

    tree->evaluatedType = promotedType->getTypeCopy(l->evaluatedType);
    return nullptr;
}

std::any TypeWalker::visitCmp(std::shared_ptr<BinaryCmpNode> tree)
{
    walkChildren(tree);
    auto lhsType = tree->getLHS()->evaluatedType;
    auto rhsType = tree->getRHS()->evaluatedType;
    auto op = tree->op;
    switch (op) {
    case BINOP::LTHAN:
    case BINOP::GTHAN:
    case BINOP::LEQ:
    case BINOP::GEQ:
        promotedType->possiblyPromoteBinop(tree->getLHS(), tree->getRHS(), promotedType->promotionTable); //  right now, only handles vectors. make sure rhs and lhs vectors are same type.promote if neccesary
        tree->evaluatedType = promotedType->getType(promotedType->comparisonResult, tree->getLHS(), tree->getRHS(), tree);
        break;
    case BINOP::EQUAL:
    case BINOP::NEQUAL:
        promotedType->possiblyPromoteBinop(tree->getLHS(), tree->getRHS(), promotedType->promotionTable); //  right now, only handles vectors. make sure rhs and lhs vectors are same type.promote if neccesary
        if (tree->getLHS()->evaluatedType->getBaseTypeEnumName() == "tuple" or tree->getRHS()->evaluatedType->getBaseTypeEnumName() == "tuple") {
            if (tree->getLHS()->evaluatedType->baseTypeEnum == TYPE::NULL_ or tree->getLHS()->evaluatedType->baseTypeEnum == TYPE::IDENTITY) {
                promotedType->promoteIdentityAndNull(tree->getRHS()->evaluatedType, tree->getLHS());
            } else if (tree->getRHS()->evaluatedType->baseTypeEnum == TYPE::NULL_ or tree->getRHS()->evaluatedType->baseTypeEnum == TYPE::IDENTITY) {
                promotedType->promoteIdentityAndNull(tree->getLHS()->evaluatedType, tree->getRHS());
            }
            if (tree->getLHS()->evaluatedType->tupleChildType.size() != tree->getRHS()->evaluatedType->tupleChildType.size()) {
                throw TypeError(tree->loc(), "#lvalues != #rvalues when comparing tuples.");
            }
            for (size_t i = 0; i < tree->getRHS()->evaluatedType->tupleChildType.size(); i++) {
                auto leftTypeString = tree->getLHS()->evaluatedType->tupleChildType[i].second->getBaseTypeEnumName();
                auto rightTypeString = tree->getRHS()->evaluatedType->tupleChildType[i].second->getBaseTypeEnumName();
                if ((promotedType->isVector(tree->getLHS()->evaluatedType->tupleChildType[i].second) and promotedType->isVector(tree->getRHS()->evaluatedType->tupleChildType[i].second))
                    or (promotedType->isMatrix(tree->getLHS()->evaluatedType->tupleChildType[i].second) and promotedType->isMatrix(tree->getRHS()->evaluatedType->tupleChildType[i].second))
                    or (promotedType->isScalar(tree->getLHS()->evaluatedType->tupleChildType[i].second) and promotedType->isScalar(tree->getRHS()->evaluatedType->tupleChildType[i].second))) {
                    if (leftTypeString != rightTypeString) {
                        auto leftIndex = promotedType->getTypeIndex(leftTypeString);
                        auto rightIndex = promotedType->getTypeIndex(rightTypeString);
                        std::string resultTypeString = promotedType->equalityResult[leftIndex][rightIndex];
                        if (resultTypeString.empty()) {
                            throw TypeError(tree->loc(), "Cannot perform equality operation between " + leftTypeString + " to " + rightTypeString);
                        }
                    }
                } else {
                    throw TypeError(tree->loc(), "Cannot perform equality operation between " + leftTypeString + " to " + rightTypeString);
                }
            }
            tree->evaluatedType = currentScope->resolveType("boolean");
        } else {
            tree->evaluatedType = promotedType->getType(promotedType->equalityResult, tree->getLHS(), tree->getRHS(), tree);
        }
        break;
    }
    return nullptr;
}

std::any TypeWalker::visitUnaryArith(std::shared_ptr<UnaryArithNode> tree)
{
    walkChildren(tree);
    switch (tree->op) {
    case NOT: {
        if (tree->getExpr()->evaluatedType->baseTypeEnum != BOOLEAN) {
            throw TypeError(tree->loc(), "cannot apply unaryNot on non boolean");
        }
    }
    }
    tree->evaluatedType = tree->getExpr()->evaluatedType;
    return nullptr;
}

std::any TypeWalker::visitID(std::shared_ptr<IDNode> tree)
{
    if (tree->sym == nullptr) {
        throw SymbolError(tree->loc(), "Unidentified Symbol referenced!");
    }
    tree->evaluatedType = tree->sym->typeSym;
    return nullptr;
}

std::any TypeWalker::visitTupleIndex(std::shared_ptr<TupleIndexNode> tree)
{
    walkChildren(tree);
    auto tupleId = std::dynamic_pointer_cast<IDNode>(tree->children[0]);
    if (std::dynamic_pointer_cast<IDNode>(tree->children[1])) {
        auto index = std::dynamic_pointer_cast<IDNode>(tree->children[1])->getName();
        for (auto c : tupleId->evaluatedType->tupleChildType) {
            if (index == c.first) {
                tree->evaluatedType = std::dynamic_pointer_cast<Type>(c.second);
                break;
            }
        }
        if (!tree->evaluatedType) {
            throw SymbolError(tree->loc(), "Undefined tuple index referenced");
        }
    } else if (std::dynamic_pointer_cast<IntNode>(tree->children[1])) {
        auto index = std::dynamic_pointer_cast<IntNode>(tree->children[1])->getVal();
        if (index < 1 || index > tupleId->evaluatedType->tupleChildType.size()) {
            throw SymbolError(tree->loc(), "Out of bound tuple index referenced");
        }
        tree->evaluatedType = std::dynamic_pointer_cast<Type>(tupleId->evaluatedType->tupleChildType[index - 1].second);
    }
    return nullptr;
}

std::any TypeWalker::visitDecl(std::shared_ptr<DeclNode> tree)
{
    walkChildren(tree);
    if (!tree->getTypeNode()) {
        tree->sym->typeSym = tree->getExprNode()->evaluatedType;
        if (tree->getExprNode()->evaluatedType->baseTypeEnum == TYPE::IDENTITY || tree->getExprNode()->evaluatedType->baseTypeEnum == TYPE::NULL_) {
            throw TypeError(tree->loc(), "cannot have identity or null when type is not defined");
        }
        tree->evaluatedType = tree->getExprNode()->evaluatedType;
        return nullptr;
    }
    if (!tree->getExprNode()) {
        // has a type node //  add a null node :) TODO: this identity and null handling is different for matrices and vector
        std::shared_ptr<ASTNode> nullNode = std::make_shared<NullNode>(tree->loc());
        tree->addChild(nullNode);
        walk(nullNode); // walk null node to popualte the type

        auto lType = tree->sym->typeSym;
        if (lType == nullptr) {
            throw SyntaxError(tree->loc(), "Declaration is missing expression to infer type.");
        }
        tree->evaluatedType = lType; //
        tree->getExprNode()->evaluatedType = lType; // set identity/null node type to this type for promotion
        tree->getTypeNode()->evaluatedType = lType;
        auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->getTypeNode());
        if (tupleNode) {
            for (int i = 0; i < tupleNode->getTypes().size(); i++) {
                tupleNode->getTypes()[i]->evaluatedType = lType->tupleChildType[i].second;
            }
        }
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
        // populate the tuple type nodes
        auto mTree = std::dynamic_pointer_cast<TupleTypeNode>(tree->getTypeNode());
        for (int i = 0; i < mTree->getTypes().size(); i++) {
            mTree->getTypes()[i]->evaluatedType = lType->tupleChildType[i].second;
        }

        tree->getTypeNode()->evaluatedType = lType;
        auto child = mTree->getTypes();
        auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->getTypeNode());
        if (tupleNode->innerTypes.size() != rType->tupleChildType.size()) {
            throw TypeError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
        }
        for (size_t i = 0; i < rType->tupleChildType.size(); i++) {
            // auto leftTypeString = std::dynamic_pointer_cast<TypeNode>(tupleNode->innerTypes[i].second)->getTypeName();
            auto leftTypeString = lType->tupleChildType[i].second->getBaseTypeEnumName();
            auto rightTypeString = rType->tupleChildType[i].second->getBaseTypeEnumName();

            if (leftTypeString != rightTypeString) {
                auto leftIndex = promotedType->getTypeIndex(rightTypeString);
                auto rightIndex = promotedType->getTypeIndex(leftTypeString);
                std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
                if (resultTypeString.empty()) {
                    throw TypeError(tree->loc(), "Cannot implicitly promote " + rightTypeString + " to " + leftTypeString);
                }
            }
            // rType->tupleChildType[i].second =
        }
        promotedType->promoteTupleElements(lType, tree->getExprNode(), promotedType->promotionTable);
        // tree->getTypeNode()->evaluatedType = symtab->resolveTypeUser(tupleNode);
        // tree->evaluatedType = symtab->resolveTypeUser(tupleNode);
        tree->evaluatedType = lType;
        return nullptr;

    } else if (lType->vectorOrMatrixEnum == TYPE::VECTOR) {
        // promote all RHS vector element to ltype if exprNode is a vectorNode

        if (promotedType->isMatrix(lType) && promotedType->isVector(tree->getExprNode()->evaluatedType)) {
            throw TypeError(tree->loc(), "cannot promote vectorNode to matrix");
        }
        if (rType->vectorOrMatrixEnum == NONE && promotedType->isScalar(rType)) {
            promotedType->promoteLiteralToArray(lType, tree->getExprNode(), promotedType->promotionTable);
        } else if (promotedType->isEmptyArrayLiteral(rType)) {
            // empty array literal. simply promote to ltype
            // promotedType->emptyArrayErrorCheck(tree->getTypeNode());
            tree->getExprNode()->evaluatedType = promotedType->getTypeCopy(lType);
        } else {
            promotedType->promoteVectorElements(lType, tree->getExprNode(), promotedType->promotionTable);
            promotedType->updateVectorNodeEvaluatedType(lType, tree->getExprNode()); // copy ltype to exprNode's type except for the size attribute
        }
        auto typeCopy = promotedType->getTypeCopy(tree->getExprNode()->evaluatedType); // copy the vectorLiteral's type into this node(mostly to copy the size attribute
        tree->evaluatedType = typeCopy;
        tree->getTypeNode()->evaluatedType = tree->evaluatedType;
        tree->sym->typeSym = tree->evaluatedType; // update the identifier's type
#ifdef DEBUG
        promotedType->printTypeClass(tree->evaluatedType);
#endif
        return nullptr;
    } else if (rType->getBaseTypeEnumName() == "null" || rType->getBaseTypeEnumName() == "identity") { // if it null then we just set it to ltype
        tree->evaluatedType = lType;
        tree->getTypeNode()->evaluatedType = lType; //
        promotedType->promoteIdentityAndNull(lType, tree->getExprNode());
    } else if (lType->getBaseTypeEnumName() != rType->getBaseTypeEnumName()) {
        auto leftIndex = promotedType->getTypeIndex(rType->getBaseTypeEnumName());
        auto rightIndex = promotedType->getTypeIndex(lType->getBaseTypeEnumName());
        std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
        if (resultTypeString.empty()) {
            throw TypeError(tree->loc(), "Cannot implicitly promote " + rType->getBaseTypeEnumName() + " to " + lType->getBaseTypeEnumName());
        }
        auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));
        tree->evaluatedType = resultType;
        tree->getTypeNode()->evaluatedType = resultType;
    } else { // normal implicit promotion
        tree->getTypeNode()->evaluatedType = rType;
        tree->evaluatedType = rType;
        tree->sym->typeSym = tree->evaluatedType;
    }
    return nullptr;
}
std::any TypeWalker::visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree)
{
    walk(tree->getCondition());
    if (tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::IDENTITY || tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::NULL_) {
        throw TypeError(tree->loc(), "idenitty in loop cond");
    }
    return nullptr;
}
std::any TypeWalker::visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree)
{
    walk(tree->getCondition());
    if (tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::IDENTITY || tree->getCondition()->evaluatedType->baseTypeEnum == TYPE::NULL_) {
        throw TypeError(tree->loc(), "idenitty in loop cond");
    }
    return nullptr;
}

std::any TypeWalker::visitAssign(std::shared_ptr<AssignNode> tree)
{
    walkChildren(tree);
    auto rhsType = tree->getRvalue()->evaluatedType;
    auto lhsCount = tree->getLvalue()->children.size();
    auto exprList = std::dynamic_pointer_cast<ExprListNode>(tree->getLvalue());

    if (lhsCount == 1) {
        if (std::dynamic_pointer_cast<IDNode>(exprList->children[0])) {
            auto lvalue = std::dynamic_pointer_cast<IDNode>(exprList->children[0]);
            auto symbol = lvalue->sym;
            if (symbol != nullptr and symbol->qualifier == QUALIFIER::CONST) {
                throw AssignError(tree->loc(), "Cannot assign to const");
            }
        }

        else if (std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0])) {
            auto lvalue = std::dynamic_pointer_cast<IDNode>(std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0])->children[0]);
            auto symbol = lvalue->sym;
            if (symbol != nullptr and symbol->qualifier == QUALIFIER::CONST) {
                throw AssignError(tree->loc(), "Cannot assign to const");
            }
        }

        else if (std::dynamic_pointer_cast<IndexNode>(exprList->children[0])) {
            auto lvalue = std::dynamic_pointer_cast<IDNode>(std::dynamic_pointer_cast<IndexNode>(exprList->children[0])->children[0]);
            auto symbol = lvalue->sym;
            if (symbol != nullptr and symbol->qualifier == QUALIFIER::CONST) {
                throw AssignError(tree->loc(), "Cannot assign to const");
            }
        }

        else {
            throw SyntaxError(tree->loc(), "incorrect lvalue");
        }

        if (rhsType != nullptr) {
            if (tree->getLvalue()->children[0]->evaluatedType == nullptr) {
                tree->evaluatedType = rhsType;
                return nullptr;
            }
            // TODO identity and null handling
            if (std::dynamic_pointer_cast<IDNode>(exprList->children[0])) {
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
                    promotedType->promoteTupleElements(lvalue->evaluatedType, tree->getRvalue(), promotedType->promotionTable);
                    // promotedType->printTypeClass(tree->getRvalue()->evaluatedType);

                    tree->evaluatedType = lvalue->evaluatedType;
                } else if (lvalue->evaluatedType->vectorOrMatrixEnum == TYPE::VECTOR) {
                    // handle vector literal. promote rhs
                    if (promotedType->isMatrix(lvalue->evaluatedType) && promotedType->isVector(tree->getRvalue()->evaluatedType)) {
                        throw TypeError(tree->loc(), "cannot promote vectorNode to matrix");
                    }
                    if (rhsType->vectorOrMatrixEnum == NONE && promotedType->isScalar(rhsType)) {
                        promotedType->promoteLiteralToArray(lvalue->evaluatedType, tree->getRvalue(), promotedType->promotionTable);
                    } else if (promotedType->isEmptyArrayLiteral(rhsType)) {
                        tree->getRvalue()->evaluatedType = promotedType->getTypeCopy(lvalue->evaluatedType);
                    } else {
                        promotedType->promoteVectorElements(lvalue->evaluatedType, tree->getRvalue(), promotedType->promotionTable);
                        promotedType->updateVectorNodeEvaluatedType(lvalue->evaluatedType, tree->getRvalue());
                    }
                    tree->evaluatedType = promotedType->getTypeCopy(tree->getRvalue()->evaluatedType); // update the tree evaluated type with promoted
                    return nullptr;
                } else if (rhsType->getBaseTypeEnumName() == "null" || rhsType->getBaseTypeEnumName() == "identity") { // if it null then we just set it to ltype
                    tree->evaluatedType = lvalue->evaluatedType; //
                    promotedType->promoteIdentityAndNull(lvalue->evaluatedType, tree->getRvalue());
                }

                if (tree->getRvalue()->evaluatedType->getBaseTypeEnumName() != tree->getLvalue()->children[0]->evaluatedType->getBaseTypeEnumName())
                    tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getRvalue(), lvalue, tree);
                else
                    tree->evaluatedType = tree->getRvalue()->evaluatedType;
            } else if (std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0])) {
                auto tupleIndexNode = std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[0]);
                if (rhsType->getBaseTypeEnumName() == "tuple") {
                    throw TypeError(tree->loc(), "Cannot assign tuple to tuple index");
                }
                // Tuple Index is a vector or a matrix
                else if (tupleIndexNode->evaluatedType->vectorOrMatrixEnum == TYPE::VECTOR) {
                    // handle vector literal. promote rhs
                    if (promotedType->isMatrix(tupleIndexNode->evaluatedType) && promotedType->isVector(tree->getRvalue()->evaluatedType)) {
                        throw TypeError(tree->loc(), "cannot promote vectorNode to matrix");
                    }
                    if (rhsType->vectorOrMatrixEnum == NONE && promotedType->isScalar(rhsType)) {
                        promotedType->promoteLiteralToArray(tupleIndexNode->evaluatedType, tree->getRvalue(), promotedType->promotionTable);
                    } else if (promotedType->isEmptyArrayLiteral(rhsType)) {
                        tree->getRvalue()->evaluatedType = promotedType->getTypeCopy(tupleIndexNode->evaluatedType);
                    } else {
                        promotedType->promoteVectorElements(tupleIndexNode->evaluatedType, tree->getRvalue(), promotedType->promotionTable);
                        promotedType->updateVectorNodeEvaluatedType(tupleIndexNode->evaluatedType, tree->getRvalue());
                    }
                    tree->evaluatedType = promotedType->getTypeCopy(tree->getRvalue()->evaluatedType); // update the tree evaluated type with promoted
                    return nullptr;
                }

                else if (rhsType->getBaseTypeEnumName() == "null" || rhsType->getBaseTypeEnumName() == "identity") { // if it null then we just set it to ltype
                    tree->evaluatedType = tupleIndexNode->evaluatedType; //
                    promotedType->promoteIdentityAndNull(tupleIndexNode->evaluatedType, tree->getRvalue());
                }
                if (tree->getRvalue()->evaluatedType->getBaseTypeEnumName() != tupleIndexNode->evaluatedType->getBaseTypeEnumName())
                    tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getRvalue(), tupleIndexNode, tree);
                else
                    tree->evaluatedType = tree->getRvalue()->evaluatedType;
            } else if (std::dynamic_pointer_cast<IndexNode>(exprList->children[0])) {
                auto vecMatrixIndexNode = std::dynamic_pointer_cast<IndexNode>(exprList->children[0]);
                if (rhsType->getBaseTypeEnumName() == "null" || rhsType->getBaseTypeEnumName() == "identity") { // if it null then we just set it to ltype
                    tree->evaluatedType = vecMatrixIndexNode->evaluatedType; //
                    promotedType->promoteIdentityAndNull(vecMatrixIndexNode->evaluatedType, tree->getRvalue());
                }
                if (rhsType->getBaseTypeEnumName() == "tuple") {
                    throw TypeError(tree->loc(), "Cannot assign tuple to vector index");
                }
                if (tree->getRvalue()->evaluatedType->getBaseTypeEnumName() != vecMatrixIndexNode->evaluatedType->getBaseTypeEnumName())
                    tree->evaluatedType = promotedType->getType(promotedType->promotionTable, tree->getRvalue(), vecMatrixIndexNode, tree);
                else
                    tree->evaluatedType = tree->getRvalue()->evaluatedType;
            } else {
                throw SyntaxError(tree->loc(), "incorrect lvalue");
            }
        }
    } else {
        if (rhsType->getBaseTypeEnumName() != "tuple")
            throw TypeError(tree->loc(), "Tuple Unpacking requires a tuple as the r-value.");
        if (lhsCount != rhsType->tupleChildType.size())
            throw AssignError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
        for (size_t i = 0; i < rhsType->tupleChildType.size(); i++) {
            std::string leftTypeString;
            std::shared_ptr<ExprNode> lvalue;
            if (std::dynamic_pointer_cast<IDNode>(exprList->children[i])) {
                lvalue = std::dynamic_pointer_cast<IDNode>(exprList->children[i]);
                leftTypeString = lvalue->evaluatedType->getBaseTypeEnumName();
            } else if (std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[i])) {
                lvalue = std::dynamic_pointer_cast<TupleIndexNode>(exprList->children[i]);
                leftTypeString = lvalue->evaluatedType->getBaseTypeEnumName();
            } else if (std::dynamic_pointer_cast<IndexNode>(exprList->children[i])) {
                lvalue = std::dynamic_pointer_cast<IndexNode>(exprList->children[i]);
                leftTypeString = lvalue->evaluatedType->getBaseTypeEnumName();
            }

            if (leftTypeString.empty()) {
                throw SyntaxError(tree->loc(), "incorrect lvalue");
            }

            auto rightTypeString = rhsType->tupleChildType[i].second->getBaseTypeEnumName();

            if (lvalue->evaluatedType->vectorOrMatrixEnum == TYPE::VECTOR) {
                // handle vector literal. promote rhs
                if (promotedType->isMatrix(lvalue->evaluatedType) && promotedType->isVector(rhsType->tupleChildType[i].second)) {
                    throw TypeError(tree->loc(), "cannot promote vectorNode to matrix");
                }
                if (rhsType->vectorOrMatrixEnum == NONE && promotedType->isScalar(rhsType)) {
                    promotedType->promoteLiteralToArray(lvalue->evaluatedType, std::dynamic_pointer_cast<TupleNode>(tree->getRvalue())->getVal()[i], promotedType->promotionTable);
                } else if (promotedType->isEmptyArrayLiteral(rhsType)) {
                    std::dynamic_pointer_cast<TupleNode>(tree->getRvalue())->getVal()[i]->evaluatedType = promotedType->getTypeCopy(lvalue->evaluatedType);
                } else {
                    promotedType->promoteVectorElements(lvalue->evaluatedType, std::dynamic_pointer_cast<TupleNode>(tree->getRvalue())->getVal()[i], promotedType->promotionTable);
                    promotedType->updateVectorNodeEvaluatedType(lvalue->evaluatedType, std::dynamic_pointer_cast<TupleNode>(tree->getRvalue())->getVal()[i]);
                }
                return nullptr;
            }

            else if (leftTypeString != rightTypeString) {
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

std::any TypeWalker::visitStreamOut(std::shared_ptr<StreamOut> tree)
{
    // streamOut supports the following types:
    // - char, integer, real, boolean
    // - vector, string, matrix (part 2)
    // basically, NOT tuples
    std::vector<TYPE> allowedTypes = { TYPE::CHAR, TYPE::INTEGER, TYPE::REAL, TYPE::BOOLEAN, TYPE::STRING };

    walkChildren(tree);
    auto exprType = tree->getExpr()->evaluatedType;
    tree->evaluatedType = exprType;
    if (exprType != nullptr) {
        if (std::find(allowedTypes.begin(), allowedTypes.end(), exprType->baseTypeEnum) == allowedTypes.end()) {
            throw TypeError(tree->loc(), "Cannot stream out a " + typeEnumToString(exprType->baseTypeEnum));
        }
    } else {
        throw TypeError(tree->loc(), "Cannot stream out unknown type");
    }
#ifdef DEBUG

    std::cout << "in streamout, evaltype=";
    promotedType->printTypeClass(tree->evaluatedType);
#endif
    return nullptr;
}

std::any TypeWalker::visitStreamIn(std::shared_ptr<StreamIn> tree)
{
    // streamIn supports reading into the following types:
    // - char, integer, real, boolean
    // NOT any other types
    std::vector<TYPE> allowedTypes = { TYPE::CHAR, TYPE::INTEGER, TYPE::REAL, TYPE::BOOLEAN };
    walkChildren(tree);
    auto exprType = tree->getExpr()->evaluatedType;
    if (exprType != nullptr) {
        if (std::find(allowedTypes.begin(), allowedTypes.end(), exprType->baseTypeEnum) == allowedTypes.end()) {
            throw TypeError(tree->loc(), "Cannot stream in a " + typeEnumToString(exprType->baseTypeEnum));
        }
    } else {
        throw TypeError(tree->loc(), "Cannot stream out unknown type");
    }

    if (!std::dynamic_pointer_cast<IDNode>(tree->getExpr()) && !std::dynamic_pointer_cast<TupleIndexNode>(tree->getExpr()) && !std::dynamic_pointer_cast<IndexNode>(tree->getExpr())) {
        throw SyntaxError(tree->loc(), "incorrect lvalue for streamin!");
    }

    return nullptr;
}

std::any TypeWalker::visitCall(std::shared_ptr<CallNode> tree)
{
    if (!tree->MethodRef->isBuiltIn() && tree->MethodRef->orderedArgs.size() != tree->children.size()) {
        throw SyntaxError(tree->loc(), "call argument size do not match method definition");
    }
    if (tree->procCall) {
        walkChildren(tree);
        tree->evaluatedType = tree->MethodRef->typeSym; // could be null if procedure call dont have ret type
    } else {
        // must be an expression then
        //  must be a function call
        walkChildren(tree);
        tree->evaluatedType = tree->MethodRef->typeSym;
        auto functionSym = tree->MethodRef;
        std::shared_ptr<Type> argType = nullptr;
        if (!tree->children.empty()) {
            argType = tree->children[0]->evaluatedType;
        }
        if (!functionSym->isBuiltIn()) {
            return nullptr;
        }
        switch (functionSym->functypeENUM) {
        case FUNC_LENGTH:
            // typecheck must be vector
            assert(tree->children.size() == 1); // all invalid arg size builtin call should be weeded out by defref
            if (!promotedType->isVector(argType)) {
                throw TypeError(tree->loc(), "length() only accepts vector as input");
            }
            tree->evaluatedType = symtab->globalScope->resolveType("integer"); // return type should just be int
            break;
        case FUNC_ROW:
        case FUNC_COLUMN:
            assert(tree->children.size() == 1); // all invalid arg size builtin call should be weeded out by defref
            if (!promotedType->isMatrix(argType)) {
                throw TypeError(tree->loc(), "rows() or columns() only accepts matrix as input");
            }
            tree->evaluatedType = symtab->globalScope->resolveType("integer"); // return type should just be int
            break;
        case FUNC_REVERSE: {
            assert(tree->children.size() == 1); // all invalid arg size builtin call should be weeded out by defref
            if (!promotedType->isVector(argType)) {
                throw TypeError(tree->loc(), "reverse only accepts vector as input");
            }
            auto argVectorType = promotedType->createArrayType(argType->getBaseTypeEnumName(), VECTOR);
            tree->evaluatedType = argVectorType;
            break;
        }
        case FUNC_FORMAT: {
            assert(tree->children.size() == 1); // all invalid arg size builtin call should be weeded out by defref
            if (!promotedType->isScalar(argType)) {
                throw TypeError(tree->loc(), "format only accepts scalar as input");
            }
            auto argVectorType = promotedType->createArrayType("character", VECTOR);
            tree->evaluatedType = argVectorType;
            break;
        }
        case FUNC_SSTATE:
            assert(tree->children.size() == 1); // all invalid arg size builtin call should be weeded out by defref
            tree->evaluatedType = promotedType->getTypeCopy(symtab->globalScope->resolveType("integer"));
            break;
        }
#ifdef DEBUG
        promotedType->printTypeClass(tree->evaluatedType);
#endif
    }
    return nullptr;
}

std::any TypeWalker::visitCast(std::shared_ptr<CastNode> tree)
{
    walkChildren(tree);
    auto toType = symtab->resolveTypeUser(tree->children[0]);
    auto exprType = tree->children[1]->evaluatedType;
    if (toType->getBaseTypeEnumName() == "tuple" && exprType->getBaseTypeEnumName() == "tuple") {
        auto mTree = std::dynamic_pointer_cast<TupleTypeNode>(tree->getType());
        for (int i = 0; i < mTree->getTypes().size(); i++) {
            mTree->getTypes()[i]->evaluatedType = toType->tupleChildType[i].second;
        }

        if (toType->tupleChildType.size() != exprType->tupleChildType.size()) {
            throw TypeError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
        }
        for (size_t i = 0; i < exprType->tupleChildType.size(); i++) {
            auto leftTypeString = exprType->tupleChildType[i].second->getBaseTypeEnumName();
            auto rightTypeString = toType->tupleChildType[i].second->getBaseTypeEnumName();

            if (leftTypeString != rightTypeString) {
                auto leftIndex = promotedType->getTypeIndex(leftTypeString);
                auto rightIndex = promotedType->getTypeIndex(rightTypeString);
                std::string resultTypeString = promotedType->castTable[leftIndex][rightIndex];
                if (resultTypeString.empty()) {
                    throw TypeError(tree->loc(), "Cannot cast " + leftTypeString + " to " + rightTypeString);
                }
            }
        }

        promotedType->promoteTupleElements(toType, tree->getExpr(), promotedType->castTable);
        // tree->children[0]->evaluatedType = toType;
        tree->getType()->evaluatedType = toType;
        tree->evaluatedType = toType; // tuple Type
    } else if (toType->getBaseTypeEnumName() == "tuple" || toType->getBaseTypeEnumName() == "tuple") {
        throw TypeError(tree->loc(), "only tuple to tuple casting is permitted");
    } else {
        auto leftIndex = promotedType->getTypeIndex(exprType->getBaseTypeEnumName());
        auto rightIndex = promotedType->getTypeIndex(toType->getBaseTypeEnumName());
        std::string resultTypeString = promotedType->castTable[leftIndex][rightIndex];
        if (resultTypeString.empty()) {
            throw TypeError(tree->loc(), "Cannot cast " + exprType->getBaseTypeEnumName() + " to " + toType->getBaseTypeEnumName());
        }
        auto resultType = std::dynamic_pointer_cast<Type>(currentScope->resolveType(resultTypeString));
        // array handling
        if (toType->vectorOrMatrixEnum == VECTOR) {
            if (promotedType->isMatrix(toType) && promotedType->isVector(tree->getExpr()->evaluatedType)) {
                throw TypeError(tree->loc(), "cannot promote vectorNode to matrix");
            }
            if (promotedType->isScalar(exprType)) {
                promotedType->promoteLiteralToArray(toType, tree->getExpr(), promotedType->castTable);
            } else if (promotedType->isEmptyArrayLiteral(exprType)) {
                // empty array literal. simply promote to ltype
                // promotedType->emptyArrayErrorCheck(tree->getTypeNode());
                tree->getExpr()->evaluatedType = promotedType->getTypeCopy(toType);
            } else {
                promotedType->promoteVectorElements(toType, tree->getExpr(), promotedType->castTable);
                promotedType->updateVectorNodeEvaluatedType(toType, tree->getExpr()); // copy ltype to exprNode's type except for the size attribute
            }
        }
        if ((exprType->baseTypeEnum == TYPE::NULL_) || (exprType->baseTypeEnum == TYPE::IDENTITY)) {
            tree->getExpr()->evaluatedType = promotedType->getTypeCopy(toType);
        }
        // array handling;
        tree->evaluatedType = promotedType->getTypeCopy(toType); // base Type
        tree->getType()->evaluatedType = promotedType->getTypeCopy(tree->evaluatedType);
    }
    return nullptr;
}

std::any TypeWalker::visitTypedef(std::shared_ptr<TypeDefNode> tree)
{
    auto typeNode = tree->children[0];
    tree->evaluatedType = symtab->resolveTypeUser(typeNode);
    tree->children[0]->evaluatedType = tree->evaluatedType;
    return nullptr;
}

std::any TypeWalker::visitVector(std::shared_ptr<VectorNode> tree)
{
    // check if this is a matrix or vector
    // innertype(evaluatedType->baseTypeEnum) will be set by the declaration node

    for (auto& exprNode : tree->getElements()) {
        walk(exprNode); // set the evaluated type of each expr
    }

    tree->evaluatedType = std::make_shared<AdvanceType>(""); // just initialize it
    tree->evaluatedType->vectorOrMatrixEnum = TYPE::VECTOR;
    if (tree->getElements().empty()) {
        // we will represent empty vector literal if it is vector but doesnt have a type
        tree->evaluatedType->baseTypeEnum = NONE;
        tree->isEmpty = true;
        return nullptr;
    }

    // TODO handle empty vector
    // promote every element to the dominant type
    auto bestType = promotedType->getDominantTypeFromVector(tree);
    if (bestType->baseTypeEnum == TYPE::IDENTITY || bestType->baseTypeEnum == TYPE::NULL_) {
        tree->evaluatedType->baseTypeEnum = bestType->baseTypeEnum;
        return nullptr;
    }
    promotedType->promoteVectorElements(bestType, tree, promotedType->promotionTable); // now every elements of vector have same type
    tree->evaluatedType->baseTypeEnum = tree->getElement(0)->evaluatedType->baseTypeEnum; // this will be modified by visitDecl when we promote all RHS
    // add the inner types to type class
    promotedType->populateInnerTypes(tree->evaluatedType, tree);

    tree->evaluatedType->dims.push_back(tree->getSize()); // the row size of this vector
    if (tree->getElement(0)->evaluatedType->vectorOrMatrixEnum == VECTOR) { // pul column size of there is any
        // assert(!tree->getElement(0)->evaluatedType->dims.empty());
        // tree->evaluatedType->dims.push_back(tree->getElement(0)->evaluatedType->dims[0]);  // TODO:
    }
    // TODO: make sure that matrix element must be vector
    return nullptr;
}
// std::any TypeWalker::visitMatrix(std::shared_ptr<MatrixNode> tree) {
//     // innertype(evaluatedType->baseTypeEnum) will be set by the declaration node
//     // NOTE: empty matrices will not gonna be here
//     int maxsize = INT32_MIN;
//     for (auto &vec: tree->getElements()) {
//         maxsize = std::max(maxsize, vec->getSize());  // get the largest inner vector size
//     }

//    walkChildren(tree);  // populate all the evaluated type

//    tree->evaluatedType = std::make_shared<AdvanceType>("");  // just initialize it
//    tree->evaluatedType->vectorOrMatrixEnum = TYPE::VECTOR;
//    // function to make sure all inner vectors have same type

//    // promote all elements
//    tree->evaluatedType = std::make_shared<AdvanceType>("matrix");
//    tree->evaluatedType->vectorOrMatrixEnum = TYPE::MATRIX;
//    return nullptr;
//}

std::any TypeWalker::visitFilter(std::shared_ptr<FilterNode> tree)
{
    // filter is just an integer tuple right?
    walkChildren(tree);
    auto tupleSize = tree->getExprList().size() + 1; // + 1 for the last vector that did not satisfy any condition
    auto tupleType = std::make_shared<AdvanceType>("tuple");
    for (int i = 0; i < tupleSize; i++) {
        auto typeOfTupleVec = promotedType->getTypeCopy(tree->getDomain()->evaluatedType); // vector type
        tupleType->tupleChildType.push_back(std::make_pair("", typeOfTupleVec));
    }
    tree->evaluatedType = tupleType;
    return nullptr;
}
std::any TypeWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree)
{
    walkChildren(tree);
    tree->evaluatedType = tree->getVectDomain()->evaluatedType;
    auto exprType = tree->getExpr()->evaluatedType;
    // walk the domain var to resolve them first
    // [i in 1..10 | 'a'] we want to make the evaltype char vec
    // [i in ['a', 'a', 'a' | i] we want ot mae evaltype char vect. first ensure 'i' is type char by copying from its domainvec
    if (tree->getVectDomain()) {
        // this is a vector generator
        tree->domainVar1Sym->typeSym = promotedType->getTypeCopy(currentScope->resolveType(tree->getVectDomain()->evaluatedType->getBaseTypeEnumName()));
        walk(tree->getExpr()); // walk again to update evaltype
        exprType = tree->getExpr()->evaluatedType;
        tree->evaluatedType = promotedType->createArrayType(exprType->getBaseTypeEnumName(), VECTOR);
    } else {
        // this is a matrix generator
        tree->domainVar1Sym->typeSym = promotedType->getTypeCopy(currentScope->resolveType(tree->getVectDomain()->evaluatedType->getBaseTypeEnumName()));
        tree->domainVar2Sym->typeSym = promotedType->getTypeCopy(currentScope->resolveType(tree->getVectDomain()->evaluatedType->getBaseTypeEnumName()));
        walk(tree->getExpr()); // walk again to update evaltype
        exprType = tree->getExpr()->evaluatedType;
        tree->evaluatedType = promotedType->createArrayType(exprType->getBaseTypeEnumName(), MATRIX);
    }
#ifdef DEBUG
    promotedType->printTypeClass(tree->evaluatedType);
#endif
    return nullptr;
}

std::any TypeWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree)
{
    walkChildren(tree);
    auto typeCopy = promotedType->getTypeCopy(tree->getStart()->evaluatedType); // returns an integer scalar
    typeCopy->vectorInnerTypes.push_back(promotedType->getTypeCopy(typeCopy)); // push integer type as inner type
    typeCopy->vectorOrMatrixEnum = VECTOR;
    // create vector type
    tree->evaluatedType = typeCopy;
    return nullptr;
}
std::any TypeWalker::visitIndex(std::shared_ptr<IndexNode> tree)
{
    walkChildren(tree); // if a[b, c]. indexee=a, indexor1=b, indexor2=c

    auto indexee = tree->getIndexee();
    auto indexor1 = tree->getIndexor1();
    auto indexor2 = tree->getIndexor2();
    if (indexor2 == nullptr) {
        if (indexor1->evaluatedType->baseTypeEnum != INTEGER || indexor1->evaluatedType->vectorOrMatrixEnum != NONE) {
            throw TypeError(tree->loc(), "can only index vector using integer literal");
        }
    } else {
        if (indexor1->evaluatedType->baseTypeEnum != INTEGER || indexor1->evaluatedType->vectorOrMatrixEnum != NONE
            || indexor2->evaluatedType->baseTypeEnum != INTEGER || indexor2->evaluatedType->vectorOrMatrixEnum != NONE) {
            throw TypeError(tree->loc(), "can only index matrix using integer literal");
        }
    }

    if (promotedType->isVector(indexee->evaluatedType)) { // case: its a vector index
        auto typeCopy = promotedType->getTypeCopy(indexee->evaluatedType);
        tree->evaluatedType = typeCopy;
        tree->evaluatedType->vectorOrMatrixEnum = NONE;
        tree->evaluatedType->dims.clear(); // evaluted type is just a non vector literal with no dimention
    } else if (promotedType->isMatrix(indexee->evaluatedType)) { // case: its a matrix indx
        auto typeCopy = promotedType->getTypeCopy(currentScope->resolveType(indexee->evaluatedType->getBaseTypeEnumName()));
    } else {
        assert(indexee->evaluatedType->dims.empty());
        throw SyntaxError(tree->loc(), "cannot index non vector or matrices"); // TODO is this the correct error
    }
    return nullptr;
}

std::any TypeWalker::visitParameter(std::shared_ptr<ArgNode> tree)
{
    tree->evaluatedType = symtab->resolveTypeUser(tree->type);
    tree->type->evaluatedType = tree->evaluatedType;
    tree->idSym->typeNode = tree->type;
    auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->type);
    if (tupleNode) {
        for (int i = 0; i < tupleNode->getTypes().size(); i++) {
            tupleNode->getTypes()[i]->evaluatedType = symtab->resolveTypeUser(tupleNode->getTypes()[i]);
        }
    }
    return nullptr;
}

std::any TypeWalker::visitReturn(std::shared_ptr<ReturnNode> tree)
{
    if (tree->returnExpr) {
        walk(tree->returnExpr);
    } else {
        if (tree->returnFunc->typeSym) {
            throw TypeError(tree->loc(), "No return expression present");
        }
        return nullptr;
    }
    auto node = std::dynamic_pointer_cast<ASTNode>(tree);
    walkChildren(node);
    tree->evaluatedType = tree->returnExpr->evaluatedType;
    auto type = tree->returnFunc->typeSym;
    auto rhsType = tree->returnExpr->evaluatedType;
    if (type->getBaseTypeEnumName() == "tuple" and rhsType->getBaseTypeEnumName() == "tuple") {
        if (type->tupleChildType.size() != rhsType->tupleChildType.size()) {
            throw TypeError(tree->loc(), "#lvalues != #rvalues when unpacking tuple.");
        }
        for (size_t i = 0; i < rhsType->tupleChildType.size(); i++) {
            auto leftTypeString = type->tupleChildType[i].second->getBaseTypeEnumName();
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
        promotedType->promoteTupleElements(type, tree->returnExpr, promotedType->promotionTable);
        // promotedType->printTypeClass(tree->getRvalue()->evaluatedType);

        tree->evaluatedType = type;
    } else if (type->vectorOrMatrixEnum == TYPE::VECTOR) {
        // handle vector literal. promote rhs
        if (promotedType->isMatrix(type) && promotedType->isVector(rhsType)) {
            throw TypeError(tree->loc(), "cannot promote vectorNode to matrix");
        }
        if (rhsType->vectorOrMatrixEnum == NONE && promotedType->isScalar(rhsType)) {
            promotedType->promoteLiteralToArray(type, tree->returnExpr, promotedType->promotionTable);
        } else if (promotedType->isEmptyArrayLiteral(rhsType)) {
            tree->returnExpr->evaluatedType = promotedType->getTypeCopy(type);
            tree->evaluatedType = tree->returnExpr->evaluatedType;
        } else {
            promotedType->promoteVectorElements(type, tree->returnExpr, promotedType->promotionTable);
            promotedType->updateVectorNodeEvaluatedType(type, tree->returnExpr);
        }
        tree->evaluatedType = promotedType->getTypeCopy(tree->returnExpr->evaluatedType); // update the tree evaluated type with promoted
        return nullptr;
    } else if (rhsType->getBaseTypeEnumName() == "null" || rhsType->getBaseTypeEnumName() == "identity") { // if it null then we just set it to ltype
        tree->evaluatedType = type; //
        promotedType->promoteIdentityAndNull(type, tree->returnExpr);
    }

    else if (rhsType->getBaseTypeEnumName() != type->getBaseTypeEnumName()) {
        auto leftIndex = promotedType->getTypeIndex(tree->returnExpr->evaluatedType->getBaseTypeEnumName());
        auto rightIndex = promotedType->getTypeIndex(type->getBaseTypeEnumName());
        std::string resultTypeString = promotedType->promotionTable[leftIndex][rightIndex];
        if (resultTypeString.empty()) {
            throw TypeError(node->loc(), "Cannot implicitly promote " + tree->returnExpr->evaluatedType->getBaseTypeEnumName() + " to " + type->getBaseTypeEnumName());
        }
    } else
        tree->evaluatedType = tree->returnExpr->evaluatedType;
    return nullptr;
}

std::any TypeWalker::visitFunction(std::shared_ptr<FunctionNode> tree)
{
    for (auto arg : tree->orderedArgs) {
        walk(arg);
    }
    tree->evaluatedType = symtab->resolveTypeUser(tree->getRetTypeNode());
    tree->getRetTypeNode()->evaluatedType = tree->evaluatedType;
    auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->getRetTypeNode());
    if (tupleNode) {
        for (int i = 0; i < tupleNode->getTypes().size(); i++) {
            tupleNode->getTypes()[i]->evaluatedType = symtab->resolveTypeUser(tupleNode->getTypes()[i]);
        }
    }
    if (tree->body) {
        auto blockNode = tree->body;
        walkChildren(blockNode);
    } else if (tree->expr) {
        walkChildren(tree->expr);
    }
    return nullptr;
}

std::any TypeWalker::visitProcedure(std::shared_ptr<ProcedureNode> tree)
{
    for (auto arg : tree->orderedArgs) {
        walk(arg);
    }
    if (tree->getRetTypeNode()) {
        tree->evaluatedType = promotedType->getTypeCopy(symtab->resolveTypeUser(tree->getRetTypeNode()));
        tree->getRetTypeNode()->evaluatedType = tree->evaluatedType;
        auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(tree->getRetTypeNode());
        if (tupleNode) {
            for (int i = 0; i < tupleNode->getTypes().size(); i++) {
                tupleNode->getTypes()[i]->evaluatedType = symtab->resolveTypeUser(tupleNode->getTypes()[i]);
            }
        }
    } else {
        tree->evaluatedType = nullptr;
    }
    if (tree->body) {
        auto blockNode = tree->body;
        walkChildren(blockNode);
    }
    return nullptr;
}
std::any TypeWalker::visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree)
{
    // resolve the domain 1st
    for (auto& domainExpr : tree->getDomainExprs()) {
        auto domain = domainExpr.second;
        walk(domain);
    }
    for (auto& domainExpr : tree->getDomainExprs()) {
        auto domain = domainExpr.second;
        auto domainVar = domainExpr.first;
        // domain var's eval type is just a scalar since domainvec cannot be a matrix
        domainVar->typeSym = promotedType->getTypeCopy(currentScope->resolveType(domain->evaluatedType->getBaseTypeEnumName()));
    }
    return nullptr;
}

std::string TypeWalker::typeEnumToString(TYPE t)
{
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
