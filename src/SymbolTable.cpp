//
// Created by Joshua Ji on 2023-10-24.
//

#include "SymbolTable.h"
#include <sstream>

#include "SymbolTable.h"
#include "BaseScope.h"
#include "ASTNode/Type/TypeNode.h"
#include "ASTNode/Type/VectorTypeNode.h"
#include "ASTNode/Type/MatrixTypeNode.h"
#include "ASTNode/Type/StringTypeNode.h"
#include "ASTNode/Type/TupleTypeNode.h"
//#define DEBUG
std::shared_ptr<Scope> SymbolTable::enterScope(std::string& name, const std::shared_ptr<Scope>& enclosingScope) {
    std::shared_ptr<Scope> newScope = std::make_shared<LocalScope>(name, enclosingScope);
    scopes.push_back(newScope);
    return newScope;
}

std::shared_ptr<Scope> SymbolTable::enterScope(std::shared_ptr<Scope> newScope) {
    scopes.push_back(newScope);
    return newScope;
}

std::string SymbolTable::toString() {
    std::stringstream str;
    str << "SymbolTable {" << std::endl;
    for (const auto& s : scopes) {
        str << s->getScopeName() << ": " << s->toString() << std::endl;
    }
    str << "}" << std::endl;
    return str.str();
}


/*
 * if typeNode is a tuple, create a new type object for tuple, and populate the children's type
 * @ param: typeNode, the typenode to be resovled
 *
 * returns:
 *      if typenode == tuple, return tuple object type with children populated with types
 *      if typenode == basetype, return regular type object
 *
 */
std::shared_ptr<Type> SymbolTable::resolveTypeUser(std::shared_ptr<ASTNode> typeNode) {
    // cast it to vector or matrix type
    if (std::dynamic_pointer_cast<VectorTypeNode>(typeNode)) {
        // resolve innertyp
        auto typeN = std::dynamic_pointer_cast<VectorTypeNode>(typeNode);
#ifdef DEBUG
        std::cout << "Resolve Type: " << typeN->getTypeName() << std::endl;
#endif
        auto innerTypeN = std::dynamic_pointer_cast<TypeNode>(typeN->innerType);
        assert(typeN);
        auto innerTypeRes = globalScope->resolveType(innerTypeN->getTypeName());
        if (innerTypeRes == nullptr) {
            throw TypeError(typeNode->loc(), "cannot resolve innner type " + innerTypeN->getTypeName());
        }
        if (innerTypeRes->baseTypeEnum == TYPE::TUPLE || innerTypeRes->baseTypeEnum == TYPE::IDENTITY || innerTypeRes->baseTypeEnum == TYPE::NULL_) {
            throw (typeNode->loc(), "vector can only be int, real, boolean, char");
        }
        // create a new type object to set the advancedTYPE to TYPE::VECTOR

        std::shared_ptr<Type> resolvedType = std::make_shared<AdvanceType>(innerTypeRes->getName());  // create typenode with integer
        resolvedType->baseTypeEnum = innerTypeRes->baseTypeEnum;
        resolvedType->vectorOrMatrixEnum = TYPE::VECTOR;

        // make innertype node, which is a vector
        std::shared_ptr<Type> innerType_ = std::make_shared<AdvanceType>(innerTypeRes->getBaseTypeEnumName());
        innerType_->baseTypeEnum = innerTypeRes->baseTypeEnum;
        innerType_->vectorOrMatrixEnum = TYPE::NONE;  //  matrices is just vector of vector
        resolvedType->vectorInnerTypes.push_back(innerType_);
        return resolvedType;

    } else if (std::dynamic_pointer_cast<MatrixTypeNode>(typeNode)) {
        auto typeN = std::dynamic_pointer_cast<MatrixTypeNode>(typeNode);
        auto innerTypeN = std::dynamic_pointer_cast<TypeNode>(typeN->innerType);
        assert(typeN);
        auto innerTypeRes = globalScope->resolveType(innerTypeN->getTypeName());
        if (innerTypeRes == nullptr) {
            throw TypeError(typeNode->loc(), "cannot resolve innner type " + innerTypeN->getTypeName());
        }
        if (innerTypeRes->baseTypeEnum == TYPE::TUPLE || innerTypeRes->baseTypeEnum == TYPE::IDENTITY || innerTypeRes->baseTypeEnum == TYPE::NULL_) {
            throw (typeNode->loc(), "matrix can only be int, real, boolean, char");
        }
        // scalar type
        auto scalarType = std::make_shared<AdvanceType>(innerTypeRes->getBaseTypeEnumName());
        scalarType->baseTypeEnum = innerTypeRes->baseTypeEnum;
        scalarType->vectorOrMatrixEnum = TYPE::NONE;
        // create a new type object to set the advancedTYPE to TYPE::VECTOR
        std::shared_ptr<Type> resolvedType = std::make_shared<AdvanceType>(innerTypeRes->getBaseTypeEnumName());
        resolvedType->baseTypeEnum = innerTypeRes->baseTypeEnum;
        resolvedType->vectorOrMatrixEnum = TYPE::VECTOR;  //  matrices is just vector of vector

        // make innertype node, which is a vector
        std::shared_ptr<Type> innerType_ = std::make_shared<AdvanceType>(innerTypeRes->getBaseTypeEnumName());
        innerType_->baseTypeEnum = innerTypeRes->baseTypeEnum;
        innerType_->vectorOrMatrixEnum = TYPE::VECTOR;  //  matrices is just vector of vector

        innerType_->vectorInnerTypes.push_back(scalarType);
        resolvedType->vectorInnerTypes.push_back(innerType_);


        return resolvedType;

    } else if (std::dynamic_pointer_cast<StringTypeNode>(typeNode)){

    } else if (std::dynamic_pointer_cast<TupleTypeNode>(typeNode)) {
        // tuple typenode
        auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(typeNode);

        std::shared_ptr<Type> resolvedType = std::make_shared<AdvanceType>("tuple");
        resolvedType->baseTypeEnum = TYPE::TUPLE;

        auto tupleChild = tupleNode->innerTypes;  // vector of inner types
        for (auto c: tupleChild) {
            // recursively resolve children's type. resolves typedef too
            auto resolvedChild = resolveTypeUser(std::dynamic_pointer_cast<TypeNode>(c.second));
            if (resolvedChild == nullptr) throw TypeError(typeNode->loc(), "cannot resolve type in tuple");
            resolvedType->tupleChildType.push_back(std::make_pair(c.first, resolvedChild));
        }

        return resolvedType;

    }
    else {
        // base typenode like [integer, real, ID, boolean, etc], resolves typedef too
        auto typeN = std::dynamic_pointer_cast<TypeNode>(typeNode);
        return globalScope->resolveType(typeN->getTypeName());
    }
}
/*
 * Eg Typedef [typeNode] [typeDefto]
 * define type def mapping in the global
 */
void SymbolTable::defineTypeDef(std::shared_ptr<TypeNode> typeNode, std::string typeDefTo, int ID) {
    if (std::dynamic_pointer_cast<TupleTypeNode>(typeNode)) {
        // tuple typenode
        auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(typeNode);

        // create a Type object with that hashed Name since the tuple TypeNode that is going to map to should be unique
        std::shared_ptr<AdvanceType> typedefType = std::make_shared<AdvanceType>("tuple" + typeDefTo + std::to_string(ID));  // string hash
        typedefType->baseTypeEnum = TYPE::TUPLE;   // set type to TUPLE

        auto tupleChild = tupleNode->innerTypes;  // vector of inner types
        for (auto c: tupleChild) {
            // resolve children's type
            auto resolvedChild = resolveTypeUser(std::dynamic_pointer_cast<TypeNode>(c.second));
            if (resolvedChild == nullptr) throw TypeError(typeNode->loc(), "cannot resolve type in tuple");
            typedefType->tupleChildType.push_back(std::make_pair(c.first ,resolvedChild));  // recursively resolve type
        }
        typedefType->typDefName = typeDefTo;
        globalScope->defineType(typedefType);
    } else if (std::dynamic_pointer_cast<VectorTypeNode>(typeNode)) {
        auto vNode = std::dynamic_pointer_cast<VectorTypeNode>(typeNode);
        auto innerType  = resolveTypeUser(vNode->getInnerType());


        auto typeDefType = std::make_shared<AdvanceType>("vector" + typeDefTo + std::to_string(ID));
        typeDefType->typDefName = typeDefTo;
        typeDefType->baseTypeEnum = innerType->baseTypeEnum;
        typeDefType->vectorOrMatrixEnum = VECTOR;

        auto c1 = std::make_shared<AdvanceType>(innerType->getBaseTypeEnumName());  // create a inner child
        c1->baseTypeEnum = innerType->baseTypeEnum;
        c1->vectorOrMatrixEnum = NONE;

        typeDefType->vectorInnerTypes.push_back(c1);
        globalScope->defineType(typeDefType);

    } else if (std::dynamic_pointer_cast<MatrixTypeNode>(typeNode)) {
        auto vNode = std::dynamic_pointer_cast<MatrixTypeNode>(typeNode);
        auto innerType  = resolveTypeUser(vNode->getInnerType());

        auto typeDefType = std::make_shared<AdvanceType>("matrix" + typeDefTo + std::to_string(ID));
        typeDefType->typDefName = typeDefTo;
        typeDefType->baseTypeEnum = innerType->baseTypeEnum;
        typeDefType->vectorOrMatrixEnum = VECTOR;

        auto c = std::make_shared<AdvanceType>(innerType->getBaseTypeEnumName());  // create a child type
        c->baseTypeEnum = innerType->baseTypeEnum;
        c->vectorOrMatrixEnum = VECTOR;

        auto c1 = std::make_shared<AdvanceType>(innerType->getBaseTypeEnumName());  // create a second child
        c1->baseTypeEnum = innerType->baseTypeEnum;
        c1->vectorOrMatrixEnum = NONE;

        c->vectorInnerTypes.push_back(c1);

        typeDefType->vectorInnerTypes.push_back(c);   // typeDefType node now a vector that contains vector
        globalScope->defineType(typeDefType);

    } else if (std::dynamic_pointer_cast<StringTypeNode>(typeNode)) {

    } else {
        // basic type [integer, real, boolean, ID, etc]
        /*
         */
        globalScope->defineType(std::make_shared<AdvanceType>(typeNode->getTypeName(), typeDefTo));
    }
}

int SymbolTable::isTypeDefed(std::string typedefTo) {
    if (globalScope->typedefTypeNode.find(typedefTo) != globalScope->typedefTypeNode.end()) {
        return  1;
    }
    return 0;
}
/*
 * tt :   AdvancedType(name="tuple");
 * tuple: tuple
 */