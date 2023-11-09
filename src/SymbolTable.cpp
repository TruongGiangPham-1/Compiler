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
        // TODO: FORGOT VECTOR IS NOT IN PART 1,
        // resolve innertyp
        auto typeN = std::dynamic_pointer_cast<VectorTypeNode>(typeNode);
#ifdef DEBUG
        std::cout << "Resolve Type: " << typeN->getTypeName() << std::endl;
#endif
        auto innerTypeN = std::dynamic_pointer_cast<TypeNode>(typeN->innerType);
        assert(typeN);
        auto innerTypeRes = globalScope->resolveType(innerTypeN->getTypeName());
        if (innerTypeRes == nullptr) throw (typeNode->loc(), "cannot resolve innner type " + innerTypeN->getTypeName());

        // create a new type object to set the advancedTYPE to TYPE::VECTOR
        return innerTypeRes;

    } else if (std::dynamic_pointer_cast<MatrixTypeNode>(typeNode)) {

    } else if (std::dynamic_pointer_cast<StringTypeNode>(typeNode)){

    } else if (std::dynamic_pointer_cast<TupleTypeNode>(typeNode)) {
        // tuple typenode
        auto tupleNode = std::dynamic_pointer_cast<TupleTypeNode>(typeNode);

        std::shared_ptr<Type> resolvedType = std::make_shared<AdvanceType>("tuple");
        resolvedType->baseTypeEnum = TYPE::TUPLE;

        auto tupleChild = tupleNode->getTypes();  // vector of inner types
        for (auto c: tupleChild) {
            // recursively resolve children's type. resolves typedef too
            auto resolvedChild = resolveTypeUser(c);
            if (resolvedChild == nullptr) throw TypeError(typeNode->loc(), "cannot resolve type in tuple");
            resolvedType->tupleChildType.push_back(resolvedChild);
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

        auto tupleChild = tupleNode->getTypes();  // vector of inner types
        for (auto c: tupleChild) {
            // resolve children's type
            auto resolvedChild = resolveTypeUser(c);
            if (resolvedChild == nullptr) throw TypeError(typeNode->loc(), "cannot resolve type in tuple");
            typedefType->tupleChildType.push_back(resolvedChild);  // recursively resolve type
        }
        typedefType->typDefName = typeDefTo;
        globalScope->defineType(typedefType);
    } else if (std::dynamic_pointer_cast<VectorTypeNode>(typeNode)) {

    } else if (std::dynamic_pointer_cast<MatrixTypeNode>(typeNode)) {

    } else if (std::dynamic_pointer_cast<StringTypeNode>(typeNode)) {

    } else {
        // basic type [integer, real, boolean, ID, etc]
        /*
         */
        globalScope->defineType(std::make_shared<AdvanceType>(typeNode->getTypeName(), typeDefTo));
    }
}
/*
 * tt :   AdvancedType(name="tuple");
 * tuple: tuple
 */