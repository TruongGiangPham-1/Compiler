#pragma once

#include <memory>
#include <vector>

#include "CompileTimeExceptions.h"
#include "BaseScope.h"
#include "ASTNode/Type/TypeNode.h"

class SymbolTable {
private:
    std::vector<std::shared_ptr<Scope>> scopes;
public:
    SymbolTable() {}
    std::shared_ptr<GlobalScope> globalScope;
    std::shared_ptr<Scope> enterScope(std::string& name, const std::shared_ptr<Scope>& currentScope);
    std::shared_ptr<Scope> enterScope(std::shared_ptr<Scope> newScope);

    //std::pair<TYPE, std::string>  resolveType(std::shared_ptr<ASTNode> typeNode);
    std::shared_ptr<Type>  resolveTypeUser(std::shared_ptr<ASTNode> typeNode);

    /*
     * [if typeNode is a tupleTYpe, we create the Tuple type object, and define this custom type in GlobalScope]
     *
     * [else: typeNode is a baseType, we just create regular mapping]:
     *
     * eg: Typedef integer int;
     * creates mapping in GlobalScope.userType dictionary
     *    {int, integer}
     * eg: Typedef tuple(integer, real, boolean) tt;
     * creates the 'tuple(integer, real, boolean)' Advance type object and create this mapping in the global scope
     *   {tt, tuple(integer, real, boolean) obj}
     *   *resolving 'tt' will give back the tuple object
     *
     * @param: ID: used to create a unique type name for the tuple Type created in typedef, used Def::getID()
     */
    void defineTypeDef(std::shared_ptr<TypeNode> typeNode, std::string typeDefTo, int ID);  // define using typenode

    std::shared_ptr<Scope> exitScope(std::shared_ptr<Scope> currentScope) {
        return currentScope->getEnclosingScope();
    }


    std::string toString();
};
/*
 *
 *  userType = { 'str'   :          }
 *
 *
 */
