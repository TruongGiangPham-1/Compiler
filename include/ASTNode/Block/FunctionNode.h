//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_FUNCTIONNODE_H
#define GAZPREABASE_FUNCTIONNODE_H

#include "BlockNode.h"
#include "Symbol.h"
#include "Type.h"
class FunctionNode : public BlockNode {
public:
    std::vector<std::shared_ptr<ASTNode>>orderedArgsID;    // array of arguments's ID node
    std::shared_ptr<Symbol> funcNameSym;
    FunctionNode(int line, std::shared_ptr<Symbol>funcNameSym);
    std::string toString() override;
    std::shared_ptr<ASTNode> getRetTypeNode();

};

class FunctionForwardNode: public  FunctionNode {
public:
    FunctionForwardNode(int line, std::shared_ptr<Symbol>funcNameSym): FunctionNode(line, funcNameSym){};
    std::string toString() override {
        return "FunctionForward";
    };
};

class FunctionSingleNode: public FunctionNode {
public:
    FunctionSingleNode(int line, std::shared_ptr<Symbol>funcNameSym): FunctionNode(line, funcNameSym){};
    std::string toString() override {
        return "FunctionSingle";
    };
};

class FunctionBlockNode: public FunctionNode {
    FunctionBlockNode(int line, std::shared_ptr<Symbol>funcNameSym): FunctionNode(line, funcNameSym){};
    std::string toString() override {
        return "FunctionBlockNode";
    };
};




#endif //GAZPREABASE_FUNCTIONNODE_H
