#pragma once
#include "ASTNode.h"

// Decl nodes are very similar to Assign nodes, but with more stuff
// Children: [ TypeNode, ExprNode ]
class DeclNode : public ASTNode {
public:
    std::shared_ptr<Symbol> sym;

    DeclNode(int line, std::shared_ptr<Symbol> sym);

    // the full Symbol class of the ID being declared
    std::shared_ptr<Symbol> getID();
    // just the name of the ID being declared
    std::string getIDName();

    std::shared_ptr<ASTNode> getTypeNode();
    std::shared_ptr<ASTNode> getExprNode();

    std::string toString() override;
};