#include <sstream>

#include "ASTNode/ASTNode.h"

ASTNode::ASTNode() {
    scope = nullptr;
    line = -1;
}

ASTNode::ASTNode(int line) {
    scope = nullptr;
    this->line = line;
}

void ASTNode::addChild(std::any t) {
    this->addChild(std::any_cast<std::shared_ptr<ASTNode>>(t)); // There is only one valid type for t. Pass it to ASTNode::addChild(ASTNode* t)
}

void ASTNode::addChild(std::shared_ptr<ASTNode> t) { children.push_back(t); }

std::string ASTNode::toString() { return "ASTNode"; }

std::string ASTNode::toStringTree() {
    if ( children.empty() ) return toString();
    std::stringstream buf;
        buf << '(' << toString() << ' ';
    for (auto iter = children.begin(); iter != children.end(); iter++) {
        std::shared_ptr<ASTNode> t = *iter; // normalized (unnamed) children
        if ( iter != children.begin() ) buf << ' ';
        buf << t->toStringTree();
    }
    buf << ')';
    return buf.str();
}

size_t ASTNode::loc() {
    return line;
}

ASTNode::~ASTNode() {}
