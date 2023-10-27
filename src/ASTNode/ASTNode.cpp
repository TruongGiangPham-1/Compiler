#include <sstream>

#include "ASTNode/ASTNode.h"

ASTNode::ASTNode() : token(nullptr), scope(nullptr) {}

ASTNode::ASTNode(antlr4::Token* token)
        : token(new antlr4::CommonToken(token)), scope(nullptr) {}

ASTNode::ASTNode(size_t tokenType) {
    token = new antlr4::CommonToken(tokenType);
    scope = nullptr;
}

size_t ASTNode::getNodeType() { return token->getType(); }

void ASTNode::addChild(std::any t) {
    this->addChild(std::any_cast<std::shared_ptr<ASTNode>>(t)); // There is only one valid type for t. Pass it to ASTNode::addChild(ASTNode* t)
}

void ASTNode::addChild(std::shared_ptr<ASTNode> t) { children.push_back(t); }

bool ASTNode::isNil() { return token == nullptr; }

std::string ASTNode::toString() { return token != nullptr ? token->getText() : "nil"; }

std::string ASTNode::toStringTree() {
    if ( children.empty() ) return toString();
    std::stringstream buf;
    if ( !isNil() ) {
        buf << '(' << toString() << ' ';
    }
    for (auto iter = children.begin(); iter != children.end(); iter++) {
        std::shared_ptr<ASTNode> t = *iter; // normalized (unnamed) children
        if ( iter != children.begin() ) buf << ' ';
        buf << t->toStringTree();
    }
    if ( !isNil() ) buf << ')';
    return buf.str();
}

size_t ASTNode::loc() {
    return (token) ? token->getLine() : -1;
}

ASTNode::~ASTNode() {}
