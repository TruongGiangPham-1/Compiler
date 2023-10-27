#include <sstream>

#include "AST.h"

AST::AST() : token(nullptr), scope(nullptr) {}

AST::AST(antlr4::Token* token)
        : token(new antlr4::CommonToken(token)), scope(nullptr) {}

AST::AST(size_t tokenType) {
    token = new antlr4::CommonToken(tokenType);
    scope = nullptr;
}

size_t AST::getNodeType() { return token->getType(); }

void AST::addChild(std::any t) {
    this->addChild(std::any_cast<std::shared_ptr<AST>>(t)); // There is only one valid type for t. Pass it to AST::addChild(AST* t)
}

void AST::addChild(std::shared_ptr<AST> t) { children.push_back(t); }

bool AST::isNil() { return token == nullptr; }

std::string AST::toString() { return token != nullptr ? token->getText() : "nil"; }

std::string AST::toStringTree() {
    if ( children.empty() ) return toString();
    std::stringstream buf;
    if ( !isNil() ) {
        buf << '(' << toString() << ' ';
    }
    for (auto iter = children.begin(); iter != children.end(); iter++) {
        std::shared_ptr<AST> t = *iter; // normalized (unnamed) children
        if ( iter != children.begin() ) buf << ' ';
        buf << t->toStringTree();
    }
    if ( !isNil() ) buf << ')';
    return buf.str();
}

size_t AST::loc() {
    return (token) ? token->getLine() : -1;
}

AST::~AST() {}
