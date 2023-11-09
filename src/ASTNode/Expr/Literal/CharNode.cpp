#include "ASTNode/Expr/Literal/CharNode.h"

std::string CharNode::toString() {
    return "Char Node";
}

char CharNode::getVal() {
    return val;
}

std::optional<char> CharNode::parseEscape(char c) {
    switch (c) {
        case '0':
            return '\0';
        case 'a':
            return '\a';
        case 'b':
            return '\b';
        case 't':
            return '\t';
        case 'n':
            return '\n';
        case 'r':
            return '\r';
        case '\\':
            return '\\';
        case '\'':
            return '\'';
        case '\"':
            return '\"';
        default:
            return std::nullopt;
    }
}