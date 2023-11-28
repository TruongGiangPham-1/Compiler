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

/**
 * given a string, return the first char and the rest of the string (if valid)
 *
 * this also handles escape chars
 *
 * Throws an exception if the char is invalid. Catch it to generate the SyntaxError.
 * @param s
 * @return
 */
std::pair<char, std::string> CharNode::consumeChar(std::string s) {
    if (s[0] == '\\') {
        // we should never get to this point since our grammar handles it,
        // but this is just for a peace of mind
        if (s.length() == 1) throw std::runtime_error("Escape symbol \\ must be followed by a char");

        auto escapedChar = CharNode::parseEscape(s[1]);
        if (escapedChar.has_value()) {
            return std::make_pair(escapedChar.value(), s.substr(2));
        } else {
            throw std::runtime_error("Invalid escape char " + s.substr(0, 2));
        }
    } else {
        // normal char
        return std::make_pair(s[0], s.substr(1));
    }
}