//
// Created by truong on 01/11/23.
//

#ifndef GAZPREABASE_ERRORLISTENER_H
#define GAZPREABASE_ERRORLISTENER_H

#include "../CompileTimeExceptions.h"
#include "antlr4-runtime.h"

class ErrorListener : public antlr4::BaseErrorListener {
    void syntaxError(antlr4::Recognizer *recognizer, antlr4::Token *offendingSymbol,
                     size_t line, size_t charPositionInLine, const std::string &msg,
                     std::exception_ptr e) override {
        std::vector<std::string> rule_stack = ((antlr4::Parser *) recognizer)->getRuleInvocationStack();
        // The rule_stack may be used for determining what rule and context the error␣ has occurred in.
        // You may want to print the stack along with the error message, or use the␣ stack contents to
        // make a more detailed error message.


        // reverse the grammar stack
        std::reverse(rule_stack.begin(), rule_stack.end());
        std::cout << "rule stack: [";
        for (auto str: rule_stack) {
            std::cout << " " << str;
        }
        std::cout <<  "]\n";
        //TODO: what else to print?
        throw SyntaxError(line,
                          msg); // Throw antlr syntax error and crash program
    };
};
#endif //GAZPREABASE_ERRORLISTENER_H
