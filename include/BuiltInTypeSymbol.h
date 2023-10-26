/**
* Excerpted from CMPUT 415 Cymbol code
*/

#pragma once

#include "Symbol.h"
#include "Type.h"

class BuiltInTypeSymbol : public Symbol, public Type {
public:
    BuiltInTypeSymbol(std::string name) : Symbol(name) {}

    std::string getName() {
        return Symbol::getName();
    }
};
