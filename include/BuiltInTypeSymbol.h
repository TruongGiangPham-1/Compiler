/**
* Excerpted from CMPUT 415 Cymbol code
*/

#pragma once

#include "Symbol.h"
#include "Type.h"
#include "BuiltinTypes/BuiltInTypes.h"

class BuiltInTypeSymbol : public Symbol, public Type {
public:
    BuiltInTypeSymbol(std::string name) : Symbol(name) {}

    std::string getName() {
        return Symbol::getName();
    }

    BuiltIn toBuiltIn() { return stringToBuiltIn(getName());}
    BuiltIn stringToBuiltIn(std::string name) {
      if (name == "char") {
        return BuiltIn::CHAR;
      } else if (name == "int") {
        return BuiltIn::INT;
      } else if (name == "real") {
        return BuiltIn::REAL;
      } else if (name == "bool") {
        return BuiltIn::BOOL;
      } else if (name == "tuple") {
        return BuiltIn::TUPLE;
      } else {
        throw std::runtime_error("unknown");
    }
}
};
