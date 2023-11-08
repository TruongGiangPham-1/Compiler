//
// Created by truong on 05/11/23.
//

#ifndef GAZPREABASE_ADVANCETYPE_H
#define GAZPREABASE_ADVANCETYPE_H
#include "Type.h"
#include "Symbol.h"
#include <vector>
#include "Types/TYPES.h"
#include "CompileTimeExceptions.h"

class AdvanceType : public Type, public Symbol{
public:
    /*
     * Typedef integer int;
     * name = integer
     * typedef name = int;
     */
    std::string typDefName;  // typedef  name?
    /*
     * vector of dimentions. if it is a Vector, dims,size() = 1, if it is a matrix, dims.size() = 2
     */
    std::vector<mlir::Value> dims;  // maybe can populate this in the backend?

    AdvanceType(std::string name) : Symbol(name), typDefName(name) {};
    // at Def.cpp push <integer, integer>
    AdvanceType(std::string name, std::string typeDefName) : Symbol(name), typDefName(typeDefName) {
        if (name == "integer") {
            baseTypeEnum = TYPE::INTEGER;
        } else if (name == "real") {
            baseTypeEnum = TYPE::REAL;
        } else if (name == "boolean") {
            baseTypeEnum = TYPE::BOOLEAN;
        } else if (name == "character") {
            baseTypeEnum = TYPE::CHAR;
        } else if (name == "tuple") {
            baseTypeEnum = TYPE::TUPLE;
        } else if (name == "string") {
            baseTypeEnum = TYPE::STRING;
        }else {
               // throw TypeError(0, "invalid typename when creating AdvancedType object");
               // custom user type
        }
    };

    std::string getName() {
        return Symbol::getName();
    };
    std::string getTypDefname() {
        return typDefName;
    }
};

class TupleType: public Type, public Symbol{
    // as stan and josh discussed, add maybe have vector<TypeNode>?

};

#endif //GAZPREABASE_ADVANCETYPE_H
