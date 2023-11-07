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
    TYPE typeEnum;   // maybe identify
    std::string name;  // user type name?
    std::string typDefName;  // typedef  name?
    /*
     * vector of dimentions. if it is a Vector, dims,size() = 1, if it is a matrix, dims.size() = 2
     */
    std::vector<mlir::Value> dims;  // maybe can populate this in the backend?

    AdvanceType(std::string name, std::string typeDefName) : Symbol(name), typDefName(typeDefName) {
        if (name == "integer") {
            typeEnum = TYPE::INTEGER;
        } else if (name == "real") {
            typeEnum = TYPE::REAL;
        } else if (name == "boolean") {
            typeEnum = TYPE::BOOLEAN;
        } else if (name == "character") {
            typeEnum = TYPE::CHAR;
        } else if (name == "tuple") {
            typeEnum = TYPE::TUPLE;
        } else if (name == "matrix" ) {
            typeEnum = TYPE::MATRIX;
        } else if (name == "string") {
            typeEnum = TYPE::STRING;
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

class TupleType: public AdvanceType {
    // as stan and josh discussed, add maybe have vector<TypeNode>?
};

#endif //GAZPREABASE_ADVANCETYPE_H
