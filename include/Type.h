#pragma once

#include "Types/TYPES.h"
#include <string>
#include <vector>
#include <memory>
class Type {
public:
    TYPE baseTypeEnum = TYPE::NONE; // [integer, real, character, boolean, string, tuple,   identity, null]
    TYPE vectorOrMatrixEnum = TYPE::NONE; // [vector, matrix]  // indicate if the type is a matrix or vector or none
    std::vector<std::string> typeString { "integer", "real", "boolean", "character", "string", "vector", "matrix", "tuple", "none", "identity", "null" };
    bool isString = false;
    // yikes, need to check if the type is TYPE::tuple then this vector will hav size when resolvd
    std::vector<std::pair<std::string, std::shared_ptr<Type>>> tupleChildType;

    // vector type children
    std::vector<std::shared_ptr<Type>> vectorInnerTypes;

    std::vector<int> dims; // maybe can populate this in the backend?

    Type() {};
    virtual std::string getName() = 0; // getname returnn Symbol::getname() when type is custome user type, used during typedef
    // I know its confusing , but getbaseTypeEnumName() returns the string form of TYPES.h enum
    virtual std::string getBaseTypeEnumName() = 0;
    virtual void setName(std::string name) = 0;
    virtual ~Type() { }
};
