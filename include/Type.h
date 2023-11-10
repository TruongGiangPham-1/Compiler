#pragma once

#include <string>
#include "Types/TYPES.h"
#include <vector>
class Type {
public:
    TYPE baseTypeEnum;  // [integer, real, character, boolean, string, tuple]
    TYPE vectorOrMatrixEnum = TYPE::NONE;  // [vector, matrix]  // indicate if the type is a matrix or vector or none

    // yikes, need to check if the type is TYPE::tuple then this vector will hav size when resolvd
    std::vector<std::pair<std::string, std::shared_ptr<Type>>> tupleChildType;

    Type() {};
    virtual std::string getName() = 0;
    virtual ~Type() {}
};

