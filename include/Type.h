#pragma once

#include <string>
#include "Types/TYPES.h"

class Type {
public:
    TYPE typeEnum;

    Type() {};
    virtual std::string getName() = 0;
    virtual ~Type() {}
};

