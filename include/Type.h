#pragma once

#include <string>

class Type {
public:
    virtual std::string getName() = 0;
    virtual ~Type() {}
};
