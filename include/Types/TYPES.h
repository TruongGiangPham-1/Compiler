#pragma once

#include <map>
#include <string>

// different types for our TypeNode
// mainly just for the Typecheck passes to figure out which type we can cast to

// had to do class because declaration conflicted with BuiltInTypes.h's enum
enum class TYPE{
    // builtins
    INTEGER,
    REAL,
    BOOLEAN,
    CHAR,
    // advanced
    STRING,
    VECTOR,
    MATRIX,
    TUPLE,
    NONE
};

inline std::map<TYPE, std::string> type_to_string = {
    {TYPE::INTEGER, "integer"},
    {TYPE::REAL, "real"},
    {TYPE::BOOLEAN, "boolean"},
    {TYPE::CHAR, "char"},
    {TYPE::STRING, "string"},
    {TYPE::VECTOR, "vector"},
    {TYPE::MATRIX, "matrix"},
    {TYPE::TUPLE, "tuple"},
    {TYPE::NONE, "none"}
};