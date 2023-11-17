#pragma once

// different types for our TypeNode
// mainly just for the Typecheck passes to figure out which type we can cast to

// had to do class because declaration conflicted with BuiltInTypes.h's enum
enum TYPE{
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
    NONE,

    IDENTITY,
    NULL_  // underline because null is a reserved keyword
};