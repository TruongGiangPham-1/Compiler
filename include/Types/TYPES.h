#pragma once

// different types for our TypeNode
// mainly just for the Typecheck passes to figure out which type we can cast to
enum TYPE {
    // builtins
    INTEGER,
    REAL,
    BOOLEAN,
    CHAR,
    // advanced
    STRING,
    VECTOR,
    MATRIX,
};
