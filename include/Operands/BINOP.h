#pragma once
enum BINOP {
    // math
    ADD,
    SUB,
    MULT,
    DIV,
    REM, // %
    EXP, // ^
    // cmp
    EQUAL,
    NEQUAL,
    GTHAN,
    LTHAN,
    LEQ,
    GEQ,
    // vectors, strings
    CONCAT,
    STRIDE,
    // vectors+ matrix
    DOT_PROD,
    // boolean
    AND,
    OR,
    XOR
};
