#ifndef GAZPREABASE_RUNTIME_INCLUDE_RUN_TIME_ERRORS_H_
#define GAZPREABASE_RUNTIME_INCLUDE_RUN_TIME_ERRORS_H_

#include <stdio.h>
#include <stdlib.h>

#define DEF_RUN_TIME_ERROR(NAME)                      \
void NAME(const char *description) {                  \
    fprintf(stderr, "%s: %s \n", #NAME, description); \
    exit(1);                                          \
}

DEF_RUN_TIME_ERROR(IndexError)

DEF_RUN_TIME_ERROR(MathError)

DEF_RUN_TIME_ERROR(SizeError)

DEF_RUN_TIME_ERROR(StrideError)

#endif // GAZPREABASE_RUNTIME_INCLUDE_RUN_TIME_ERRORS_H_
