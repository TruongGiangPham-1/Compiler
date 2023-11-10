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

DEF_RUN_TIME_ERROR(RuntimeOPError)

DEF_RUN_TIME_ERROR(PromotionError)

DEF_RUN_TIME_ERROR(CastError)

DEF_RUN_TIME_ERROR(UnsupportedTypeError)

#endif // GAZPREABASE_RUNTIME_INCLUDE_RUN_TIME_ERRORS_H_
