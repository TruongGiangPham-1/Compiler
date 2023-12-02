#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
#include "run_time_errors.h"

// memory stuff in here.
#include "runtimeMemory.c"
// all operations in here
#include "runtimeOperations.c"
// stream stuff in here
#include "runtimeStream.c"
#include <stdio.h>

//#define DEBUGTUPLE
//#define DEBUGTYPES
//#define DEBUGMEMORY
//#define DEBUGPRINT


// STDLIB EXAMPLE: silly function
// TODO: delete once we have a proper stdlib
commonType* __silly(commonType* toPrint) {
    printf("Silly called with ");
    printCommonType(toPrint);
    printf("\n");
    return toPrint;
}
