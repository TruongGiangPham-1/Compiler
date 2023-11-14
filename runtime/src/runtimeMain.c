#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
#include "run_time_errors.h"

// memory stuff in here.
#include "runtimeMemory.c"
// all operations in here
#include "runtimeOperations.c"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


//#define DEBUGTUPLE
//#define DEBUGTYPES
//#define DEBUGMEMORY
//#define DEBUGPRINT

void printType(commonType *type, bool nl) {
  switch (type->type) {
    case INTEGER:
      #ifdef DEBUGPRINT
      printf("\nPRINTING INTEGER\n");
      #endif /* ifdef DEBUGPRINT */
      printf("%d", *(int*)type->value);
      break;
    case CHAR:
      #ifdef DEBUGPRINT
      printf("\nPRINTING CHAR\n");
      #endif /* ifdef DEBUGPRINT */
      printf("%c", *(char*)type->value);
      break;
    case BOOLEAN:
      #ifdef DEBUGPRINT
      printf("\nPRINTING BOOL:\n");
      #endif /* ifdef DEBUGPRINT */
      printf("%s", *(bool*)type->value ? "T" : "F");
      break;
    case REAL:
      #ifdef DEBUGPRINT
      printf("\nPRINTING REAL\n");
      #endif /* ifdef DEBUGPRINT */
      printf("%g", *(float*)type->value);
      break;
    case TUPLE:
      #ifdef DEBUGPRINT
      printf("\nPRINTING TUPLE\n");
      #endif /* ifdef DEBUGPRINT */
      // {} bc we can't declare variables in switch
      {
        tuple *mTuple = ((tuple*)type->value);
        #ifdef DEBUGTUPLE
        printf("Printing tuple %p\n", mTuple);
        #endif
        printf("(");
        for (int i = 0 ; i < mTuple->size ; i++) {
          #ifdef DEBUGTUPLE
          printf("\nprinting tuple value at %p\n", &mTuple->values[i]);
          #endif
          printType(mTuple->values[i], false);
          if (i != mTuple->size-1) printf(" ");
        }
        printf(")");
      }
      break;
  }

  if (nl) printf("\n");
  return;
}

// set a commonType to its null value
void setToNullValue(commonType *type) {
  switch (type->type) {
    case INTEGER:
      *(int*)type->value = 0;
          break;
    case CHAR:
      *(char*)type->value = '\0';
          break;
    case BOOLEAN:
      *(bool*)type->value = false;
          break;
    case REAL:
      *(float*)type->value = 0.0f;
          break;
  }
}

void printCommonType(commonType *type) {
  printType(type, true);
}

void streamOut(commonType *type) {
  printType(type, false);
}

void streamIn(commonType *type) {
  // to handle scanf errors, we check the return value
  // the return value of scanf is the number of validly converted args
  // https://stackoverflow.com/a/5969152
  int check = 1;

  switch (type->type) {
    case INTEGER:
//      printf("Enter an int: ");
      check = scanf("%d", (int*)type->value);
      break;
    case CHAR:
//      printf("Enter a char: ");
      // CHAR CAN NEVER FAIL (except if it's an end of file)
      check = scanf("%c", (char*)type->value);
      break;
    case BOOLEAN: {
//      printf("Enter a boolean value (T/F): ");
      // scan char. If it's T, true, else false
      char buffer[1024]; // how big do I make this?
      check = scanf("%s", buffer);
//      printf("buffer: '%s'\n", buffer);
      if (strcmp(buffer, "T") == 0) {
        *(bool*)type->value = true;
      } else {
        // "default" boolean value is also false
        *(bool *) type->value = false;
      }
      break;
    }
    case REAL:
//      printf("Enter a real: ");
      check = scanf("%f", (float*)type->value);
      break;
  }

  if (check == 0) {
    // invalid input!

    // check end of file (for char)
    // https://stackoverflow.com/a/1428924
    if (feof(stdin) && type->type == CHAR) {
      // val is now -1
      *(char*)type->value = -1;
      return;
    }

    // in all other cases, set to the "default" value of the type
    setToNullValue(type);
  }
}
