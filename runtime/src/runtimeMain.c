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
#include <errno.h>
#include <limits.h>


//#define DEBUGTUPLE
//#define DEBUGTYPES
//#define DEBUGMEMORY
//#define DEBUGPRINT
 #define DEBUGSTREAM

// global variable streamBuffer for streamIn
// this is a buffer for the rewind feature
const int MAX_REWIND_BUFFER_SIZE = 1024;
char STREAM_REWIND_BUFFER[MAX_REWIND_BUFFER_SIZE] = {0};
int BUF_HEAD = 0;
int BUF_TAIL = 0;


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
    // tuple is just for debug
    // we don't disambiguate. similar behavior
    case VECTOR:
    case MATRIX:
    case STRING:
      {
        list* mListable = ((list*)type->value);

        if (type->type != STRING) printf("[");

        for (int i = 0 ; i < mListable->currentSize; i++) {

          printType(mListable->values[i], false);
          if (i != mListable->currentSize-1 && type->type != STRING) printf(" ");
        }
        if (type->type != STRING) printf("]");
      }
      break;
    default:
    UnsupportedTypeError("Attempting to print a type not recognized by the backend (this happens when type enums are bad)");
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

enum StreamState {
  STREAM_STATE_OK = 0,
  STREAM_STATE_ERR = 1,
  STREAM_STATE_EOF = 2,
};

void setStreamState(int* state, int newState, commonType *type) {
#ifdef DEBUGSTREAM
  printf("Setting streamState to %d\n", newState);
#endif /* ifdef DEBUGSTREAM */

    // given the streamState error and the type, set the streamState
    if (type->type == CHAR) {
        // the only possible error for a char is EOF, where we set streamState to 0
        *state = 0;
        return;
    } else {
        // in all other cases, set the state to the integer value of the streamStateErr
        *state = newState;
    }
}


void streamIn(commonType *type, int* streamState) {
  // to handle scanf errors, we check the return value
  // the return value of scanf is the number of validly converted args
  // https://stackoverflow.com/a/5969152
  int check = 0;

  // to implement the rewind buffer, we might want to make this a global var
  // but the rewind buffer is another problem!
  char buffer[1024] = {0}; // how big do I make this?

  switch (type->type) {
    case INTEGER: {
        check = scanf("%s", buffer);
        // convert string to an int
        // https://stackoverflow.com/a/18544436
        long lnum;
        char *end;
        errno = 0;

#ifdef DEBUGSTREAM
        printf("buffer: '%s'\n", buffer);
#endif /* ifdef DEBUGSTREAM */

        lnum = strtol(buffer, &end, 10);        //10 specifies base-10
        if (end == buffer) {
            // no digits consumed
#ifdef DEBUGSTREAM
            printf("ERROR: no digits were found\n");
#endif /* ifdef DEBUGSTREAM */
            check = 0;
        } else if (*end != '\0') {
            // extra characters at the end
#ifdef DEBUGSTREAM
            printf("ERROR: extra characters at the end\n");
#endif /* ifdef DEBUGSTREAM */
            check = 0;
        } else if (((lnum == LONG_MAX || lnum == LONG_MIN) && errno == ERANGE) ||
                   (lnum > INT_MAX) || (lnum < INT_MIN)) {
            // number is out of range
#ifdef DEBUGSTREAM
            printf("ERROR: input out of range");
#endif /* ifdef DEBUGSTREAM */
            check = 0;
        } else {
#ifdef DEBUGSTREAM
            printf("Successful int read: %d\n", (int) lnum);
#endif /* ifdef DEBUGSTREAM */

            // number is valid
            check = 1;
            *(int *) type->value = (int) lnum;
        }
        break;
    }
    case CHAR: {
        // CHAR CAN NEVER FAIL (except if it's an end of file)
        char c;
        check = scanf("%c", &c);
        *(char*)type->value = c;

        if (c == '\0') {
            // on mac, an EOF won't be detected by scanf as a -1
            check = -1;
        }

#ifdef DEBUGSTREAM
        printf("Scanned char: '%c'\n", *(char*)type->value);
#endif /* ifdef DEBUGSTREAM */
        break;
    }
    case BOOLEAN: {
//      printf("Enter a boolean value (T/F): ");
      // scan char. If it's T, true, else false
      check = scanf("%s", buffer);
#ifdef DEBUGSTREAM
      printf("buffer: '%s'\n", buffer);
#endif /* ifdef DEBUGSTREAM */
      if (strcmp(buffer, "T") == 0) {
        *(bool*)type->value = true;
      } else if (strcmp(buffer, "F") == 0) {
        *(bool *) type->value = false;
      } else {
          check = 0;
      }
      break;
    }
    case REAL:
//      printf("Enter a real: ");
//      check = scanf("%f", buffer);
//      *(float*)type->value = atof(buffer);
      check = scanf("%f", (float*)type->value);
      break;
  }

#ifdef DEBUGSTREAM
  printf("scanf check is %d\n", check);
  printf("Buffer is '%s'\n", buffer);
#endif /* ifdef DEBUGSTREAM */

  // check if the scanf failed
  // i used to check if check == 0, but that doesn't work on linux for EOF
  // on EOF, scanf returns -1 on linux, but 0 on mac
  if (check != 1) {
    // invalid input! (scanf didn't convert anything)
    // set the val to "default"
    setToNullValue(type);

    // check end of file (for char)
    // https://stackoverflow.com/a/1428924
    if (feof(stdin)) {
#ifdef DEBUGSTREAM
        printf("EOF encountered\n");
#endif /* ifdef DEBUGSTREAM */

      if (type->type == CHAR) {
          // if a char encounters a streamState, set to -1
          // val is now -1
          *(char *) type->value = -1;
          return;
      }

      // set new streamState to EOF (2)
      setStreamState(streamState, STREAM_STATE_EOF, type);
    } else {
      // in all other cases, set to err state (1)
      setStreamState(streamState, STREAM_STATE_ERR, type);
    }

  } else {
    // valid input! reset stream_state to 0 (it might have been 1 before)
    setStreamState(streamState, STREAM_STATE_OK, type);
  }
}

// STDLIB EXAMPLE: silly function
// TODO: delete once we have a proper stdlib
commonType* __silly(commonType* toPrint) {
    printf("Silly called with ");
    printCommonType(toPrint);
    printf("\n");
    return toPrint;
}
