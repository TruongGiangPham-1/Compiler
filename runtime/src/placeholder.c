#include "Operands/BINOP.h"
#include "BuiltinTypes/BuiltInTypes.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

//#define DEBUGTUPLE
//#define DEBUGPROMOTION
//#define DEBUGMEMORY
typedef struct vecStruct {
  int* base;
  int sizeOf;
} vecStruct;

typedef struct commonType {
  enum BuiltIn type; 
  void* value; 
} commonType;

typedef struct tuple {
  int size;
  int currentSize;
  commonType** values; // list of values
} tuple;


commonType* allocateCommonType(void* value, enum BuiltIn type);
void deallocateCommonType(commonType* object);

void printType(commonType *type, bool nl) {
  switch (type->type) {
    case INT:
      printf("%d", *(int*)type->value);
      break;
    case CHAR:
      printf("%c", *(char*)type->value);
      break;
    case BOOL:
      printf("%b", *(bool*)type->value);
      break;
    case REAL:
      printf("%f", *(float*)type->value);
      break;
    case TUPLE:
      // {} bc we can't declare variables in switch
      {
        tuple *mTuple = (*(tuple**)type->value);
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

void printCommonType(commonType *type) {
  printType(type, true);
}

/**
 * Big switch case that I didn't want in the allocate common type function
 */
void extractAndAssignValue(void* value, commonType *dest) {
  switch (dest->type) {
    case INT:
      {
        int* newIntVal = malloc(sizeof(int));
        *newIntVal = *(int*)value;
        dest->value = newIntVal;
      }
      break;
    case BOOL:
      {
        bool* newBoolVal = malloc(sizeof(bool));
        *newBoolVal = *(bool*)value;
        dest->value = newBoolVal;
      }
      break;
    case REAL:
      {
        float* newFloatVal = malloc(sizeof(float));
        *newFloatVal = *(float*)value;
        dest->value = newFloatVal;
      }
      break;
    case TUPLE:
      {
        dest->value = value;
      }
      break;
    case CHAR: 
      {
        char* newCharVal = malloc(sizeof(char));
        *newCharVal = *(char*)value;
        dest->value = newCharVal;
      }
      break;
  }
}

commonType* allocateCommonType(void* value, enum BuiltIn type) {
  commonType* newType = (commonType*)malloc(sizeof(commonType));
  newType->type = type;
  extractAndAssignValue(value, newType);

#ifdef DEBUGMEMORY
  printf("Allocated common type: %p\n",newType);
#endif

  return newType;
}

void deallocateTuple(tuple* tuple) {
#ifdef DEBUGMEMORY
  printf("Deallocating Tuple at %p...\n", tuple);
  printf("=== TUPLE ===\n");
#endif /* ifdef DEBUGMEMORY */

    for (int i = 0; i < tuple->currentSize ; i++) {
      deallocateCommonType(tuple->values[i]);
    }
    free(tuple->values);
    free(tuple);

#ifdef DEBUGMEMORY
  printf("=== TUPLE ===\n");
  printf("Tuple deallocation success!\n");
#endif
}

void deallocateCommonType(commonType* object) {
  // we keep track of more object than we allocate
  // (we don't take some branches, won't initialize variables)
#ifdef DEBUGMEMORY
  printf("Deallocating commonType at %p...\n", object);
#endif /* ifdef DEBUGMEMORY */
  if (object != NULL) {
    switch (object->type) {
      case TUPLE:
      deallocateTuple(*(tuple**)object->value);
      break;
      default:
      free(object->value);
    }
  }

#ifdef DEBUGMEMORY
  printf("Commontype deallocation success!\n");
#endif /* ifdef DEBUGMEMORY */
}

tuple* allocateTuple(int size) {
  tuple* newTuple = (tuple*) malloc(sizeof(tuple));
  commonType** valueList = (commonType**) calloc(size*2, sizeof(commonType*));

  newTuple->size = size;
  newTuple->currentSize = 0;
  newTuple->values= valueList;

#ifdef DEBUGMEMORY
  printf("Allocated tuple at %p\n", newTuple);
  printf("Tuple list beings at %p\n", valueList);
#endif
  
  return newTuple;
};

/**
 * Add item to our tuple, we can potentially go over bounds....
 * but we shoulnd't due to spec, right?
 */
void appendTuple(tuple* tuple, commonType *value) {
#ifdef DEBUGTUPLE
  printf("====== Appending to tuple\n");
  printf("Tuple currently holding %p  at index %d address %p\n", tuple->values[tuple->currentSize], tuple->currentSize, &tuple->values[tuple->currentSize]);
#endif

  // dont want the real thing, make copy
  tuple->values[tuple->currentSize] = allocateCommonType(value->value, value->type);

#ifdef DEBUGTUPLE
  printf("appended to tuple at %p, %p\n", &tuple->values[tuple->currentSize], value);
  printf("Tuple now holding %p  at index %d address %p\n", tuple->values[tuple->currentSize], tuple->currentSize, &tuple->values[tuple->currentSize]);
  printf("====== Append complete\n"); 
#endif /* ifdef DEBUGTUPLE */

  tuple->currentSize++;
}

commonType* boolPromotion(bool fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from int\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
  default:
  return allocateCommonType(&fromValue, BOOL);
  }
}

commonType* intPromotion(int fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Promotion from int\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
    case REAL:
#ifdef DEBUGPROMOTION
  printf("To real\n");
#endif /* ifdef DEBUGPROMOTION */
    {
      float newfloat = (float)fromValue;
      return allocateCommonType((void*)&newfloat, REAL);
    }

    default:
    return allocateCommonType(&fromValue, INT);
  }
}

commonType* charPromotion(char fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Promotion from char\n");
#endif /* ifdef DEBUGPROMOTION */

    switch (toType) {
    default:
#ifdef DEBUGPROMOTION
  printf("To char\n");
#endif /* ifdef DEBUGPROMOTION */
    return allocateCommonType(&fromValue, CHAR);
  }
}

commonType* realPromotion(float fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Promotion from real\n");
#endif /* ifdef DEBUGPROMOTION */
    switch (toType) {
#ifdef DEBUGPROMOTION
  printf("To real\n");
#endif /* ifdef DEBUGPROMOTION */
    default:
    return allocateCommonType(&fromValue, REAL);
  }
}

// promote and return temporary
commonType* promotion(commonType* from, commonType* to) {
  switch (from->type) {
    case BOOL:
    return boolPromotion(*(bool*)from->value, to->type);
    case INT:
    return intPromotion(*(int*)from->value, to->type);
    case CHAR:
    return charPromotion(*(char*)from->value, to->type);
    case TUPLE:
    // don't think we need this 
    break;
    case REAL:
    return realPromotion(*(float*)from->value, to->type);
  }
}

bool boolBINOP(bool l, bool r, enum BINOP op) {
  switch (op) {
    case ADD:
    return l + r;
    case SUB:
    return l - r;
    case MULT:
    return l * r;
    case DIV:
    return l/r;
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case GTHAN:
    return r > l;
  }
}

int intBINOP(int l, int r, enum BINOP op) {
  switch (op) {
    case ADD:
    return l + r;
    case SUB:
    return l - r;
    case MULT:
    return l * r;
    case DIV:
    return l/r;
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case GTHAN:
    return r > l;
  }
}

float realBINOP(float l, float r, enum BINOP op) {{}
  switch (op) {
    case ADD:
    return l + r;
    case SUB:
    return l - r;
    case MULT:
    return l * r;
    case DIV:
    return l/r;
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case GTHAN:
    return r > l;
  }
}

char charBINOP(char l, char r, enum BINOP op) {
  switch (op) {
    case ADD:
    return l + r;
    case SUB:
    return l - r;
    case MULT:
    return l * r;
    case DIV:
    return l/r;
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case GTHAN:
    return r > l;
  }
}

tuple tupleBINOP(tuple* l, tuple* r, enum BINOP op) {

}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op) {
  commonType* promotedLeft;
  commonType* promotedRight;
  promotedLeft = promotion(left,right);
  promotedRight = promotion(right,left);

  printCommonType(promotedRight);
  printCommonType(promotedLeft);

  commonType* result;
  // arbitrary, after promo they are the same. if they are not, there is something wrong
  // I tried to do a switch chain like before but the scoping was messed up.
  if(promotedLeft->type == BOOL) {

    bool tempBool = intBINOP(*(bool*)promotedLeft->value, *(bool*)promotedRight->value, op);
    result = allocateCommonType(&tempBool, BOOL);

  } else if (promotedLeft->type == REAL) {

    float tempFloat = realBINOP(*(float*)promotedLeft->value, *(float*)promotedRight->value, op);
    result = allocateCommonType(&tempFloat, REAL);

  } else if (promotedLeft->type == INT) {

    int tempInt = intBINOP(*(int*)promotedLeft->value, *(int*)promotedRight->value, op);
    result = allocateCommonType(&tempInt, INT);

  } else if (promotedLeft->type == CHAR) {

    char tempChar = charBINOP(*(char*)promotedLeft->value, *(char*)promotedRight->value, op);
    result = allocateCommonType(&tempChar, CHAR);

  } else if (promotedLeft->type == TUPLE) {

    tuple tempTuple = tupleBINOP(*(tuple**)promotedLeft->value, *(tuple**)promotedRight->value, op);
    result = allocateCommonType(&tempTuple, TUPLE);

  }
  
  // temporary operands
#ifdef DEBUGMEMORY
  printf("=== de allocating temporary operands...\n");
#endif /* ifdef DEBUGMEMORY */

  deallocateCommonType(promotedLeft);
  deallocateCommonType(promotedRight);

#ifdef DEBUGMEMORY
  printf("=== complete\n");
#endif /* ifdef DEBUGMEMORY */

  return result;
}
