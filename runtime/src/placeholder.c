#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
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
  enum TYPE type; 
  void* value; 
} commonType;

typedef struct tuple {
  int size;
  int currentSize;
  commonType** values; // list of values
} tuple;


commonType* allocateCommonType(void* value, enum TYPE type);
void deallocateCommonType(commonType* object);

void printType(commonType *type, bool nl) {
  switch (type->type) {
    case INTEGER:
      printf("%d", *(int*)type->value);
      break;
    case CHAR:
      printf("%c", *(char*)type->value);
      break;
    case BOOLEAN:
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
    case INTEGER:
      {
        int* newIntVal = malloc(sizeof(int));
        *newIntVal = *(int*)value;
        dest->value = newIntVal;
      }
      break;
    case BOOLEAN:
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

commonType* allocateCommonType(void* value, enum TYPE type) {
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
  commonType** valueList = (commonType**) calloc(size, sizeof(commonType*));

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

commonType* boolPromotion(bool fromValue, enum TYPE toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from int\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
  default:
  return allocateCommonType(&fromValue, BOOLEAN);
  }
}

commonType* intPromotion(int fromValue, enum TYPE toType) {
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
    return allocateCommonType(&fromValue, INTEGER);
  }
}

commonType* charPromotion(char fromValue, enum TYPE toType) {
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

commonType* realPromotion(float fromValue, enum TYPE toType) {
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
    case BOOLEAN:
    return boolPromotion(*(bool*)from->value, to->type);
    case INTEGER:
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
    return l > r;
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
    return l > r;
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
    return l > r;
  }
}

tuple tupleBINOP(tuple* l, tuple* r, enum BINOP op) {

}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op) {
  commonType* promotedLeft;
  commonType* promotedRight;
  promotedLeft = promotion(left,right);
  promotedRight = promotion(right,left);

  commonType* result;
  // arbitrary, after promo they are the same. if they are not, there is something wrong
  // I tried to do a switch chain like before but the scoping was messed up.
  if(promotedLeft->type == BOOLEAN) {

    bool tempBool = intBINOP(*(bool*)promotedLeft->value, *(bool*)promotedRight->value, op);
    result = allocateCommonType(&tempBool, BOOLEAN);

  } else if (promotedLeft->type == REAL) {

    float tempFloat = realBINOP(*(float*)promotedLeft->value, *(float*)promotedRight->value, op);
    result = allocateCommonType(&tempFloat, REAL);

  } else if (promotedLeft->type == INTEGER) {

    int tempInt = intBINOP(*(int*)promotedLeft->value, *(int*)promotedRight->value, op);
    result = allocateCommonType(&tempInt, INTEGER);

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

bool boolUNARYOP(bool val, enum UNARYOP op) {
  // implement once we have UNARYOP::NOT
  switch (op) {
    case NOT:
      return !val;
    default:
      return val;
  }
}

int intUNARYOP(int val, enum UNARYOP op) {
  switch (op) {
    case NEGATE:
      return -val;
      // op should never be NOT, since this would have been handled in Typecheck
    default:
      return val;
  }
}

float floatUNARYOP(float val, enum UNARYOP op) {
  switch (op) {
    case NEGATE:
      return -val;
    default:
      return val;
  }
}

commonType* performCommonTypeUNARYOP(commonType* val, enum UNARYOP op) {
  commonType* result;

  if (val->type == BOOLEAN) {

    bool tempBool = boolUNARYOP(*(bool*)val->value, op);
    result = allocateCommonType(&tempBool, BOOLEAN);

  } else if (val->type == REAL) {

    float tempFloat = floatUNARYOP(*(float*)val->value, op);
    result = allocateCommonType(&tempFloat, REAL);

  } else if (val->type == INTEGER) {

    int tempInt = intUNARYOP(*(int*)val->value, op);
    result = allocateCommonType(&tempInt, INTEGER);

  }

  return result;
}

// https://cmput415.github.io/415-docs/gazprea/spec/type_casting.html#scalar-to-scalar
// only bool, int and char can be downcast to bools
bool commonTypeToBool(commonType* val) {
  switch (val->type) {
    case BOOLEAN:
      return *(bool*)val->value;
    case INTEGER: {
        // any integer not equal to zero is considered true
        int tmpInt = *(int*)val->value;
//        printf("tmpInt: %d != 0 = %d\n", tmpInt, tmpInt != 0);
        return tmpInt != 0;
    }
    case CHAR:
    {
        char tmpChar = *(char*)val->value;
//        printf("tmpChar: %c != \\0 = %d\n", tmpChar, tmpChar != '\0');
        // chars not equal to '\0' are considered true
        return tmpChar != '\0';
    }
  }
}
