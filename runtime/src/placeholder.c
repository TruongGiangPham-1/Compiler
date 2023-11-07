#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "BuiltinTypes/BuiltInTypes.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

//#define DEBUGTUPLE
#define DEBUGPROMOTION
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
      printf("%s", *(bool*)type->value ? "true" : "false");
      break;
    case REAL:
      printf("%f", *(float*)type->value);
      break;
    case TUPLE:
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
        dest->value = *(tuple**)value;
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

commonType* boolCast(bool fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from bool\n");
#endif /* ifdef DEBUGPROMOTION */
  switch (toType) {
    case BOOL:
    {
      return allocateCommonType(&fromValue, BOOL);
    }
    case INT:
    {
      int tempInt = (int)fromValue;
      return allocateCommonType(&tempInt, INT);
    }
    case REAL:
    {
      float tempReal = fromValue ? 1.0f : 0.0;
      return allocateCommonType(&tempReal, REAL);
    }
    case CHAR:
    {
      char tempChar = fromValue;
      return allocateCommonType(&tempChar, CHAR);
    }
    case TUPLE:
    // we will never have typles here
  }
}

commonType* intCast(int fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from int\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
    case BOOL:
    {
      bool tempBool = (bool)fromValue;
      return allocateCommonType(&tempBool, BOOL);
    }
    case INT:
    {
      return allocateCommonType(&fromValue, INT);
    }
    case REAL:
    {
      float tempReal = (float)fromValue;
      return allocateCommonType(&tempReal, REAL);
    }
    case CHAR:
    {
      char tempChar =  ((unsigned int) fromValue) % 256;
      return allocateCommonType(&tempChar, CHAR);
    }
    case TUPLE:
    // cannot cast to tuple
    return NULL;
  }
}

commonType* charCast(char fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from char\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
    case BOOL:
    {
      bool tempBool = (bool)fromValue;
      return allocateCommonType(&tempBool, BOOL);
    }
    case INT:
    {
      int tempInt = (int)fromValue;
      return allocateCommonType(&tempInt, INT);
    }
    case REAL:
    {
      float tempReal = (float)fromValue;
      return allocateCommonType(&tempReal, REAL);
    }
    case CHAR:
    {
      return allocateCommonType(&fromValue, CHAR);
    }
    case TUPLE:
    // cannot cast to tuple
    return NULL;
  }
}

commonType* realCast(float fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from real\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
    case BOOL:
    {
      return NULL;
    }
    case INT:
    {
      int tempInt = (int)fromValue;
      return allocateCommonType(&fromValue, INT);
    }
    case REAL:
    {
      return allocateCommonType(&fromValue, REAL);
    }
    case CHAR:
    {
      return NULL;
    }
    case TUPLE:
    // cannot cast to tuple
    return NULL;
  }
}

commonType* cast(commonType* from, enum BuiltIn toType) {
  switch (from->type) {
    case BOOL:
    return boolCast(*(bool*)from->value, toType);
    case INT:
    return intCast(*(int*)from->value, toType);
    case CHAR:
    return charCast(*(char*)from->value, toType);
    case TUPLE:
    // don't think we need this 
    break;
    case REAL:
    return realCast(*(float*)from->value, toType);
  }
}

commonType* boolPromotion(commonType* fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Cast from int\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
  default:
  return cast(fromValue, BOOL);
  }
}

commonType* intPromotion(commonType* fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Promotion from int\n");
#endif /* ifdef DEBUGPROMOTION */

  switch (toType) {
    case REAL:
#ifdef DEBUGPROMOTION
  printf("To real\n");
#endif /* ifdef DEBUGPROMOTION */
    return cast(fromValue, REAL);
    default:
    return cast(fromValue, INT);
  }
}

commonType* charPromotion(commonType* fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Promotion from char\n");
#endif /* ifdef DEBUGPROMOTION */

    switch (toType) {
    default:
#ifdef DEBUGPROMOTION
  printf("To char\n");
#endif /* ifdef DEBUGPROMOTION */
    return cast(fromValue, CHAR);
  }
}

commonType* realPromotion(commonType* fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("Promotion from real\n");
#endif /* ifdef DEBUGPROMOTION */
    switch (toType) {
#ifdef DEBUGPROMOTION
  printf("To real\n");
#endif /* ifdef DEBUGPROMOTION */
    default:
    return cast(fromValue, REAL);
  }
}

// promote and return temporary
commonType* promotion(commonType* from, commonType* to) {
  switch (from->type) {
    case BOOL:
    return boolPromotion(from, to->type);
    case INT:
    return intPromotion(from, to->type);
    case CHAR:
    return charPromotion(from, to->type);
    case TUPLE:
    // don't think we need this 
    break;
    case REAL:
    return realPromotion(from->value, to->type);
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
    case LEQ:
    return l <= r;
    case GTHAN:
    return r > l;
    case GEQ:
    return r >= l;
    case REM:
    return l % r;
    case EXP:
    // we do a little truth table analysis
    return !(!l & r);
    case AND:
    return l & r;
    case OR:
    return l | r;
    case XOR:
    return l ^ r;
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
    case LEQ:
    return l <= r;
    case GTHAN:
    return r > l;
    case GEQ:
    return r >= l;
    case REM:
    return l % r;
    case EXP:
    return pow(l,r);
    case AND:
    return l & r;
    case OR:
    return l | r;
    case XOR:
    return l ^ r;
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
    case LEQ:
    return l <= r;
    case GTHAN:
    return r > l;
    case GEQ:
    return r >= l;
    case REM:
    return fmod(l, r);
    case EXP:
    return pow(l, r);
    case AND:
      {
      // more memory hacking
      // floats cannot be binop'd. Have to put their bits in an int and do it
      uint32_t leftTemp;
      memcpy(&leftTemp, &l, sizeof(float));

      uint32_t rightTemp;
      memcpy(&rightTemp, &r, sizeof(float));

      leftTemp = leftTemp & rightTemp;
      float result;

      memcpy(&result, &leftTemp, sizeof(float));
      return result;
    }
    case OR:
    {
      // more memory hacking
      // floats cannot be binop'd. Have to put their bits in an int and do it
      uint32_t leftTemp;
      memcpy(&leftTemp, &l, sizeof(float));

      uint32_t rightTemp;
      memcpy(&rightTemp, &r, sizeof(float));

      leftTemp = leftTemp | rightTemp;
      float result;

      memcpy(&result, &leftTemp, sizeof(float));
      return result;
    }    
    case XOR:
    {
      // more memory hacking
      // floats cannot be binop'd. Have to put their bits in an int and do it
      uint32_t leftTemp;
      memcpy(&leftTemp, &l, sizeof(float));

      uint32_t rightTemp;
      memcpy(&rightTemp, &r, sizeof(float));

      leftTemp = leftTemp ^ rightTemp;
      float result;

      memcpy(&result, &leftTemp, sizeof(float));
      return result;
    }  
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
    case LEQ:
    return l <= r;
    case GTHAN:
    return r > l;
    case GEQ:
    return r >= l;
    case REM:
    return l % r;
    case EXP:
    return pow(l,r);
    case AND:
    return l & r;
    case OR:
    return l | r;
    case XOR:
    return l ^ r;
  }
}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op);


commonType* tupleBINOP(tuple* l, tuple* r, enum BINOP op) {
  tuple *tuple = allocateTuple(l->size);

  for (int i = 0 ; i < l->currentSize ; i ++) {
    appendTuple(tuple, performCommonTypeBINOP(l->values[i], r->values[i], op));
  }

  for (int i = 0; i < tuple->size ; i++) {
    printCommonType(tuple->values[i]);
  }

  commonType *result = allocateCommonType(&tuple, TUPLE);

  return result;
}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op) {
  commonType* promotedLeft;
  commonType* promotedRight;

  // tuples treated differenly
  if (!(left->type == TUPLE)) {
    promotedLeft = promotion(left,right);
    promotedRight = promotion(right,left);
  }
  
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

  } else {
    // tuples don't need promotions, their held items do.
    result = tupleBINOP(*(tuple**)left->value, *(tuple**)right->value, op);
  }
  
  // temporary operands
#ifdef DEBUGMEMORY
  printf("=== de allocating temporary operands...\n");
#endif /* ifdef DEBUGMEMORY */

  if (!(left->type == TUPLE)) {
    deallocateCommonType(promotedLeft);
    deallocateCommonType(promotedRight);
  }

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

  if (val->type == BOOL) {

    bool tempBool = boolUNARYOP(*(bool*)val->value, op);
    result = allocateCommonType(&tempBool, BOOL);

  } else if (val->type == REAL) {

    float tempFloat = floatUNARYOP(*(float*)val->value, op);
    result = allocateCommonType(&tempFloat, REAL);

  } else if (val->type == INT) {

    int tempInt = intUNARYOP(*(int*)val->value, op);
    result = allocateCommonType(&tempInt, INT);

  }

  return result;
}
