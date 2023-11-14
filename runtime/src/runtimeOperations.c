#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "runtimeCasting.c"
#include "Types/TYPES.h"
#include "run_time_errors.h"
#include "runtimeMemory.c"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
// defined binary operations between types
int intBINOP(int l, int r, enum BINOP op);
bool intCOMP(int l, int r, enum BINOP op);
float realBINOP(float l, float r, enum BINOP op);
bool realCOMP(float l, float r, enum BINOP op);
char charBINOP(char l, char r, enum BINOP op);
bool charCOMP(char l, char r, enum BINOP op);
bool boolBINOP(bool l, bool r, enum BINOP op);
bool boolUNARYOP(bool val, enum UNARYOP op);
int intUNARYOP(int val, enum UNARYOP op);
float floatUNARYOP(float val, enum UNARYOP op);
// these act differently, apply operations to each internal member
commonType* listBINOP(commonType* l, commonType* r, enum BINOP op);
commonType* listCOMP(commonType* l, commonType* r, enum BINOP op);

// perform operation between two types
commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op);
commonType* performCommonTypeUNARYOP(commonType* val, enum UNARYOP op);

// index a type
commonType* indexCommonType(commonType* indexee, int indexor);

// 'composite'. internally, it holds a list of commonTypes
bool isCompositeType(enum TYPE type) {
  switch (type) {
    case STRING:
    case VECTOR:
    case MATRIX:
    case TUPLE:
    return true;
    default:
    return false;
  }
}

// turn into bool for llvm control flow
bool commonTypeToBool(commonType* val);


int intBINOP(int l, int r, enum BINOP op) {
  switch (op) {
    case ADD:
    return l + r;
    case SUB:
    return l - r;
    case MULT:
    return l * r;
    case DIV:
    if (r==0) MathError("cannot divide by zero");
    return l/r;
    case REM:
    return fmod(l, r);
    case EXP:
    if (r==0 && r ==0) MathError("cannot exponentiate zero to the power of zero");
    return pow(l, r);
    default:
    RuntimeOPError("Unknown binary operation for INT");
    return NULL;
  }
}

bool intCOMP(int l, int r, enum BINOP op) {
  switch (op) {
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case LEQ:
    return l <= r;
    case GTHAN:
    return l > r;
    case GEQ:
    return l >= r;
    default:
    RuntimeOPError("Unknown comparison operation for INT");
  }
}

float realBINOP(float l, float r, enum BINOP op) {
  switch (op) {
    case ADD:
    return l + r;
    case SUB:
    return l - r;
    case MULT:
    return l * r;
    case DIV:
    if (r==0) MathError("cannot divide by zero");
    return l/r;
    case REM:
    return fmod(l, r);
    case EXP:
    if (r==0 && r ==0) MathError("cannot exponentiate zero to the power of zero");
    return pow(l, r);
    default:
    RuntimeOPError("Unknown binary operation for REAL");
  }
}

bool realCOMP(float l, float r, enum BINOP op) {
  switch (op) {
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case LEQ:
    return l <= r;
    case GTHAN:
    return l > r;
    case GEQ:
    return l >= r;
    default:
    RuntimeOPError("Unknown comparison operation for REAL");
  }
}

char charBINOP(char l, char r, enum BINOP op) {
  switch (op) {
    default:
    RuntimeOPError("Char does not support arithmetic BINOPs");
  }
}

bool charCOMP(char l, char r, enum BINOP op) {
  switch (op) {
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
    case LTHAN:
    return l < r;
    case LEQ:
    return l <= r;
    case GTHAN:
    return l > r;
    case GEQ:
    return l >= r;
    default:
    RuntimeOPError("Unknown comparison operation for CHAR");
  }
}

bool boolBINOP(bool l, bool r, enum BINOP op) {
  switch (op) {
    case AND:
    return l & r;
    case OR:
    return l | r;
    case XOR:
    return l ^ r;
    default:
    RuntimeOPError("Unknown binary operation for BOOL");
  }
}

commonType* listBINOP(commonType* l, commonType* r, enum BINOP op) {

  list *mlist;
  enum TYPE resultingType;

  // ugly duplicate code ahead
  
  // figure out which one of these is composite (one has to be or something is broken)
  if (isCompositeType(l->type) && isCompositeType(r->type)) {

    list* left = l->value;
    list* right = r->value;
    resultingType = l->type;

    // both lists need to be the same size in order to OP on, this is arbitrary
    mlist = allocateList(left->size);

    for (int i = 0 ; i < left->size; i ++) {
      commonType* result = performCommonTypeBINOP(left->values[i], right->values[i], op);
      appendList(mlist, result);
    }

  } else if (isCompositeType(l->type) && !isCompositeType(r->type)) {

    list* left = l->value;
    resultingType = l->type;
    mlist = allocateList(left->size);

    for (int i = 0 ; i < left->size; i ++) {
      commonType* result = performCommonTypeBINOP(left->values[i], r, op);
      appendList(mlist, result);
    }

  } else if (!isCompositeType(l->type) && isCompositeType(r->type)) {

    list* right = r->value;
    resultingType = r->type;
    mlist = allocateList(right->size);

    for (int i = 0 ; i < right->size; i ++) {
      commonType* result = performCommonTypeBINOP(l, right->values[i], op);
      appendList(mlist, result);
    }

  } else {
    UnsupportedTypeError("Reached list comparison, but neither operand is listable type");
  }

  commonType *result = allocateCommonType(&mlist, resultingType);

  return result;
}

commonType* listCOMP(commonType* l, commonType* r, enum BINOP op) {
  list *list; 

  bool compResult = true;

  // figure out which one of this is composite (one has to be or something is broken)
  if (isCompositeType(l->type) && isCompositeType(r->type)) {

  } else if (isCompositeType(l->type) && !isCompositeType(r->type)) {

  } else if (!isCompositeType(l->type) && isCompositeType(r->type)) {

  } else {
    UnsupportedTypeError("Reached list comparison, but neither operand is list");
  }

  commonType *result = allocateCommonType(&compResult, BOOLEAN);

  return result;
}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op) {
  commonType* promotedLeft;
  commonType* promotedRight;

  if (!ValidType(left->type) || !ValidType(right->type)) {
    UnsupportedTypeError("BINOP recieved a type it could not recognize");
  }

  // composites treated differenly
  if (!(isCompositeType(left->type) || isCompositeType(right->type))) {
    promotedLeft = promotion(left,right);
    promotedRight = promotion(right,left);
  }
  
  commonType* result;

  // god is dead and i have killed him
  if (!isComparison(op)) {

    if (isCompositeType(left->type) || isCompositeType(right->type)) {

      result = listBINOP(left, right, op);

    } else if(promotedLeft->type == BOOLEAN) {

      bool tempBool = boolBINOP(*(bool*)promotedLeft->value, *(bool*)promotedRight->value, op);
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
    } 
  } else {
    if (isCompositeType(left->type) || isCompositeType(right->type)) {
      result = listCOMP(left, right, op);
    } else if(promotedLeft->type == BOOLEAN) {

      bool tempBool = boolBINOP(*(bool*)promotedLeft->value, *(bool*)promotedRight->value, op);
      result = allocateCommonType(&tempBool, BOOLEAN);

    } else if (promotedLeft->type == REAL) {

      bool tempFloat = realCOMP(*(float*)promotedLeft->value, *(float*)promotedRight->value, op);
      result = allocateCommonType(&tempFloat, BOOLEAN);

    } else if (promotedLeft->type == INTEGER) {
      bool tempInt = intCOMP(*(int*)promotedLeft->value, *(int*)promotedRight->value, op);
      result = allocateCommonType(&tempInt, BOOLEAN);

    } else if (promotedLeft->type == CHAR) {

      bool tempChar = charCOMP(*(char*)promotedLeft->value, *(char*)promotedRight->value, op);
      result = allocateCommonType(&tempChar, BOOLEAN);
    } 
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

// assume we are indexing a tuploe item
commonType* indexCommonType(commonType* indexee, int indexor) {
  list* list = indexee->value;
  return list->values[indexor];
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
