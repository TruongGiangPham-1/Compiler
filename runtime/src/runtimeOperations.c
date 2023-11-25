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
commonType* indexCommonType(commonType* indexee, commonType* indexor);

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

list* listify(commonType* item) {
  list* newList = allocateList(1);
  appendList(newList, copyCommonType(item));
  return newList;
}

list* concat(list* l, list* r) {
  int concatenatedSize = l->size + r->size;
  list* newList = allocateList(concatenatedSize);

  for (int i = 0; i < l->size ; i ++) {
    
    commonType* result = (r->size > 0) ? promotion(l->values[i], r->values[0]) : copyCommonType(l->values[i]);

    appendList(newList, result);
  }

  for (int i = 0; i < r->size ; i ++) {

    commonType* result = (l->size > 0) ? promotion(r->values[i], l->values[0]) : copyCommonType(r->values[i]);

    appendList(newList, result);
  }
  
  return newList;
}

list* stride(list* l, int stride) {
  int finalSize = ceil(l->size / (float)stride);
  list* newList = allocateList(finalSize);

  for (int i = 0 ; i < l->size ; i += stride) {
    appendList(newList, l->values[i]);
  }
  return newList;
}

commonType* listBINOP(commonType* l, commonType* r, enum BINOP op) {
  if (!isCompositeType(l->type) && !isCompositeType(r->type)) {
    UnsupportedTypeError("Reached list binop, but neither operand is listable type");
  }

  switch (op) {
    case STRIDE:
    {
      commonType* castedRight = cast(r, INTEGER);

      list* newlist = stride((list*)l->value, *(int*)castedRight->value);

      deallocateCommonType(castedRight);

      // TODO: leaking here 
      return allocateCommonType(&newlist, l->type);

    }
    case ADD:
    case MULT:
    case SUB:
    case DIV:
    case EXP:
    case REM:
    {
      // if not one then the other
      int listSize = isCompositeType(l->type) ? ((list*) l->value)->size : ((list*) r->value)->size;
      list *mlist = allocateList(listSize);

      for (int i = 0 ; i < listSize ; i ++) {
        commonType* left = isCompositeType(l->type) ? ((list*) l->value)->values[i]: l;
        commonType* right = isCompositeType(r->type) ? ((list*) r->value)->values[i]: r;

        commonType* result = performCommonTypeBINOP(left, right, op);
        appendList(mlist, result);
      }

      enum TYPE resultingType = isCompositeType(l->type) ? l->type : r->type;;
      commonType *result = allocateCommonType(&mlist, resultingType);

      return result;
    }
    default:
    RuntimeOPError("Unknown binary operation between lists");
  }
}

commonType* listCOMP(commonType* l, commonType* r, enum BINOP op) {
  if (!isCompositeType(l->type) && !isCompositeType(r->type)) {
    UnsupportedTypeError("Reached list comparison, but neither operand is listable type");
  }
  
  // if not one then the other
  int listSize = isCompositeType(l->type) ? ((list*) l->value)->size : ((list*) r->value)->size;
  bool compResult = true;

  for (int i = 0 ; i < listSize ; i ++) {
    commonType* left = isCompositeType(l->type) ? ((list*) l->value)->values[i]: l;
    commonType* right = isCompositeType(r->type) ? ((list*) r->value)->values[i]: r;

    commonType* result = performCommonTypeBINOP(left, right, op);

    if (!*(bool*)result->value) {
      compResult = false;
    }

    deallocateCommonType(result);
  }

  commonType *result = allocateCommonType(&compResult, BOOLEAN);

  return result;
}

commonType* vectorFromRange(commonType* lower, commonType* upper) {
  commonType* castedLower = cast(lower, INTEGER);
  commonType* castedUpped = cast(upper, INTEGER);

  list* newList = allocateList(*(int*)upper->value - *(int*)lower->value);

  for(int i = *(int*)lower->value ; i < *(int*)upper->value ; i ++) {
    commonType* newItem = allocateCommonType(&i, INTEGER);
    appendList(newList, newItem); 
  }

  return allocateCommonType(&newList, VECTOR);
}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op) {
  commonType* promotedLeft;
  commonType* promotedRight;

  if (!ValidType(left->type) || !ValidType(right->type)) {
    UnsupportedTypeError("BINOP recieved a type it could not recognize");
  }

  commonType* result;
  // kind of ugly to put here but i am exhausted
  if (op == CONCAT) {
    list* l = isCompositeType(left->type) ? (list*)left->value : listify(left);
    list* r = isCompositeType(right->type) ? (list*)right->value : listify(right);

    list* newlist = concat((list*)l, (list*)r);

    // concat should be between vectors or strings
    // TODO: leaking here 
    result = allocateCommonType(&newlist, (left->type == STRING || right->type == STRING) ? STRING : VECTOR);
    if (!isCompositeType(left->type)) {
      deallocateList(l);
    }
    if (!isCompositeType(right->type)) {
      deallocateList(r);
    }
    return result;
  }

  if (!(isCompositeType(left->type) || isCompositeType(right->type))) {
    promotedLeft = promotion(left,right);
    promotedRight = promotion(right,left);
  }
  
    
  if (op == RANGE) {
    return vectorFromRange(left, right);
  }

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

  if (!(isCompositeType(left->type) || isCompositeType(right->type))) {
    deallocateCommonType(promotedLeft);
    deallocateCommonType(promotedRight);
  }

#ifdef DEBUGMEMORY
  printf("=== complete\n");
#endif /* ifdef DEBUGMEMORY */

  return result;
}
commonType* listUNARYOP(commonType* l, enum UNARYOP op) {
  if (!isCompositeType(l->type)) {
    UnsupportedTypeError("Reached list unaryop, but neither operand is listable type");
  }

  switch (op) {
    case NEGATE:
    case NOT:
    {
      // if not one then the other
      int listSize = ((list*) l->value)->currentSize;
      list *mlist = allocateList(listSize);

      for (int i = 0 ; i < listSize ; i ++) {
        commonType* newItem = performCommonTypeUNARYOP(((list*) l->value)->values[i], op);
        appendList(mlist, newItem);
      }

      enum TYPE resultingType = l->type;
      commonType *result = allocateCommonType(&mlist, resultingType);

      return result;
    }
    default:
    RuntimeOPError("Unknown unary operation on list");
  }
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
  
  if (isCompositeType(val->type)) {

    result = listUNARYOP(val, op);

  } else if (val->type == BOOLEAN) {

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
commonType* indexCommonType(commonType* indexee, commonType* indexor) {
  list* list = indexee->value;
  return list->values[*(int*)indexor->value];
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

// STANDARD LIBRARY. They are prefixed with __ because they can be called with regular
// function calls in the walker.
commonType* __length(commonType* vector)  {
  if (vector->type != VECTOR) {
    UnsupportedTypeError("Trying to take length of non-vector type");
  }

  int length = ((list*)vector->value)->currentSize;

  return allocateCommonType(&length, INTEGER);
}

commonType* __rows(commonType* matrix) {
  // we don't differnetiate matrices and vectors
  if (matrix->type != VECTOR) {
    UnsupportedTypeError("Trying to take row of non-matrix type");
  }

  return __length(matrix);
}

commonType* __columns(commonType* matrix) {
  // we don't differnetiate matrices and vectors
  if (matrix->type != VECTOR) {
    UnsupportedTypeError("Trying to take column of non-matrix type");
  }

  commonType* row = ((list*)matrix->value)->values[0];

  return __length(row);
}

commonType* allocateFromRange(commonType* lower, commonType* upper) {
  commonType* castedLower = cast(lower, INTEGER);
  commonType* castedUpper = cast(upper, INTEGER);

  int lowerVal = *(int*)castedLower->value;
  int upperVal = *(int*)castedUpper->value;

  // allocate of size 1 if nothing. 1 just lets us stay consistent with de-alloc
  list* newList = allocateList(upperVal - lowerVal + 1<= 0 ? 1 : upperVal - lowerVal + 1);

  for (int i = lowerVal ; i <= upperVal ; i++) {
    commonType* newItem = allocateCommonType(&i, INTEGER);

    appendList(newList, newItem);
  }

  deallocateCommonType(castedLower);
  deallocateCommonType(castedUpper);

  return allocateCommonType(&newList, VECTOR);
}
