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
void printCommonType(commonType *type);

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

commonType* __rows(commonType* matrix);
commonType* __columns(commonType* matrix);


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
    case EQUAL:
    return l == r;
    case NEQUAL:
    return l != r;
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
  if (stride <= 0) StrideError("Bad stride");

  int finalSize = ceil(l->currentSize / (float)stride);
  list* newList = allocateList(finalSize);

  for (int i = 0 ; i < l->currentSize; i += stride) {
    appendList(newList, l->values[i]);
  }

  return newList;
}


commonType* matrixMultiply(commonType* left, commonType* right) {
  list* lList = (list*)left->value;
  list* rList = (list*)right->value;

  commonType* lCols = __columns(left);
  commonType* lRows = __rows(left);

  commonType* rCols = __columns(right);
  commonType* rRows = __rows(right);

  int intLCols = *(int*)lCols->value;
  int intRows = *(int*)rRows->value;

  if (intLCols != intRows) SizeError("Incompatible matrix multiply");

  int oneInit = 1;
  int zero = 0;
  commonType* one = allocateCommonType(&oneInit, INTEGER);

  commonType* row = allocateCommonType(&oneInit, INTEGER) ;
  list* rowList = allocateListFromCommon(lRows);

  while (commonTypeToBool(performCommonTypeBINOP(row, lRows, LEQ))) {

    commonType* col  = allocateCommonType(&oneInit, INTEGER);
    list* colList = allocateListFromCommon(rCols);

    while (commonTypeToBool(performCommonTypeBINOP(col, rCols, LEQ))) {

      commonType* newItem = allocateCommonType(&zero, INTEGER);
      commonType* k = allocateCommonType(&oneInit, INTEGER);
          
      while (commonTypeToBool(performCommonTypeBINOP(k, rRows, LEQ))) {

        commonType *leftItem = indexCommonType(indexCommonType(left, row), k);
        commonType *rightItem = indexCommonType(indexCommonType(right, k), col);

        commonType* product = performCommonTypeBINOP(leftItem, rightItem, MULT);

        assignByReference(newItem, performCommonTypeBINOP(newItem, product, ADD));

        assignByReference(k, performCommonTypeBINOP(k, one, ADD));
      }

      deallocateCommonType(k);                   
      appendList(colList, newItem);

      assignByReference(col, performCommonTypeBINOP(col, one, ADD));
    }
    
    appendList(rowList, allocateCommonType(&colList, VECTOR));
    assignByReference(row, performCommonTypeBINOP(row, one, ADD));
    deallocateCommonType(col);
  }

  deallocateCommonType(row);
  deallocateCommonType(one);

  deallocateCommonType(lCols);
  deallocateCommonType(lRows);

  deallocateCommonType(rCols);
  deallocateCommonType(rRows);

  return allocateCommonType(&rowList, VECTOR);
}

commonType* crossProduct(commonType* left, commonType* right) {
  list* lList = (list*)left->value;
  list* rList = (list*)right->value;

  int zero = 0;
  commonType* sum = allocateCommonType(&zero, INTEGER);

  for (int i = 0 ; i < lList->currentSize ; i ++) {
    commonType* leftItem = lList->values[i];
    commonType* rightItem = rList->values[i];

    commonType* result = performCommonTypeBINOP(leftItem, rightItem, MULT);
    assignByReference(sum, performCommonTypeBINOP(sum, result, ADD));
    deallocateCommonType(result);
  }

  return sum;
}


/* covers matrix multiply + dot product. General
 */
commonType* dotProduct(commonType* left, commonType* right) {
  if (isCompositeType(((list*)left->value)->values[0]->type)) {
    return matrixMultiply(left, right);
  } else {
    return crossProduct(left, right);
  }
}

commonType* listBINOP(commonType* l, commonType* r, enum BINOP op) {
  if (!isCompositeType(l->type) && !isCompositeType(r->type)) {
    UnsupportedTypeError("Reached list binop, but neither operand is listable type");
  }

  switch (op) {
    case STRIDE:
    {
      commonType* castedRight = castHelper(r, INTEGER);

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
        commonType* left = ((list*) l->value)->values[i];
        commonType* right = ((list*) r->value)->values[i];
        
        commonType* result = performCommonTypeBINOP(left, right, op);
        appendList(mlist, result);
      }
      enum TYPE resultingType = l->type;
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
  commonType* castedLower = castHelper(lower, INTEGER);
  commonType* castedUpped = castHelper(upper, INTEGER);

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

  if (op == STRIDE) {
      commonType* castedRight = castHelper(right, INTEGER);

      list* newlist = stride((list*)left->value, *(int*)castedRight->value);

      deallocateCommonType(castedRight);

      // TODO: leaking here 
      return allocateCommonType(&newlist, left->type);
  }

  if (op == DOT_PROD) {
    return dotProduct(left, right);
  }

  promotedLeft = promotion(left,right);
  promotedRight = promotion(right,left);
    
  if (op == RANGE) {
    return vectorFromRange(left, right);
  }




  // god is dead and i have killed him
  if (!isComparison(op)) {

    if (isCompositeType(promotedLeft->type)) {

      result = listBINOP(promotedLeft, promotedRight, op);

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
    if (isCompositeType(promotedLeft->type)) {

      result = listCOMP(promotedLeft, promotedRight, op);

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

  deallocateCommonType(promotedLeft);
  deallocateCommonType(promotedRight);

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
      RuntimeOPError("Unknown unary operation on bool");
      return val;
  }
}

int intUNARYOP(int val, enum UNARYOP op) {
  switch (op) {
    case NEGATE:
      return -val;
      // op should never be NOT, since this would have been handled in Typecheck
    default:
      RuntimeOPError("Unknown unary operation on int");
      return val;
  }
}

float floatUNARYOP(float val, enum UNARYOP op) {
  switch (op) {
    case NEGATE:
      return -val;
    default:
      RuntimeOPError("Unknown unary operation on float");
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

  } else {

    RuntimeOPError("Unknown unary operation");

  }

  return result;
}

// assume we are indexing a tuploe item
commonType* indexCommonType(commonType* indexee, commonType* indexor) {
  list* list = indexee->value;
  int index = *(int*)indexor->value - 1;

  if (index >= list->currentSize || index < 0) IndexError("out of bounds index");

  return list->values[index];
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

// *state is a globalop defined in the BackEnd::setupStreamRuntime
commonType* __stream_state(int* state) {
  return allocateCommonType(state, INTEGER);
}

commonType* __length(commonType* vector)  {
  if (!isCompositeType(vector->type)) {
    UnsupportedTypeError("Trying to take length of non-vector type");
  }

  int length = ((list*)vector->value)->currentSize;

  return allocateCommonType(&length, INTEGER);
}

commonType* __rows(commonType* matrix) {
  // we don't differnetiate matrices and vectors
  if (!isCompositeType(matrix->type)) {
    UnsupportedTypeError("Trying to take row of non-matrix type");
  }

  return __length(matrix);
}

commonType* __columns(commonType* matrix) {
  // we don't differnetiate matrices and vectors
  if (!isCompositeType(matrix->type)) {
    UnsupportedTypeError("Trying to take column of non-matrix type");
  }

  commonType* row = ((list*)matrix->value)->values[0];

  return __length(row);
}

commonType* __reverse(commonType* vector)  {
  if (!isCompositeType(vector->type)) {
    UnsupportedTypeError("Trying to take length of non-vector type");
  }

  list* mlist = (list*)vector->value;
  list* newList = allocateList(mlist->currentSize);

  for (int i = mlist->currentSize - 1; i >= 0 ; i--) {
    appendList(newList,copyCommonType(mlist->values[i]));
  }

  return allocateCommonType(&newList, VECTOR);
}

commonType* __format(commonType* value) {
  int len;
  char* charArr;
  switch (value->type)  {
    case INTEGER: 
    {
      len = snprintf(NULL, 0, "%d", *(int*)value->value);
      snprintf(charArr, len+1,"%d", *(int*)value->value);
      break;
    }
    case REAL:
    {
      len = snprintf(NULL, 0, "%g", *(float*)value->value);
      snprintf(charArr, len+1, "%g", *(float*)value->value);
      break;
    }
    case CHAR:
    {
      len = 1;
      charArr = malloc(sizeof(char) * 2);
      charArr[0] = *(char*)value->value;
      break;
    }
    case BOOLEAN:
    {
      len = 1;
      charArr = malloc(sizeof(char) * 2);
      charArr[0] = *(bool*)value->value == true ? 'T' : 'F';
      break;
    }
    default:
    RuntimeOPError("Trying to stringify unstringable type");
  }

  list* newList = allocateList(len);
  for (int i = 0; i < len ; i++) {
    commonType* newChar = allocateCommonType(&charArr[i], CHAR);
    appendList(newList, newChar);
  }

  return allocateCommonType(&newList, STRING);
}

commonType* allocateFromRange(commonType* lower, commonType* upper) {

  commonType* castedLower = castHelper(lower, INTEGER);
  commonType* castedUpper = castHelper(upper, INTEGER);

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
