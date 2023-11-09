#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

bool isCOMP(enum BINOP op) {
  switch (op) {
    case EQUAL:
    case NEQUAL:
    case GTHAN:
    case LTHAN:
    case GEQ:
    case LEQ:
    return true;
    default:
    return false;
  }
}
#define DEBUGTUPLE
//#define DEBUGTYPES
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
tuple* copyTuple(tuple* copyFrom);


void printType(commonType *type, bool nl) {
  switch (type->type) {
    case INTEGER:
      printf("%d", *(int*)type->value);
      break;
    case CHAR:
      printf("%c", *(char*)type->value);
      break;
    case BOOLEAN:
      printf("%s", *(bool*)type->value ? "true" : "false");
      break;
    case REAL:
      printf("%g", *(float*)type->value);
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
commonType* copyCommonType(commonType* copyFrom);

void assignByReference(commonType* dest, commonType* from) {
  dest->value = copyCommonType(from);
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

commonType* copyCommonType(commonType* copyFrom) {
  if (copyFrom->type == TUPLE) {
    printf("\n\ncopying a tuple");
    tuple* copiedTuple = copyTuple((tuple*)copyFrom->value);
    return allocateCommonType(&copiedTuple, TUPLE);
  } else {
    printf("\n\ncopying a regular type");
    return allocateCommonType(copyFrom->value, copyFrom->type);
  }
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
      deallocateTuple((tuple*)object->value);
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

tuple* copyTuple(tuple* copyFrom) {
  tuple* copiedTo = allocateTuple(copyFrom->size);

  for (int i = 0 ; i < copyFrom->size ; i ++) {
    commonType* newVal = copyCommonType(copyFrom->values[i]);
    appendTuple(copiedTo, newVal);
  }

  return copiedTo;
}

commonType* boolCast(bool fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Cast from bool\n");
#endif /* ifdef DEBUGTYPES */
  switch (toType) {
    case BOOLEAN:
    {
      return allocateCommonType(&fromValue, BOOLEAN);
    }
    case INTEGER:
    {
      int tempInt = (int)fromValue;
      return allocateCommonType(&tempInt, INTEGER);
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
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

commonType* intCast(int fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Cast from int\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType) {
    case BOOLEAN:
    {
      bool tempBool = (bool)fromValue;
      return allocateCommonType(&tempBool, BOOLEAN);
    }
    case INTEGER:
    {
      return allocateCommonType(&fromValue, INTEGER);
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
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

commonType* charCast(char fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Cast from char\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType) {
    case BOOLEAN:
    {
      bool tempBool = (bool)fromValue;
      return allocateCommonType(&tempBool, BOOLEAN);
    }
    case INTEGER:
    {
      int tempInt = (int)fromValue;
      return allocateCommonType(&tempInt, INTEGER);
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
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

commonType* realCast(float fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Cast from real\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType) {
    case INTEGER:
    {
  #ifdef DEBUGTYPES
    printf("To int!\n");
  #endif /* ifdef DEBUGTYPES */
      int tempInt = (int)fromValue;
      return allocateCommonType(&tempInt, INTEGER);
    }
    case REAL:
    {
  #ifdef DEBUGTYPES
    printf("To real!\n");
  #endif /* ifdef DEBUGTYPES */
      return allocateCommonType(&fromValue, REAL);
    }
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
      return NULL;
  }
}

commonType* cast(commonType* from, enum TYPE toType) {
#ifdef DEBUGTYPES
    printf("Choosing appropriate case...\n");
#endif /* ifdef DEBUGTYPES */

  switch (from->type) {
    case BOOLEAN:
#ifdef DEBUGTYPES
    printf("Bool!\n");
#endif /* ifdef DEBUGTYPES */
    return boolCast(*(bool*)from->value, toType);

    case INTEGER:
#ifdef DEBUGTYPES
    printf("Int!\n");
#endif /* ifdef DEBUGTYPES */
    return intCast(*(int*)from->value, toType);

    case CHAR:
#ifdef DEBUGTYPES
    printf("Char!\n");
#endif /* ifdef DEBUGTYPES */
    return charCast(*(char*)from->value, toType);

    break;
    case REAL:
#ifdef DEBUGTYPES
    printf("Real!\n");
#endif /* ifdef DEBUGTYPES */

    return realCast(*(float*)from->value, toType);
    default:

#ifdef DEBUGTYPES
    printf("Error! Uncastable type!\n");
#endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

commonType* boolPromotion(commonType* fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Promotion from bool\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType) {
  case BOOLEAN:
#ifdef DEBUGTYPES
  printf("To bool!\n");
#endif /* ifdef DEBUGTYPES */
  return cast(fromValue, BOOLEAN);

  default:
#ifdef DEBUGTYPES
  printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
  return NULL;
  }
}

commonType* intPromotion(commonType* fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Promotion from int\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType) {
    case REAL:
#ifdef DEBUGTYPES
  printf("To real\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, REAL);

    case INTEGER:
#ifdef DEBUGTYPES
  printf("To int\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, INTEGER);

    default:
#ifdef DEBUGTYPES
  printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

commonType* charPromotion(commonType* fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Promotion from char\n");
#endif /* ifdef DEBUGTYPES */

    switch (toType) {
    case CHAR:
#ifdef DEBUGTYPES
    printf("To char\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, CHAR);

    default:
#ifdef DEBUGTYPES
    printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

commonType* realPromotion(commonType* fromValue, enum TYPE toType) {
#ifdef DEBUGTYPES
  printf("Promotion from real\n");
#endif /* ifdef DEBUGTYPES */
    switch (toType) {
    case REAL:
#ifdef DEBUGTYPES
  printf("To real\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, REAL);
    case INTEGER:
#ifdef DEBUGTYPES
  printf("To real\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, REAL);
    default:
#ifdef DEBUGTYPES
  printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
    return NULL;
  }
}

// promote and return temporary
commonType* promotion(commonType* from, commonType* to) {
  switch (from->type) {
    case BOOLEAN:
    return boolPromotion(from, to->type);
    case INTEGER:
    return intPromotion(from, to->type);
    case CHAR:
    return charPromotion(from, to->type);
    case TUPLE:
    // don't think we need this 
    return NULL;
    break;
    case REAL:
    return realPromotion(from, to->type);
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
    default:
    // should never be here
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
    // should never be here
    return NULL;
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
    return l/r;
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
    default:
    return 0;
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
    return NULL;
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
    default:
    return NULL;
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
    return NULL;
  }
}

commonType* performCommonTypeBINOP(commonType* left, commonType* right, enum BINOP op);


commonType* tupleBINOP(tuple* l, tuple* r, enum BINOP op) {
  tuple *tuple = allocateTuple(l->size);

  for (int i = 0 ; i < l->currentSize ; i ++) {
    appendTuple(tuple, performCommonTypeBINOP(l->values[i], r->values[i], op)); 
  }

  commonType *result = allocateCommonType(&tuple, TUPLE);

  return result;
}

commonType* tupleCOMP(tuple* l, tuple* r, enum BINOP op) {
  tuple *tuple = allocateTuple(l->size);

  bool compResult = true;

  for (int i = 0 ; i < l->currentSize ; i ++) {
    commonType* result = performCommonTypeBINOP(l->values[i], r->values[i], op); 
    if (! *(bool*)result->value) {
      compResult = false;
    }
  }

  commonType *result = allocateCommonType(&compResult, BOOLEAN);

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

  // god is dead and i have killed him
  if (!isCOMP(op)) {
    if(promotedLeft->type == BOOLEAN) {

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

    } else {
      // tuples don't need promotions, their held items do.
      result = tupleBINOP((tuple*)left->value, (tuple*)right->value, op);
    }
  } else {
    if(promotedLeft->type == BOOLEAN) {

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

    } else {
      // tuples don't need promotions, their held items do.
      result = tupleCOMP((tuple*)left->value, (tuple*)right->value, op);
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



