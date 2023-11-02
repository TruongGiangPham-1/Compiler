#include "Operands/BINOP.h"
#include "BuiltinTypes/BuiltInTypes.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define DEBUGTUPLE
#define DEBUGPROMOTION

void print(int i) {
  printf("%d\n", i);
}

typedef struct vecStruct {
  int* base;
  int sizeOf;
} vecStruct;

typedef struct commonType {
  enum BuiltIn type; // type. defined in an enum
  long int* value; // long int is technically void ptr. i'm seeing some unexpected interactions
  // someone smarter should ask me about this and investigate
} commonType;

typedef struct tuple {
  int size;
  int currentSize;
  commonType** values; // list of values
} tuple;

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
        tuple *mTuple = (tuple *)(*type->value);
        #ifdef DEBUGTUPLE
        printf("Printing tuple %p\n", mTuple);
        #endif
        printf("(");
        for (int i = 0 ; i < mTuple->size ; i++) {
          #ifdef DEBUGTUPLE
          printf("\nprinting tuple value at %p\n", &mTuple->values[i]);
          #endif
          printType(mTuple->values[i], false);
          if (i != mTuple->size) printf(" ");
        }
        printf(")");
      }
      break;
  }

  if (nl) printf("\n");
  return;
}

void printCommonType(commonType *type) {
  printf("Printing common type %p ", type);
  printType(type, true);
}

void printAddress(long int* value) {
  printf("HERE %p\n", value);
}


// can later check if list types to de-allocate 
commonType* allocateCommonType(void* value, enum BuiltIn type) {
  commonType* newType = (commonType*)malloc(sizeof(commonType));
  newType->type = type;

  switch (type) {
    case INT:
      {
        int* newIntVal = malloc(sizeof(int));
        *newIntVal = *(int*)value;
        newType->value = (long*)newIntVal;
      }
      break;
    case BOOL:
      {
        bool* newBoolVal = malloc(sizeof(bool));
        *newBoolVal = *(bool*)value;
        newType->value = (long*)newBoolVal;
      }
      break;
    case REAL:
      {
        float* newFloatVal = malloc(sizeof(float));
        *newFloatVal = *(float*)value;
        newType->value = (long*)newFloatVal;
      }
      break;
    case TUPLE:
      {
        newType->value = (tuple*)value;
      }
      break;
    case CHAR: 
      {
        char* newCharVal = malloc(sizeof(char));
        *newCharVal = *(char*)value;
        newType->value = (long*)newCharVal;
      }

      break;
  }

  printf("===\nAllocated common type: %p\nPrinting Contents\n",newType);
  printCommonType(newType);
  printf("===\n");

  return newType;
}

tuple* allocateTuple(int size) {
  tuple* newTuple = (tuple*) malloc(sizeof(tuple));
  commonType** valueList = (commonType**) calloc(size, sizeof(commonType*));
  memset(valueList, 0, size * sizeof(commonType));

  newTuple->size = size;
  newTuple->currentSize = 0;
  newTuple->values= valueList;

  #ifdef DEBUGTUPLE
  printf("Created tuple at %p\n", newTuple);
  printf("tuple list starts at %p\n", valueList);
  #endif
  
  return newTuple;
};

void appendTuple(tuple* tuple, commonType *value) {
#ifdef DEBUGTUPLE
  printf("====== Appending to tuple\n");
  printf("Tuple currently holding %p  at index %d address %p\n", tuple->values[tuple->currentSize], tuple->currentSize, &tuple->values[tuple->currentSize]);
#endif

  tuple->values[tuple->currentSize] = value;

#ifdef DEBUGTUPLE
  printf("appended to tuple at %p, %p\n", &tuple->values[tuple->currentSize], value);
  printf("Tuple now holding %p  at index %d address %p\n", tuple->values[tuple->currentSize], tuple->currentSize, &tuple->values[tuple->currentSize]);
  printf("appended value is:\n");
  printCommonType(value);
  printf("====== Append complete\n"); 
#endif /* ifdef DEBUGTUPLE */

  tuple->currentSize++;
}

// bjarne stoustrup would've taken my keyboard away
commonType* boolPromotion(bool fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("promoting from bool\n");
#endif /* ifdef DEBUGPROMOTION */
  switch (toType) {
    case BOOL:
    return allocateCommonType((long int*)&fromValue, BOOL);
    case INT:
    return allocateCommonType((long int*)&fromValue, BOOL);
    case CHAR:
    return allocateCommonType((long int*)&fromValue, BOOL);
    case TUPLE:
    return allocateCommonType((long int*)&fromValue, BOOL);
    case REAL:
    return allocateCommonType((long int*)&fromValue, BOOL);
  }
}

commonType* intPromotion(int fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("promoting from int\n");
#endif /* ifdef DEBUGPROMOTION */
    switch (toType) {
    case BOOL:
#ifdef DEBUGPROMOTION
  printf("To bool\n");
#endif /* ifdef DEBUGPROMOTION */
    bool new = (bool)fromValue;
    return allocateCommonType((void*)&new, BOOL);
    case INT:
#ifdef DEBUGPROMOTION
  printf("To int\n");
#endif /* ifdef DEBUGPROMOTION */
    return allocateCommonType((void*)&fromValue, INT);
    case CHAR:
#ifdef DEBUGPROMOTION
  printf("To char\n");
#endif /* ifdef DEBUGPROMOTION */
    char newchar = (char)(fromValue%256);
    printf("%d\n", fromValue);
    printf("%c\n",(char) fromValue);
    return allocateCommonType((long int*)&newchar, CHAR);
    case TUPLE:
#ifdef DEBUGPROMOTION
  printf("To tuple\n");
#endif /* ifdef DEBUGPROMOTION */
    return allocateCommonType((void*)&fromValue, TUPLE);
    case REAL:
#ifdef DEBUGPROMOTION
  printf("To real\n");
#endif /* ifdef DEBUGPROMOTION */
    float newfloat = (float)fromValue;
    return allocateCommonType((void*)&newfloat, REAL);
  }
}

commonType* charPromotion(char fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("promoting from char\n");
#endif /* ifdef DEBUGPROMOTION */
  switch (toType) {
      case BOOL:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case INT:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case CHAR:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case TUPLE:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case REAL:
      return allocateCommonType((long int*)&fromValue, BOOL);
    }
}

commonType* realPromotion(char fromValue, enum BuiltIn toType) {
#ifdef DEBUGPROMOTION
  printf("promoting from real\n");
#endif /* ifdef DEBUGPROMOTION */
  switch (toType) {
      case BOOL:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case INT:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case CHAR:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case TUPLE:
      return allocateCommonType((long int*)&fromValue, BOOL);
      case REAL:
      return allocateCommonType((long int*)&fromValue, BOOL);
    }
}

// promote and return temporary
commonType* promotion(commonType* from, commonType* to) {
  switch (from->type) {
    case BOOL:
    return boolPromotion((bool)*from->value, to->type);
    case INT:
    return intPromotion((int)*from->value, to->type);
    case CHAR:
    return charPromotion((int)*from->value, to->type);
    case TUPLE:
    return realPromotion((float)*from->value, to->type);
    case REAL:
    return realPromotion((float)*from->value, to->type);
  }
}

commonType* performCommonTypeBinop(commonType* left, commonType* right, enum BINOP op) {
  commonType* promotedLeft;
  commonType* promotedRight;
  promotedLeft = promotion(left,right);
  promotedRight = promotion(left,right);

  return promotedLeft;
}




void fill(vecStruct *a, int lower, int upper) {
  memset(a->base, 0, sizeof(int) * a->sizeOf);
  for (int i = 0 ; i < a->sizeOf ; i++) {
    *(a->base + i) = lower + i;
  }
}

// (residual memory)
void zeroOut(vecStruct *a) {
  memset(a->base, 0, sizeof(int) * a->sizeOf);
}

int getTrueVectorSize(int a) {
  return (a > 0 ? a : 0);
}

void printVec(vecStruct *a) {
  printf("[");
  for (int i = 0 ; i < a->sizeOf ; i++) {
    printf("%d",*(a->base + i));
    if (i != a->sizeOf - 1) {
      printf(" ");
    }
  }
  printf("]\n");
}

vecStruct* allocateVector(int size) {
  vecStruct* newVector = (vecStruct*)malloc(sizeof(vecStruct));
  int* newList = (int*)malloc(sizeof(int) * size);

  newVector->sizeOf = size;
  newVector->base = newList;

  return newVector;
}

void deallocateVector(vecStruct* vector) {
  // we don't allocate every vector we keep track of
  // if we find a label that wasn't allocated, leave it so we don't blow our legs off.
  if (vector) {
    free(vector->base);
    free(vector);
  }
}


int getMaxSize(vecStruct *a, vecStruct *b) {
  if (a->sizeOf > b->sizeOf) {
    return a->sizeOf;
  } else {
    return b->sizeOf;
  }
}

int performBINOP(int l, int r, enum BINOP op) {
  switch (op) {
    case ADD:
      return l + r;
    case SUB:
      return l - r;
    case MULT:
      return l * r;
    case DIV:
      return l / (r == 0 ? 1 : r);
    case EQUAL:
      return l == r;
    case NEQUAL:
      return l != r;
    case GTHAN:
      return l > r;
    case LTHAN:
      return l < r;
  }
}

// this is sinful and ugly. I wish i was a better programmer.

void vectorToVectorBINOP(vecStruct *a, vecStruct *b, vecStruct *result, enum BINOP op) {
  // ternary stuff looks good tho
  for (int i = 0; i < result->sizeOf ; i ++) {
    result->base[i] = performBINOP((i < a->sizeOf ? a->base[i] : 0), (i < b->sizeOf ? b->base[i] : 0), op);
  }
}

void integerToVectorBINOP(int a, vecStruct *b, vecStruct *result, enum BINOP op) {
  for (int i = 0; i < result->sizeOf ; i ++) {
    result->base[i] = performBINOP(a, b->base[i], op);
  }
}

// SO CHEAP. I am angry
int vectorToIntegerIndex(vecStruct *a, int b)  {
  return (b < 0 || b >= a->sizeOf) ? 0 : a->base[b];
}

void vectorToIntegerBINOP(vecStruct *a, int b, vecStruct *result, enum BINOP op) {
  for (int i = 0; i < result->sizeOf ; i ++) {
    result->base[i] = performBINOP(vectorToIntegerIndex(a, i), b, op);
  }
}

void vectorToVectorIndex(vecStruct *a, vecStruct *b, vecStruct *result) {
  for (int i = 0; i < result->sizeOf ; i ++) {
    result->base[i] = vectorToIntegerIndex(a, vectorToIntegerIndex(b, i));
  }
}

void vectorStoreValueAtIndex(vecStruct *a, int index, int val) {
    a->base[index] = val;
}

int vectorLoadValueAtIndex(vecStruct *a, int index) {
    return a->base[index];
}
