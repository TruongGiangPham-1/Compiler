#ifndef RUNTIMEMEMORY
#define RUNTIMEMEMORY

#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct commonType {
  enum TYPE type; 
  void* value; 
} commonType;

typedef struct tuple {
  int size;
  int currentSize;
  commonType** values; // list of values
} tuple;

// allocate some memory for a new commontype
commonType* allocateCommonType(void* value, enum TYPE type);
tuple* allocateTuple(int size);
void appendTuple(tuple* tuple, commonType *value);
void extractAndAssignValue(void* value, commonType *dest);

// de-allocation. common types which are 'list' types hold an address to a list of common types
void deallocateTuple(tuple* tuple);
void deallocateCommonType(commonType* object);

commonType* copyCommonType(commonType* copyFrom);
tuple* copyTuple(tuple* copyFrom);

// copy the value that the common type is pointing to. KEY WORD COPY
void* copyValue(commonType* copyFrom);

void assignByReference(commonType* dest, commonType* from);

commonType* copyCommonType(commonType* copyFrom) {
  commonType* copy = (commonType*)malloc(sizeof(commonType));
  copy->type = copyFrom->type;
  copy->value = copyValue(copyFrom);
  return copy;
}

void assignByReference(commonType* dest, commonType* from) {
  dest->value = copyValue(from);
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

void* copyValue(commonType* copyFrom) {
   switch (copyFrom->type) {
    case INTEGER:
      {
        int* newIntVal = malloc(sizeof(int));
        *newIntVal = *(int*)copyFrom->value;
        return newIntVal;
      }
      break;
    case BOOLEAN:
      {
        bool* newBoolVal = malloc(sizeof(bool));
        *newBoolVal = *(bool*)copyFrom->value;
        return newBoolVal;
      }
      break;
    case REAL:
      {
        float* newFloatVal = malloc(sizeof(float));
        *newFloatVal = *(float*)copyFrom->value;
        return newFloatVal;
      }
      break;
    case TUPLE:
      {
        return copyTuple((tuple*)copyFrom->value);
      }
      break;
    case CHAR: 
      {
        char* newCharVal = malloc(sizeof(char));
        *newCharVal = *(char*)copyFrom->value;
        return newCharVal;
      }
      break;
  } 
}

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

#endif
