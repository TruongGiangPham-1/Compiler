#ifndef RUNTIMEMEMORY
#define RUNTIMEMEMORY

#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
#include "run_time_errors.h"
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

typedef struct list {
  int size;
  int currentSize;
  commonType** values; // list of values
} list;
commonType* promotion(commonType* from, commonType* to);
void printCommonType(commonType *type);
// allocate some memory for a new commontype
commonType* allocateCommonType(void* value, enum TYPE type);
commonType* allocateListOfSize(int size);
list* allocateList(int size);
list* allocateListFromCommon(commonType* size);
void appendList(list* list, commonType *value);
void appendCommon(commonType* list, commonType *value);
void extractAndAssignValue(void* value, commonType *dest);

// de-allocation. common types which are 'list' types hold an address to a list of common types
void deallocateList(list* list);
void deallocateCommonType(commonType* object);

commonType* copyCommonType(commonType* copyFrom);
list* copyList(list* copyFrom);

// copy the value that the common type is pointing to. KEY WORD COPY
void* copyValue(commonType* copyFrom);

void assignByReference(commonType* dest, commonType* from);

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

commonType* copyCommonType(commonType* copyFrom) {
  commonType* copy = (commonType*)malloc(sizeof(commonType));
  copy->type = copyFrom->type;
  copy->value = copyValue(copyFrom);
  return copy;
}

void assignByReference(commonType* dest, commonType* from) {
  commonType* promotedVal = promotion(from, dest);
  if (isCompositeType(dest->type)) {
    deallocateList(dest->value);
  } else {
    free(dest->value);
  }

  dest->value = copyValue(promotedVal);
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
    case VECTOR:
    case MATRIX:
    case STRING:
      {
        return copyList((list*)copyFrom->value);
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
    case STRING:
    case VECTOR:
    case MATRIX:
      {
        dest->value = *(list**)value;
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

void deallocateList(list* list) {
#ifdef DEBUGMEMORY
  printf("Deallocating List at %p...\n", list);
  printf("=== LIST ===\n");
#endif /* ifdef DEBUGMEMORY */

    free(list->values);
    free(list);

#ifdef DEBUGMEMORY
  printf("=== LIST ===\n");
  printf("List deallocation success!\n");
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
      case VECTOR:
      case MATRIX:
      case STRING:
      deallocateList((list*)object->value);
      break;
      default:
      free(object->value);
    }
  }

#ifdef DEBUGMEMORY
  printf("Commontype deallocation success!\n");
#endif /* ifdef DEBUGMEMORY */
}

list* allocateList(int size) {
  list* newList = (list*) malloc(sizeof(list));
  commonType** valueList = (commonType**) calloc(size, sizeof(commonType*));

  newList->size = size;
  newList->currentSize = 0;
  newList->values= valueList;

#ifdef DEBUGMEMORY
  printf("Allocated tuple at %p\n", newList);
  printf("Tuple list beings at %p\n", valueList);
#endif
  
  return newList;
};

list* allocateListFromCommon(commonType* size) {
  return allocateList(*(int*)size->value);
}

/**
 * Add item to our tuple, we can potentially go over bounds....
 * but we shoulnd't due to spec, right?
 */
void appendList(list* list, commonType *value) {
#ifdef DEBUGTUPLE
  printf("====== Appending to list\n");
  printf("List currently holding %p  at index %d address %p\n", tuple->values[tuple->currentSize], tuple->currentSize, &tuple->values[tuple->currentSize]);
#endif

  if (list->currentSize +1 > list->size) {
    printCommonType(value);
    RuntimeOPError("Writing past array");
  }

  list->values[list->currentSize] = value;

#ifdef DEBUGTUPLE
  printf("appended to list at %p, %p\n", &list->values[list->currentSize], value);
  printf("List now holding %p  at index %d address %p\n", list->values[list->currentSize], list->currentSize, &list->values[list->currentSize]);
  printf("====== Append complete\n"); 
#endif /* ifdef DEBUGTUPLE */

  list->currentSize++;
}

void appendCommon(commonType* list, commonType *value) {
  appendList(list->value, copyCommonType(value));
}

list* copyList(list* copyFrom) {
  list* copiedTo = allocateList(copyFrom->size);

  for (int i = 0 ; i < copyFrom->currentSize; i ++) {
    commonType* newVal = copyCommonType(copyFrom->values[i]);
    appendList(copiedTo, newVal);
  }

  return copiedTo;
}
#endif
