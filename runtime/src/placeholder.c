#include "BINOP.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void print(int i) {
  printf("%d\n", i);
}

typedef struct vecStruct {
  int* base;
  int sizeOf;
} vecStruct;

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
