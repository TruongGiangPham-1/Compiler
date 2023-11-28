#include "Operands/BINOP.h"
#include "Operands/UNARYOP.h"
#include "Types/TYPES.h"
#include "run_time_errors.h"
#include "runtimeMemory.c"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

bool isComparison(enum BINOP op) {
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

bool ValidType(enum TYPE type) {
  switch (type) {
    case INTEGER:
    case CHAR:
    case REAL:
    case BOOLEAN:
    case TUPLE:
    case STRING:
    case VECTOR:
    case MATRIX:
    return true;
    default:
    return false;
  }
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
    CastError("Invalid cast from bool, type not recognized or implemented");
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
    CastError("Invalid cast from int, type not recognized or implemented");
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
    CastError("Invalid cast from char, type not recognized or implemented");
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
    CastError("Invalid cast from real, type not recognized or implemented");
    return NULL;
  }
}
commonType* cast(commonType* from, enum TYPE toType);
/*
 *
 */
commonType* listCast(commonType* from, enum TYPE toType) {
  list* mlist = (list*)from->value;

  for (int i = 0 ; i < mlist->size ; i++) {
    commonType* casted = cast(mlist->values[i], toType);
    appendList(mlist, casted);
  }

  return allocateCommonType(mlist, VECTOR);
}

commonType* cast(commonType* from, enum TYPE toType) {
  if (!ValidType(toType)) {
    UnsupportedTypeError("Cast recieved a type it could not recognize");
  }

  switch (from->type) {
    case VECTOR:
    return listCast(from, toType);
    default:
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
      CastError("Invalid cast, type not recognized or implemented");
      return NULL;
    }


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
  PromotionError("Invalid promotion from bool");
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
    PromotionError("Invalid promotion from int");
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
    PromotionError("Invalid promotion from char");
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
    PromotionError("Invalid promotion from real");
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
    case REAL:
    return realPromotion(from, to->type);
    default:
    PromotionError("Attempting promotion on invalid or tuple type");
    return NULL;
  }
}
