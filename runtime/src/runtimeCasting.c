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
commonType* cast(commonType* from, commonType* toType);
commonType* promotion(commonType* from, commonType* to);
bool isCompositeType(enum TYPE type);
void printCommonType(commonType *type);
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
commonType* nullFrom(commonType* value) {
  switch(value->type) {
    case INTEGER:
    {
      int temp = 0;
      return allocateCommonType(&temp, INTEGER);
    }
    case CHAR:
    {
      char temp = 0x0;
      return allocateCommonType(&temp, CHAR);
    }
    case BOOLEAN:
    {
      bool temp = false;
      return allocateCommonType(&temp, BOOLEAN);
    }
    case REAL:
    {
      float temp = 0.0;
      return allocateCommonType(&temp, REAL);
    }
    case VECTOR:
    case STRING:
    {
      list* oldList = ((list*)value->value);
      list* nulledList = allocateList(oldList->currentSize);
      for (int i = 0 ; i < oldList->currentSize ; i ++) {
        appendList(nulledList, nullFrom(oldList->values[i]));
      }
      
      return allocateCommonType(&nulledList, VECTOR);
    }
    default: 
    RuntimeOPError("attempting to get null intialized value of unknown type");
    return NULL;
  }

}

commonType* castHelper(commonType* fromValue, enum TYPE type) {
  switch (type) {
    case INTEGER:
      {
      int temp = 0;
      return cast(fromValue, allocateCommonType(&temp, INTEGER));
      }
    case CHAR:
      {
      char temp = 0;
      return cast(fromValue, allocateCommonType(&temp, CHAR));
      }
    case REAL:
      {
      float temp = 0.0;
      return cast(fromValue, allocateCommonType(&temp, REAL));
      }
    case BOOLEAN:
      {
      bool temp = false;
      return cast(fromValue, allocateCommonType(&temp, BOOLEAN));
      }
    default:
    RuntimeOPError("something went wrong");
    return NULL;
  }
}

commonType* boolCast(bool fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Cast from bool\n");
#endif /* ifdef DEBUGTYPES */
  switch (toType->type) {
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
    case VECTOR :
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);

      for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, boolCast(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    CastError("Invalid cast from bool, type not recognized or implemented");
  }
}

commonType* intCast(int fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Cast from int\n");
#endif /* ifdef DEBUGTYPES */
  
  switch (toType->type) {
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
    case VECTOR:
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);

      for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, intCast(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    CastError("Invalid cast from int, type not recognized or implemented");
  }
}

commonType* charCast(char fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Cast from char\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType->type) {
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
    case VECTOR:
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);

      for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, charCast(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    CastError("Invalid cast from char, type not recognized or implemented");
  }
}

commonType* realCast(float fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Cast from real\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType->type) {
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
    case VECTOR:
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);
      
        for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, realCast(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
  #ifdef DEBUGTYPES
    printf("Cast fail!\n");
  #endif /* ifdef DEBUGTYPES */
    CastError("Invalid cast from real, type not recognized or implemented");
    return NULL;
  }
}

commonType* vectorCast(list* fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Cast from vector\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType->type) {
    // vectors only castable to vectors. Otherwise something dangerous is going on. time to do a little manipulation
    case STRING:
    case VECTOR: 
    {
      list* toList = ((list*)toType->value);
      int toSize = toList->currentSize;
      list* newList = allocateList(toSize);
      for (int i = 0 ; i < toSize ; i ++) {
          commonType* promotedItem;
          if (i < fromValue->currentSize) {
            promotedItem = cast(fromValue->values[i], toList->values[i]);
          } else { 
            // padding
            promotedItem = nullFrom(toList->values[i]);
          }
          appendList(newList, promotedItem);
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
    RuntimeOPError("wierd stuff goin on man");
  }
}

commonType* vectorPromotion(list* from, commonType* to) {
  if (isCompositeType(to->type)) {
    list* toList = (list*)to->value;

    if (from->currentSize > toList->currentSize) {
      SizeError("Promoting down, possible data loss");
    } else {
      return vectorCast(from, to);
    }
  } else {
    list* newList = allocateList(from->currentSize);
    for (int i = 0 ; i < from->currentSize ; i++) {
      appendList(newList, promotion(from->values[i], to));
    }

   return allocateCommonType(&newList, VECTOR);
  }
  
  return NULL;
}
/*
 *
 */

// vector promotion

commonType* cast(commonType* from, commonType* toType) {
  if (!ValidType(toType->type)) {
    UnsupportedTypeError("Cast recieved a type it could not recognize");
  }

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

      case VECTOR:
      case STRING:

      return vectorCast((list*)from->value, toType);

      default:

#ifdef DEBUGTYPES
      printf("Error! Uncastable type!\n");
#endif /* ifdef DEBUGTYPES */
      CastError("Invalid cast, type not recognized or implemented");
      return NULL;
  }
}

commonType* boolPromotion(commonType* fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Promotion from bool\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType->type) {
  case BOOLEAN:
#ifdef DEBUGTYPES
  printf("To bool!\n");
#endif /* ifdef DEBUGTYPES */
  return cast(fromValue, toType);

  case VECTOR:
  {
    list* toList = (list*)toType->value;
    list* newList = allocateList(toList->currentSize);
    
      for (int i = 0 ; i < toList->currentSize ; i++) {
      appendList(newList, boolPromotion(fromValue, toList->values[i]));
    }

    return allocateCommonType(&newList, VECTOR);
  }
  default:
#ifdef DEBUGTYPES
  printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
  PromotionError("Invalid promotion from bool");
  return NULL;
  }
}

commonType* intPromotion(commonType* fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Promotion from int\n");
#endif /* ifdef DEBUGTYPES */

  switch (toType->type) {
    case REAL:
#ifdef DEBUGTYPES
  printf("To real\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, toType);

    case INTEGER:
#ifdef DEBUGTYPES
  printf("To int\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, fromValue);

    case VECTOR:
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);
      
        for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, intPromotion(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
#ifdef DEBUGTYPES
  printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
    PromotionError("Invalid promotion from int");
    return NULL;
  }
}

commonType* charPromotion(commonType* fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Promotion from char\n");
#endif /* ifdef DEBUGTYPES */

    switch (toType->type) {
    case CHAR:
#ifdef DEBUGTYPES
    printf("To char\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, toType);

    case VECTOR:
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);
      
        for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, charPromotion(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
    default:
#ifdef DEBUGTYPES
    printf("Error! Promotion not possible\n");
#endif /* ifdef DEBUGTYPES */
    PromotionError("Invalid promotion from char");
    return NULL;
  }
}

commonType* realPromotion(commonType* fromValue, commonType* toType) {
#ifdef DEBUGTYPES
  printf("Promotion from real\n");
#endif /* ifdef DEBUGTYPES */
    switch (toType->type) {
    case REAL:
#ifdef DEBUGTYPES
  printf("To real\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, toType);
    case INTEGER:
#ifdef DEBUGTYPES
  printf("To real\n");
#endif /* ifdef DEBUGTYPES */
    return cast(fromValue, fromValue);

    case VECTOR:
    {
      list* toList = (list*)toType->value;
      list* newList = allocateList(toList->currentSize);
      
        for (int i = 0 ; i < toList->currentSize ; i++) {
        appendList(newList, realPromotion(fromValue, toList->values[i]));
      }

      return allocateCommonType(&newList, VECTOR);
    }
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
    return boolPromotion(from, to);
    case INTEGER:
    return intPromotion(from, to);
    case CHAR:
    return charPromotion(from, to);
    case REAL:
    return realPromotion(from, to);
    case VECTOR:
    return vectorPromotion((list*)from->value, to);
    default:
    PromotionError("Attempting promotion on invalid or tuple type");
    return NULL;
  }
}
