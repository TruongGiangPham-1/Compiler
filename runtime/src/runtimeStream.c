#include <errno.h>
#include <limits.h>

//#define DEBUGSTREAM
//#define DEBUGPRINT

// global variable streamBuffer for streamIn
#define MAX_REWIND_BUFFER_SIZE 1024
char STREAM_BUF[MAX_REWIND_BUFFER_SIZE] = {0};
int BUFFER_PTR = 0;

enum StreamState {
    STREAM_STATE_OK = 0,
    STREAM_STATE_ERR = 1,
    STREAM_STATE_EOF = 2,
};

// read individual chars from stdin into the rewind buffer
// keep leading whitespace and read until we hit a non-whitespace char
void readToBuf();
// read from the rewind buffer into the corresponding type.
enum StreamState readFromBuf(commonType* type);
void pushToBuf(char c);
// push the unread chars back into stdin with ungetc
void resetBuf(int charsRead);
// helper function
bool whitespaceOrEOF(char c);

void printType(commonType *type, bool nl) {
    switch (type->type) {
        case INTEGER:
#ifdef DEBUGPRINT
            printf("\nPRINTING INTEGER\n");
#endif /* ifdef DEBUGPRINT */
            printf("%d", *(int*)type->value);
            break;
        case CHAR:
#ifdef DEBUGPRINT
            printf("\nPRINTING CHAR\n");
#endif /* ifdef DEBUGPRINT */
            printf("%c", *(char*)type->value);
            break;
        case BOOLEAN:
#ifdef DEBUGPRINT
            printf("\nPRINTING BOOL:\n");
#endif /* ifdef DEBUGPRINT */
            printf("%s", *(bool*)type->value ? "T" : "F");
            break;
        case REAL:
#ifdef DEBUGPRINT
            printf("\nPRINTING REAL\n");
#endif /* ifdef DEBUGPRINT */
            printf("%g", *(float*)type->value);
            break;
        case TUPLE:
            // tuple is just for debug
            // we don't disambiguate. similar behavior
        case VECTOR:
        case MATRIX:
        case STRING:
        {
            list* mListable = ((list*)type->value);

            if (type->type != STRING) printf("[");

            for (int i = 0 ; i < mListable->currentSize; i++) {

                printType(mListable->values[i], false);
                if (i != mListable->currentSize-1 && type->type != STRING) printf(" ");
            }
            if (type->type != STRING) printf("]");
        }
            break;
        default:
            UnsupportedTypeError("Attempting to print a type not recognized by the backend (this happens when type enums are bad)");
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

void handleStreamState(int* state, int newState, commonType *type) {
#ifdef DEBUGSTREAM
    printf("Setting streamState to %d\n", newState);
#endif /* ifdef DEBUGSTREAM */

    // given the streamState error and the type, set the streamState
    if (type->type == CHAR) {
        // the only possible error for a char is EOF, where we set streamState to 0
        *state = 0;
        return;
    } else {
        // in all other cases, set the state to the integer value of the streamStateErr
        *state = newState;
    }

    // set the value of *type->value to the null value if there was an error
    if (newState == STREAM_STATE_ERR || newState == STREAM_STATE_EOF) {
        setToNullValue(type);
    }
}


void streamIn(commonType *type, int* streamStatePtr) {
    readToBuf();

    if (strlen(STREAM_BUF) == 0) {
        // if we read nothing, we hit EOF
        handleStreamState(streamStatePtr, STREAM_STATE_EOF, type);
        return;
    }

    // now, read from a value from the buffer
    enum StreamState newStreamState = readFromBuf(type);
    handleStreamState(streamStatePtr, newStreamState, type);
}

void readToBuf() {
    // read up to 1024 chars into the buffer
    // stop when we hit EOF or until buffer is full

    for (int i = 0; i < MAX_REWIND_BUFFER_SIZE; i++) {
        char c = getchar();
        if (c == EOF) {
            // if we hit EOF, we're done
            STREAM_BUF[i] = '\0';
            break;
        } else {
            // otherwise, read the char
            pushToBuf(c);
        }
    }
#ifdef DEBUGSTREAM
    printf("Read %lu chars into buffer\n", strlen(STREAM_BUF));
    printf("Buffer is now '%s'\n", STREAM_BUF);
#endif /* ifdef DEBUGSTREAM */
}

void pushToBuf(char c) {
    STREAM_BUF[BUFFER_PTR] = c;
    BUFFER_PTR++;
}

// WARNING: this function is very ugly
enum StreamState readFromBuf(commonType* type) {
    // assume STREAM_REWIND_BUFFER is populated (so not EOF)
    // now, read a value from the buffer and store it into the *type

#ifdef DEBUGSTREAM
    printf("readFromBuf: STREAM_BUF = '%s'\n", STREAM_BUF);
#endif /* ifdef DEBUGSTREAM */

    int check = 0;
    int charsRead = 0;

    switch (type->type) {
        case INTEGER: {
            int n;
            check = sscanf(STREAM_BUF, "%d%n", &n, &charsRead);
            if (check != 1) {
                // if we didn't read anything, we hit EOF
#ifdef DEBUGSTREAM
                printf("ERROR (int): didn't successfully read an int (check = %d)\n", check);
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_EOF;
                resetBuf(0);
            } else if (!whitespaceOrEOF(STREAM_BUF[charsRead])) {
                // if we didn't read a valid ending, we hit an error
#ifdef DEBUGSTREAM
                printf("ERROR (int): didn't successfully read an int\n");
#endif /* ifdef DEBUGSTREAM */
                resetBuf(0);
                return STREAM_STATE_ERR;
            } else {
                // success
#ifdef DEBUGSTREAM
                printf("OK (int): Scanned '%d', reading '%d' chars\n", n, charsRead);
#endif /* ifdef DEBUGSTREAM */
                *(int*)type->value = n;
                resetBuf(charsRead);
                return STREAM_STATE_OK;
            }
        }
        case CHAR: {
            // CHAR CAN NEVER FAIL (except if it's an end of file)
            // BUT: we assume the buffer is nonempty so this must succeed.
            *(char*)type->value = STREAM_BUF[0];
#ifdef DEBUGSTREAM
            printf("OK (char): Scanned '%c'\n", *(char*)type->value);
#endif /* ifdef DEBUGSTREAM */
            resetBuf(1);
            return STREAM_STATE_OK;
        }
        case BOOLEAN: {
            // scan char. If it's T, true, else false
            char untilWhitespace[1024]; // new char array to scan a string into
            check = sscanf(STREAM_BUF, "%s%n", &untilWhitespace, &charsRead);
            if (strcmp(untilWhitespace, "T") == 0) {
                *(bool*)type->value = true;
            } else if (strcmp(untilWhitespace, "F") == 0) {
                *(bool*) type->value = false;
            } else {
#ifdef DEBUGSTREAM
                printf("ERROR (bool): Invalid boolean value '%s'\n", STREAM_BUF);
#endif /* ifdef DEBUGSTREAM */
                resetBuf(0);
                return STREAM_STATE_ERR;
            }
#ifdef DEBUGSTREAM
            printf("OK (bool): Scanned '%s'\n", *(bool*)type->value ? "T" : "F");
#endif /* ifdef DEBUGSTREAM */
            resetBuf(1);
            return STREAM_STATE_OK;
        }
        case REAL: {
            float f;
            check = sscanf(STREAM_BUF, "%f%n", &f, &charsRead);
            if (check != 1) {
                // if we didn't read anything, we hit EOF
#ifdef DEBUGSTREAM
                printf("ERROR (real): didn't successfully read a real (check = %d)\n", check);
#endif /* ifdef DEBUGSTREAM */
                resetBuf(0);
                return STREAM_STATE_EOF;
            } else if (!whitespaceOrEOF(STREAM_BUF[charsRead])) {
                // if we didn't read a valid ending, we hit an error
#ifdef DEBUGSTREAM
                printf("ERROR (real): didn't successfully read a real\n");
#endif /* ifdef DEBUGSTREAM */
                resetBuf(0);
                return STREAM_STATE_ERR;
            } else {
                // success
#ifdef DEBUGSTREAM
                printf("OK (real): Scanned '%f', with '%d' chars\n", f, charsRead);
#endif /* ifdef DEBUGSTREAM */
                *(float*)type->value = f;
                resetBuf(charsRead);
                return STREAM_STATE_OK;
            }
        }
        default:
            UnsupportedTypeError("Attempting to read a type not recognized by the backend (this happens when type enums are bad)");
            return STREAM_STATE_ERR;
    }
}

void resetBuf(int charsRead) {
    // push the unread chars back into stdin with ungetc
    // also reset the buffer to zero
#ifdef DEBUGSTREAM
    printf("resetBuf: pushing %d char(s) back into stdin: '%s'\n", BUFFER_PTR-charsRead, STREAM_BUF+charsRead);
#endif /* ifdef DEBUGSTREAM */
    for (int i = BUFFER_PTR - 1; i >= charsRead; i--) {
#ifdef DEBUGSTREAM
        printf("ungetting '%c'\n", STREAM_BUF[i]);
#endif /* ifdef DEBUGSTREAM */
        ungetc(STREAM_BUF[i], stdin);
    }
    STREAM_BUF[0] = '\0';
    BUFFER_PTR = 0;
}

bool whitespaceOrEOF(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == EOF;
}
