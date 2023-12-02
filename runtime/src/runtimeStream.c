#include <errno.h>
#include <limits.h>

#define DEBUGSTREAM
//#define DEBUGPRINT

// global variable streamBuffer for streamIn
// this is a buffer for the rewind feature
#define MAX_REWIND_BUFFER_SIZE 1024
char STREAM_REWIND_BUFFER[MAX_REWIND_BUFFER_SIZE] = {0};
int BUF_HEAD = 0;
int BUF_TAIL = 0;
bool LAST_STREAMIN_ERR = false;

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
// write STREAM_REWIND_BUFFER into a zero-indexed newbuf
void normalizeRewindBuffer(char newbuf[1024]);
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
}


void streamIn(commonType *type, int* streamStatePtr) {
#ifdef DEBUGSTREAM
    char buf[1024] = {0};
    normalizeRewindBuffer(buf);
    printf("streamIn with curr buf '%s'\n", buf);
#endif /* ifdef DEBUGSTREAM */

    if (!LAST_STREAMIN_ERR) {
#ifdef DEBUGSTREAM
        printf("Last streamIn was successful, reading from stdin\n");
#endif /* ifdef DEBUGSTREAM */

        // if the last streamIn was successful, read from stdin
        // otherwise, we're just going to read from the rewind buffer
        readToBuf();
    }

    // now, read from a value from the buffer
    enum StreamState newStreamState = readFromBuf(type);
    handleStreamState(streamStatePtr, newStreamState, type);

    // finally, set the last streamIn error
    if (newStreamState == STREAM_STATE_ERR) {
        LAST_STREAMIN_ERR = true;
    } else {
        LAST_STREAMIN_ERR = false;
    }
}

// TODO: handle buffer rewind size.
void readToBuf() {
    // read individual chars from stdin into the rewind buffer
    // keep leading whitespace until we hit a non-whitespace char
    // after that, read until we hit a whitespace char
    char c;
    bool leadingWhitespace = true;

    // debug vals
    int whitespaceCount = 0;
    int nonWhitespaceCount = 0;

    // read until we hit a non-whitespace char
    while (true) {
        c = getchar();
        if (c == EOF) {
            // if we hit EOF, we're done
            return;
        } else if (c == ' ' || c == '\t' || c == '\n') {
            // if we hit whitespace, keep reading
            pushToBuf(c);

            // debug
            whitespaceCount++;
        } else {
            // if we hit a non-whitespace char, stop reading
            break;
        }
    }

    // read until we hit a whitespace char
    while (c != ' ' && c != '\t' && c != '\n') {
        // if we hit EOF, we're done
        if (c == EOF) {
            return;
        }

        // otherwise, read the char
        pushToBuf(c);
        c = getchar();

        // debug
        nonWhitespaceCount++;
    }

    // add trailing whitespace to buffer
    pushToBuf(c);

#ifdef DEBUGSTREAM
    printf("Read %d non-whitespace chars and %d whitespace chars\n", nonWhitespaceCount, whitespaceCount);
    printf("Buffer is now '%s'\n", STREAM_REWIND_BUFFER);
#endif /* ifdef DEBUGSTREAM */
}

void pushToBuf(char c) {
    STREAM_REWIND_BUFFER[BUF_TAIL] = c;
    BUF_TAIL = (BUF_TAIL + 1) % MAX_REWIND_BUFFER_SIZE;
}

void normalizeRewindBuffer(char newBuf[1024]) {
    // write STREAM_REWIND_BUFFER into a zero-indexed newbuf
    // this is so we can use strtol and other functions on it
    int i = BUF_HEAD;
    int j = 0;
    while (i != BUF_TAIL) {
        newBuf[j] = STREAM_REWIND_BUFFER[i];
        i = (i + 1) % MAX_REWIND_BUFFER_SIZE;
        j++;
    }
    newBuf[j] = '\0';

}

// WARNING: this function is very ugly
enum StreamState readFromBuf(commonType* type) {
    // assume STREAM_REWIND_BUFFER is populated (so not EOF)
    // now, read a value from the buffer and store it into the *type

    // since the buffer is circular, we first have to "normalize" it into a zero-indexed array
    char buf[2014];
    normalizeRewindBuffer(buf);

#ifdef DEBUGSTREAM
    printf("readFromBuf: buf = '%s'\n", buf);
#endif /* ifdef DEBUGSTREAM */

    switch (type->type) {
        case INTEGER: {
            // convert string to an int
            // https://stackoverflow.com/a/18544436
            long lnum;
            char *end;
            errno = 0;

            lnum = strtol(buf, &end, 10);        //10 specifies base-10
            if (end == buf) {
                // no digits consumed
#ifdef DEBUGSTREAM
                printf("ERROR (int): no digits were found\n");
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_ERR;
            } else if (*end != '\0') {
                // extra characters at the end
#ifdef DEBUGSTREAM
                printf("ERROR (int): extra characters at the end\n");
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_ERR;
            } else if (((lnum == LONG_MAX || lnum == LONG_MIN) && errno == ERANGE) ||
                       (lnum > INT_MAX) || (lnum < INT_MIN)) {
                // number is out of range
#ifdef DEBUGSTREAM
                printf("ERROR (int): input out of range");
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_ERR;
            } else {
#ifdef DEBUGSTREAM
                printf("OK (int): Successful int read: %d\n", (int) lnum);
#endif /* ifdef DEBUGSTREAM */

                // number is valid
                *(int *) type->value = (int) lnum;
                return STREAM_STATE_OK;
            }
            break;
        }
        case CHAR: {
            // CHAR CAN NEVER FAIL (except if it's an end of file)
            // BUT: we assume the buffer is nonempty so this must succeed.
            *(char*)type->value = buf[0];
#ifdef DEBUGSTREAM
            printf("OK (char): Scanned '%c'\n", *(char*)type->value);
#endif /* ifdef DEBUGSTREAM */
            break;
        }
        case BOOLEAN: {
//      printf("Enter a boolean value (T/F): ");
            // scan char. If it's T, true, else false
            if (strcmp(buf, "T") == 0) {
                *(bool*)type->value = true;
            } else if (strcmp(buf, "F") == 0) {
                *(bool *) type->value = false;
            } else {
#ifdef DEBUGSTREAM
                printf("ERROR (bool): Invalid boolean value '%s'\n", buf);
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_ERR;
            }
#ifdef DEBUGSTREAM
            printf("OK (bool): Scanned '%s'\n", *(bool*)type->value ? "T" : "F");
#endif /* ifdef DEBUGSTREAM */
            return STREAM_STATE_OK;
            break;
        }
        case REAL: {
            // convert string to a float
            // https://stackoverflow.com/a/18544436
            char *end;
            errno = 0;
            float fnum = strtof(buf, &end);        //10 specifies base-10
            if (*end != '\0') {
                // extra characters at the end
#ifdef DEBUGSTREAM
                printf("ERROR (real): extra characters at the end\n");
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_ERR;
            } else if (errno == ERANGE) {
                // number is out of range
#ifdef DEBUGSTREAM
                printf("ERROR (real): input out of range");
#endif /* ifdef DEBUGSTREAM */
                return STREAM_STATE_ERR;
            } else {
#ifdef DEBUGSTREAM
                printf("OK (real): Successful real read: %g\n", fnum);
#endif /* ifdef DEBUGSTREAM */
                // number is valid
                *(float *) type->value = fnum;
                return STREAM_STATE_OK;
            }
            break;
        }
    }
}