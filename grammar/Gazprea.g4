grammar Gazprea;

file
    : (globalDecl | typedef | procedure | function)* EOF
    ;

globalDecl
    : RESERVED_CONST (type)? ID '=' expression ';'
    ;

vardecl
    : qualifier? inferred_sized_type ID '=' expression ';'  #inferred_size
    | qualifier? known_sized_type ID ('=' expression)? ';'  #sized
    | qualifier ID '=' expression ';'                       #inferred_type
    ;

assign
    : lvalue '=' expression ';'
    ;

lvalue
    : ID                                            #variable
    | tuple_index                                   #tupleVariable
    | (ID | tuple_index) (',' (ID | tuple_index))+  #tupleUnpack
    | ID '[' expression ']'                         #vectorIndex
    | ID '[' expression ',' expression ']'          #matrixIndex
    | tuple_index '[' expression ']'                #tupleVectorIndex
    | tuple_index '[' expression ',' expression ']' #tupleMatrixIndex
    ;

non_decl
    : assign | cond | loop | break | continue | return | stream | procedure_call
    ;

block
    : '{' (vardecl)* (non_decl | block)* '}'
    | non_decl
    ;

cond
    : RESERVED_IF '(' expression ')' block (RESERVED_ELSE RESERVED_IF '(' expression ')' block)* (RESERVED_ELSE block)?
    ;

loop
    : RESERVED_LOOP block                                                               #infiniteLoop
    | RESERVED_LOOP RESERVED_WHILE '(' expression ')' block                             #predicatedLoop
    | RESERVED_LOOP ID RESERVED_IN expression (',' ID RESERVED_IN expression)* block    #iteratorLoop
    | RESERVED_LOOP block RESERVED_WHILE '(' expression ')' ';'                         #postPredicatedLoop
    ;

break
    : RESERVED_BREAK ';'
    ;

continue
    : RESERVED_CONTINUE ';'
    ;

return
    : RESERVED_RETURN expression? ';'
    ;

function
    : RESERVED_FUNCTION ID '(' (type ID (',' type ID)*)? ')' RESERVED_RETURNS type '=' expression ';' #functionSingle
    | RESERVED_FUNCTION ID '(' (type ID (',' type ID)*)? ')' RESERVED_RETURNS type block              #functionBlock
    | RESERVED_FUNCTION ID '(' (type ID (',' type ID)*)? ')' RESERVED_RETURNS type ';'                #functionForward
    ;

procedure
    : RESERVED_PROCEDURE ID '(' (qualifier? type ID (',' qualifier? type ID)*)? ')' (RESERVED_RETURNS type)? block  #procedureBlock
    | RESERVED_PROCEDURE ID '(' (qualifier? type ID (',' qualifier? type ID)*)? ')' (RESERVED_RETURNS type)? ';'    #procedureForward
    ;

procedure_call
    : RESERVED_CALL ID '(' (expression (',' expression)*)? ')' ';'
    | RESERVED_CALL RESERVED_STREAM_STATE '(' RESERVED_STD_INPUT ')' ';' // Since in built stream_state() is a procedure defined in Gazprea
    ;

function_call
    : ID '(' (expression (',' expression)*)? ')' // no semicolon for functions because they always return and hence can be used as an expression
    | (RESERVED_LENGTH | RESERVED_ROWS | RESERVED_COLUMNS | RESERVED_REVERSE | RESERVED_FORMAT) '(' expression ')'
    | RESERVED_STREAM_STATE '(' RESERVED_STD_INPUT ')'
    ;

type: known_sized_type | inferred_sized_type;
tuple_allowed_type: built_in_type | vector_type | string_type | matrix_type | inferred_sized_type;

known_sized_type: built_in_type | tuple_type | vector_type | string_type | matrix_type;
inferred_sized_type
    : built_in_type '[' MULT ']'                #vector
    | RESERVED_STRING '[' MULT ']'              #string
    | built_in_type '[' MULT ',' expression ']' #matrixFirst
    | built_in_type '[' expression ',' MULT ']' #matrixSecond
    | built_in_type '[' MULT ',' MULT ']'       #matrix
    ;

qualifier: RESERVED_CONST | RESERVED_VAR;
built_in_type: RESERVED_BOOLEAN | RESERVED_CHARACTER | RESERVED_INTEGER | RESERVED_REAL | ID; // ID incase of typedefs... This might be changed
tuple_type: RESERVED_TUPLE '(' tuple_allowed_type ID? (',' tuple_allowed_type ID?)+ ')';
vector_type: built_in_type '[' expression ']';
string_type: RESERVED_STRING ('[' expression ']')?;
matrix_type: built_in_type '[' expression ',' expression ']';

expression // root of an expression tree
    :   expr
    ;
expr
    : '(' expr ')'                                                                                      #parentheses
    | cast                                                                                              #typeCast
    | function_call                                                                                     #funcCall
    | expr '[' expr (',' expr)? ']'                                                                     #index
    | expr RANGE_OPERATOR expr                                                                          #range
    | '[' ID RESERVED_IN expression (',' ID RESERVED_IN expression)? GENERATOR_OPERATOR expression ']'  #generator
    | '[' ID RESERVED_IN expression FILTER_OPERATOR expression (',' expression)* ']'                    #filter
    | <assoc=right> op=(ADD | SUB | RESERVED_NOT) expr                                                  #unary
    | <assoc=right> expr op=EXP expr                                                                    #math
    | expr op=(MULT | DIV | REM | DOT_PRODUCT) expr                                                     #math
    | expr op=(ADD | SUB) expr                                                                          #math
    | expr RESERVED_BY expr                                                                             #stride
    | expr op=(LT | GT | LE | GE) expr                                                                  #cmp
    | expr op=(EQ | NEQ) expr                                                                           #cmp
    | expr RESERVED_AND expr                                                                            #binary
    | expr (RESERVED_OR | RESERVED_XOR) expr                                                            #binary
    | expr CONCAT expr                                                                                  #concatenation
    | RESERVED_IDENTITY                                                                                 #identity
    | RESERVED_NULL                                                                                     #null
    | LITERAL_BOOLEAN                                                                                   #literalBoolean
    | LITERAL_CHARACTER                                                                                 #literalCharacter
    | INT                                                                                               #literalInt
    | LITERAL_REAL                                                                                      #literalReal
    | literal_tuple                                                                                     #literalTuple
    | tuple_index                                                                                       #tupleIndex
    | literal_vector                                                                                    #literalVector
    | LITERAL_STRING                                                                                    #literalString
    | literal_matrix                                                                                    #literalMatrix
    | ID                                                                                                #literalID
    ;

literal_tuple: '(' expression (',' expression)+ ')';
literal_vector: '[' (expression (',' expression)*)? ']'; // empty vectors allowed
literal_matrix: '[' (literal_vector (',' literal_vector)*)? ']'; // empty matrices allowed
cast: RESERVED_AS LT known_sized_type GT '(' expression ')';
typedef: RESERVED_TYPEDEF type ID ';'; // inferred types allowed in typedefs
stream
    : expression RIGHT_ARROW RESERVED_STD_OUTPUT ';'        #outputStream
    | ID LEFT_ARROW RESERVED_STD_INPUT ';'                  #inputStream
    | tuple_index LEFT_ARROW RESERVED_STD_INPUT ';'         #inputStream
    ;
tuple_index: ID DOT (INT | ID);

// operators
MULT: '*';
DIV: '/';
ADD: '+';
SUB: '-';
LT: '<';
GT: '>';
EQ: '==';
NEQ: '!=';
REM: '%';
EXP: '^';
LE: '<=';
GE: '>=';
DOT: '.';
CONCAT: '||';
DOT_PRODUCT: '**';
RANGE_OPERATOR: '..';
FILTER_OPERATOR: '&';
GENERATOR_OPERATOR: '|';
RIGHT_ARROW: '->';
LEFT_ARROW: '<-';


// reserved keywords
RESERVED_AND: 'and';
RESERVED_AS: 'as';
RESERVED_BOOLEAN: 'boolean';
RESERVED_BREAK: 'break';
RESERVED_BY: 'by';
RESERVED_CALL: 'call';
RESERVED_CHARACTER: 'character';
RESERVED_COLUMNS: 'columns';
RESERVED_CONST: 'const';
RESERVED_CONTINUE: 'continue';
RESERVED_ELSE: 'else';
RESERVED_FALSE: 'false';
RESERVED_FORMAT: 'format';
RESERVED_FUNCTION: 'function';
RESERVED_IDENTITY: 'identity';
RESERVED_IF: 'if';
RESERVED_IN: 'in';
RESERVED_INTEGER: 'integer';
RESERVED_LENGTH: 'length';
RESERVED_LOOP : 'loop';
RESERVED_NOT: 'not';
RESERVED_NULL: 'null';
RESERVED_OR: 'or';
RESERVED_PROCEDURE: 'procedure';
RESERVED_REAL: 'real';
RESERVED_RETURN: 'return';
RESERVED_RETURNS: 'returns';
RESERVED_REVERSE: 'reverse';
RESERVED_ROWS: 'rows';
RESERVED_STD_INPUT: 'std_input';
RESERVED_STD_OUTPUT: 'std_output';
RESERVED_STREAM_STATE: 'stream_state';
RESERVED_STRING: 'string';
RESERVED_TRUE: 'true';
RESERVED_TUPLE: 'tuple';
RESERVED_TYPEDEF: 'typedef';
RESERVED_VAR: 'var';
RESERVED_WHILE: 'while';
RESERVED_XOR: 'xor';

ID : ('_' | ALPHABET) ('_' | ALPHABET | DIGIT)*;
INT : DIGIT+;

LITERAL_BOOLEAN: RESERVED_TRUE | RESERVED_FALSE;
LITERAL_CHARACTER: '\'' SCHAR '\'';
LITERAL_STRING: '"' (SCHAR+)? '"';
LITERAL_REAL
    : INT? DOT INT EXPONENT?
    | INT DOT? EXPONENT?
    ;
EXPONENT: ('e' | 'E') (SUB | ADD)? INT;

// Skip comments & whitespace
BLOCK_COMMENT : '/*' .*? '*/' -> skip ;
LINE_COMMENT : '//' ~[\r\n]* -> skip ;
WS : [ \t\r\n]+ -> skip;

fragment
DIGIT : [0-9];
ALPHABET : [a-zA-Z];
SCHAR : ('\\' [0abtnr"'\\] | ~["\\\r\n]);