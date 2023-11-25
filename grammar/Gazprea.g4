grammar Gazprea;

file
    : block EOF
    ;

block :
    statement*
    ;

statement:
    ( assign
    | vardecl
    | cond
    | loop
    | break
    | continue
    | return
    | stream
    | procedure
    | procedureCall
    | function
    | typedef
    | '{' block '}'
    );

vardecl
    : qualifier? type? ID ('=' expression)? ';'
    ;

assign
    : lvalue '=' rvalue=expression ';'
    ;

lvalue
    : expression (',' expression)*
    ;

cond
    : RESERVED_IF '(' expression ')' bodyStatement
    (RESERVED_ELSE RESERVED_IF '(' expression ')' bodyStatement )*
    (RESERVED_ELSE bodyStatement )?
    ;

loop
    : RESERVED_LOOP bodyStatement                                                               #infiniteLoop
    | RESERVED_LOOP RESERVED_WHILE '(' expression ')' bodyStatement                             #predicatedLoop
    | RESERVED_LOOP ID RESERVED_IN expression (',' ID RESERVED_IN expression)* bodyStatement    #iteratorLoop
    | RESERVED_LOOP bodyStatement RESERVED_WHILE '(' expression ')' ';'                         #postPredicatedLoop
    ;

// the body of a conditional or a loop could either be a braced block or a valid statement

bodyStatement
    : '{' block '}'
    | assign
    | break
    | continue
    | return
    | stream
    | cond
    | loop
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

funcParameter : type ID;
parameter: qualifier? type ID;

function
    : RESERVED_FUNCTION ID '(' (funcParameter (',' funcParameter)*)? ')' RESERVED_RETURNS type '=' expression ';'
    | RESERVED_FUNCTION ID '(' (funcParameter (',' funcParameter)*)? ')' RESERVED_RETURNS type '{' block '}'
    | RESERVED_FUNCTION ID '(' (funcParameter (',' funcParameter)*)? ')' RESERVED_RETURNS type ';'
    ;

procedure
    : RESERVED_PROCEDURE ID '(' (parameter (',' parameter)*)? ')' (RESERVED_RETURNS type)? '{' block '}'
    | RESERVED_PROCEDURE ID '(' (parameter (',' parameter)*)? ')' (RESERVED_RETURNS type)? ';'
    ;

procedureCall
    : RESERVED_CALL ID '(' (expression (',' expression)*)? ')' ';'
    // syntax error if someone calls these with different arguments.
    // | RESERVED_CALL RESERVED_STREAM_STATE '(' RESERVED_STD_INPUT ')' ';' // Since in built stream_state() is a procedure defined in Gazprea
    ;

functionCall
    : ID '(' (expression (',' expression)*)? ')' // no semicolon for functions because they always return and hence can be used as an expression
    // I'm commenting these out because we will get syntax errors if someone calls these with different arguments,
    //| (RESERVED_LENGTH | RESERVED_ROWS | RESERVED_COLUMNS | RESERVED_REVERSE | RESERVED_FORMAT) '(' expression ')'
    //| RESERVED_STREAM_STATE '(' RESERVED_STD_INPUT ')'
    ;

type
    : type '[' typeSize ']'                                         #vectorType
    | type '[' typeSize ',' typeSize ']'                            #matrixType
    | RESERVED_TUPLE '(' tupleTypeField (',' tupleTypeField)+ ')'   #tupleType
    | typeString=(ID | BUILT_IN_TYPE)                               #baseType
    ;

typeSize : expression | MULT;
tupleTypeField : type ID?;

qualifier: RESERVED_CONST | RESERVED_VAR;


string_type: RESERVED_STRING ('[' expression ']')?;

expression // root of an expression tree
    : expr
    ;

expr
    : '(' expr ')'                                                                                      #parentheses
    | cast                                                                                              #typeCast
    | functionCall                                                                                      #funcCall
    | expr '[' expr (',' expr)? ']'                                                                     #index
    | ID DOT (INT | ID)                                                                                 #tupleIndex //need to fix this. no time
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
    | expr op=RESERVED_AND expr                                                                         #math
    | expr op=(RESERVED_OR | RESERVED_XOR) expr                                                         #math
    | expr CONCAT expr                                                                                  #concatenation
    | ID                                                                                                #literalID
    | RESERVED_IDENTITY                                                                                 #identity
    | RESERVED_NULL                                                                                     #null
    | LITERAL_BOOLEAN                                                                                   #literalBoolean
    | LITERAL_CHARACTER                                                                                 #literalCharacter
    | INT                                                                                               #literalInt
    | literal_real                                                                                      #literalReal
    | '(' expr (',' expr )+ ')'                                                                         #literalTuple
 //   | literal_matrix                                                                                    #literalMatrix
    | literal_vector                                                                                    #literalVector
    | LITERAL_STRING                                                                                    #literalString
    ;

literal_vector: '[' (expression (',' expression)*)? ']'; // empty vectors allowed
//literal_matrix: '[' literal_vector (',' literal_vector)* ']'; // empty matrices allowed
literal_real
    : INT DOT INT EXPONENT?
    | DOT INT EXPONENT?
    | INT DOT? EXPONENT?
    ;
cast: RESERVED_AS LT type GT '(' expression ')';
typedef: RESERVED_TYPEDEF type ID ';'; // inferred types allowed in typedefs
stream
    : expression RIGHT_ARROW RESERVED_STD_OUTPUT ';' #streamOut
    | expression LEFT_ARROW RESERVED_STD_INPUT ';' #streamIn
    ;

// operators
BUILT_IN_TYPE
    : RESERVED_STRING
    | RESERVED_BOOLEAN
    | RESERVED_CHARACTER
    | RESERVED_INTEGER
    | RESERVED_REAL;


INT : DIGIT+;
LITERAL_BOOLEAN: RESERVED_TRUE | RESERVED_FALSE;
LITERAL_CHARACTER: '\'' SCHAR '\'';
LITERAL_STRING: '"' (SCHAR+)? '"';
EXPONENT: ('e' | 'E') (SUB | ADD)? INT;

DOT: '.';
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

// Skip comments & whitespace
BLOCK_COMMENT : '/*' .*? '*/' -> skip ;
LINE_COMMENT : '//' ~[\r\n]* -> skip ;
WS : [ \t\r\n]+ -> skip;

fragment
DIGIT : [0-9];
ALPHABET : [a-zA-Z];
SCHAR : ('\\' [0abtnr"'\\] | ~["\\\r\n]);