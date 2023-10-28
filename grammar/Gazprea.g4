grammar Gazprea;

tokens {
  VAR_DECL,
  ASSIGN,
  CONDITIONAL,
  LOOP,
  PRINT,
  RANGE,
  FILTER,
  GENERATOR,
  INDEX,
  EXPRESSION,
  PARENTHESES,
  TYPE
}

file
    : statement* EOF
    ;

// vardecls, assignments, nested loops, nested conditionals, prints
statement
    : vardecl | assign | loop | cond | print
    ;

vardecl
    : qualifier? inferred_sized_type ID '=' expr ';'
    | qualifier? known_sized_type ID ('=' expr)? ';'
    | qualifier ID '=' expr ';';

type: known_sized_type | inferred_sized_type;
tuple_allowed_type: built_in_type | vector_type | string_type | matrix_type | inferred_sized_type;

known_sized_type: built_in_type | tuple_type | vector_type | string_type | matrix_type;
inferred_sized_type
    : built_in_type '[' MULT ']'            #vector
    | RESERVED_STRING '[' MULT ']'          #string
    | built_in_type '[' MULT ',' expr ']'   #matrixFirst
    | built_in_type '[' expr ',' MULT ']'   #matrixSecond
    | built_in_type '[' MULT ',' MULT ']'   #matrix
    ;

qualifier: RESERVED_CONST | RESERVED_VAR;
built_in_type: RESERVED_BOOLEAN | RESERVED_CHARACTER | RESERVED_INTEGER | RESERVED_REAL | ID; // ID incase of typedefs... This might be changed
tuple_type: RESERVED_TUPLE '(' tuple_allowed_type ID? (',' tuple_allowed_type ID?)+ ')';
vector_type: built_in_type '[' expr ']';
string_type: RESERVED_STRING ('[' expr ']')?;
matrix_type: built_in_type '[' expr ',' expr ']';

assign
    : ID '=' expression ';'
    ;
loop
    : RESERVED_LOOP '(' expression ')' statement* RESERVED_POOL ';'
    ;
cond
    : RESERVED_IF '(' expression ')' statement* RESERVED_FI ';'
    ;
print
    : RESERVED_PRINT '(' expression ')' ';'
    ;
expression // root of an expression tree
    :   expr
    ;
expr
    : '(' expr ')'                                      #paren
    | expr '[' expr ']'                                 #index
    | expr '..' expr                                    #range
    | '[' ID RESERVED_IN expression '|' expression ']'  #generator
    | '[' ID RESERVED_IN expression '&' expression ']'  #filter
    | expr op=(MULT | DIV) expr                         #math
    | expr op=(ADD | SUB) expr                          #math
    | expr op=(LT | GT) expr                            #cmp
    | expr op=(EQ | NEQ) expr                           #cmp
    | INT                                               #literalInt
    | ID                                                #literalID
    ;

// operators
MULT: '*';
DIV: '/';
ADD: '+';
SUB: '-';
LT: '<';
GT: '>';
EQ: '==';
NEQ: '!=';

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

ID : ('_' | CHAR) ('_' | CHAR | DIGIT)*;
INT : DIGIT+;

// Skip comments & whitespace
BLOCK_COMMENT : '/*' .*? '*/' -> skip ;
LINE_COMMENT : '//' ~[\r\n]* -> skip ;
WS : [ \t\r\n]+ -> skip;

fragment
DIGIT : [0-9];
CHAR : [a-zA-Z];