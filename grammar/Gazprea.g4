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
  PARENTHESES
}

file
    : statement* EOF
    ;

// vardecls, assignments, nested loops, nested conditionals, prints
statement
    : vardecl | assign | loop | cond | print
    ;

vardecl
    : type ID '=' expression ';'
    ;
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

type : RESERVED_VECTOR | RESERVED_INT;

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
RESERVED_IF : 'if';
RESERVED_FI : 'fi';
RESERVED_LOOP : 'loop';
RESERVED_POOL : 'pool';
RESERVED_INT : 'int';
RESERVED_PRINT : 'print';
RESERVED_IN : 'in';
RESERVED_VECTOR: 'vector';

ID : CHAR (CHAR | DIGIT)*;
INT : DIGIT+;

// Skip whitespace
WS : [ \t\r\n]+ -> skip;

fragment
DIGIT : [0-9];
CHAR : [a-zA-Z];