//
// Created by truong on 02/11/23.
//

#ifndef GAZPREABASE_FUNCCALLTYPE_H
#define GAZPREABASE_FUNCCALLTYPE_H
#include <string>
enum FUNCTYPE {FUNC_NORMAL, FUNC_COLUMN, FUNC_FORMAT, FUNC_LENGTH, FUNC_REVERSE, FUNC_ROW, FUNC_STDIN, FUNC_SSTATE};
extern std::string funcTypeStr[8];
#endif //GAZPREABASE_FUNCCALLTYPE_H
