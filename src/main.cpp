#include "GazpreaLexer.h"
#include "GazpreaParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "AST.h"
#include "ASTBuilder.h"
#include "SymbolTable.h"
#include "ASTWalker.h"
#include "BackEnd.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n";
    return 1;
  }

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  gazprea::GazpreaLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  gazprea::GazpreaParser parser(&tokens);

  // Get the root of the parse tree. Use your base rule name.
  antlr4::tree::ParseTree *tree = parser.file();

  // build AST
  std::cerr << "Building AST" << std::endl;
  gazprea::ASTBuilder builder;
  AST* ast = std::any_cast<AST*>(builder.visit(tree));

  std::cerr << "\n\n=== Building SymbolTable" << std::endl;
  SymbolTable symbolTable;

  gazprea::DefRef defref(&symbolTable, ast);
  defref.visit(ast);

  std::cerr << "\n\n=== Computing Types" << std::endl;
  gazprea::ComputeType typeCompute(&symbolTable);
  typeCompute.visit(ast);
  std::cout << "Finished type computing" << std::endl;

  // HOW TO WRITE OUT.
  // std::ofstream out(argv[2]);
  // out << "This is out...\n";

  return 0;
}
