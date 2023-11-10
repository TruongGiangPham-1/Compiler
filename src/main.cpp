#include "GazpreaLexer.h"
#include "GazpreaParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "ASTNode/ASTNode.h"
#include "ASTBuilder.h"
#include "SymbolTable.h"
#include "ASTWalker.h"
#include "TypeWalker.h"
#include "BackendWalker.h"
#include "Def.h"
#include "Ref.h"
#include "../include/customError/ErrorListener.h"

#include <iostream>
#include <fstream>

//#define DEBUG
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

  parser.removeErrorListeners();
  parser.addErrorListener(new ErrorListener());

  std::ofstream out(argv[2]);

  // Get the root of the parse tree. Use your base rule name.
  antlr4::tree::ParseTree *tree = parser.file();

  // build ASTNode
#ifdef DEBUG
  std::cout << "\n\n=== Building ASTNode" << std::endl;
#endif
  gazprea::ASTBuilder builder;
  auto ast = std::any_cast<std::shared_ptr<ASTNode>>(builder.visit(tree));
#ifdef DEBUG
  std::cout << ast->toStringTree() << std::endl;

  std::cout << "\n\n=== Building SymbolTable" << std::endl;

  std::cout << "\n\n=== DEF PASS\n";
#endif


  int mlirID = 1;
  std::shared_ptr<int>mlirIDptr = std::make_shared<int>(mlirID);
  std::shared_ptr<SymbolTable> symbolTable = std::make_shared<SymbolTable>();
  gazprea::Def def(symbolTable, mlirIDptr);
  try {
      def.walk(ast);
  }
  catch (CompileTimeException& e) {
      std::cerr << e.what();
      return 1;
  }


#ifdef DEBUG
  std::cout << "\n\n=== REF PASS\n";
#endif
  gazprea::Ref ref(symbolTable, mlirIDptr);
  try {
      ref.walk(ast);
  }
  catch (CompileTimeException& e) {
      std::cerr << e.what();
      return 1;
  }

#ifdef DEBUG
  std::cout << "\n\n=== TYPECHECK PASS\n";
#endif
  //Type Check
  auto promotionTypes = std::make_shared<gazprea::PromotedType>(symbolTable);
  gazprea::TypeWalker typeWalker(symbolTable, promotionTypes);
    try {
        typeWalker.walk(ast);
    }
    catch (CompileTimeException& e) {
        std::cerr << e.what();
        return 1;
    }

#ifdef DEBUG
  std::cout << "\n\n=== CODEGEN\n";
#endif
  BackendWalker backend(out);
  try {
      backend.generateCode(ast);
  }
  catch (CompileTimeException& e) {
      std::cerr << e.what();
      return 1;
  }
#ifdef DEBUG
    std::cout << "\n\n=== BACKEND END\n";
#endif

//  gazprea::DefRef defref(&symbolTable, ast);
//  defref.visit(ast);

  // HOW TO WRITE OUT.
  // std::ofstream out(argv[2]);
  // out << "This is out...\n";

  return 0;
}
