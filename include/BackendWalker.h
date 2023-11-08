#include "ASTNode/Block/BlockNode.h"
#include "ASTNode/Expr/Literal/BoolNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
#include "ASTWalker.h"
#include "BackEnd.h"

class BackendWalker : private gazprea::ASTWalker {
private:
  BackEnd codeGenerator;
  std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
  std::any visitDecl(std::shared_ptr<DeclNode> tree) override;
  std::any visitPrint(std::shared_ptr<StreamOut> tree) override;

  // === EXPRESSION AST NODES ===
  std::any visitID(std::shared_ptr<IDNode> tree) override;
  std::any visitInt(std::shared_ptr<IntNode> tree) override;
  std::any visitReal(std::shared_ptr<RealNode> tree) override;
  std::any visitChar(std::shared_ptr<CharNode> tree) override;
  std::any visitBool(std::shared_ptr<BoolNode> tree) override;
  std::any visitTuple(std::shared_ptr<TupleNode> tree) override;

  // Expr/Binary
  std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
  std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
  std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
  std::any visitCast(std::shared_ptr<CastNode> tree) override;
  // Expr/Vector
  std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
  std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
  std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;

  // === BLOCK AST NODES ===
  std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
  std::any visitBlock(std::shared_ptr<BlockNode> tree) override;

  // method definitions
  std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;

  std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;

  std::any visitCall(std::shared_ptr<CallNode> tree) override;

  std::any visitReturn(std::shared_ptr<ReturnNode> tree) override;

public:
  explicit BackendWalker(std::ofstream &out) : codeGenerator(out){};
  void generateCode(std::shared_ptr<ASTNode> tree);
};
