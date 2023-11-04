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
  // Expr/Binary
  std::any visitArith(std::shared_ptr<ArithOpNode> tree) override;
  std::any visitCmp(std::shared_ptr<CmpOpNode> tree) override;
  std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
  // Expr/Vector
  std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
  std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
  std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;

  // === BLOCK AST NODES ===
  std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
  std::any visitLoop(std::shared_ptr<LoopNode> tree) override;

public:
  explicit BackendWalker(std::ofstream &out) : codeGenerator(out){};
  void generateCode(std::shared_ptr<ASTNode> tree);
};
