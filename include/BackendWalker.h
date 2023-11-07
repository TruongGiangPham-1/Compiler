#include "ASTWalker.h"
#include "BackEnd.h"

class BackendWalker : private gazprea::ASTWalker {
private:
  BackEnd codeGenerator;

  // if we are inside a loop, we want to track the start and end blocks
  // this is so we can know where to jump to when we encounter a break or continue
  std::vector<std::pair<mlir::Block *, mlir::Block *>> loopBlocks;

  std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
  std::any visitDecl(std::shared_ptr<DeclNode> tree) override;
  std::any visitPrint(std::shared_ptr<StreamOut> tree) override;

  // === EXPRESSION AST NODES ===
  std::any visitID(std::shared_ptr<IDNode> tree) override;
  std::any visitInt(std::shared_ptr<IntNode> tree) override;
  // Expr/Binary
  std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
  std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
  std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
  // Expr/Vector
  std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
  std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
  std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;

  // === BLOCK AST NODES ===
  std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
  std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree);
  std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree);
  std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree);
  std::any visitBreak(std::shared_ptr<BreakNode> tree);
  std::any visitContinue(std::shared_ptr<ContinueNode> tree);

public:
  explicit BackendWalker(std::ofstream &out) : codeGenerator(out){};
  void generateCode(std::shared_ptr<ASTNode> tree);
};
