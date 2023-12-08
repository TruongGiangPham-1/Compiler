#include "ASTNode/ASTNode.h"
#include "ASTNode/Block/BlockNode.h"
#include "ASTNode/Expr/Literal/BoolNode.h"
#include "ASTNode/Expr/Literal/CharNode.h"
#include "ASTWalker.h"
#include "BackEnd.h"

class BackendWalker : private gazprea::ASTWalker {
private:
  BackEnd codeGenerator;
  std::any walk(std::shared_ptr<ASTNode> tree) override;

  // if we are inside a loop, we want to track the start and end blocks
  // this is so we can know where to jump to when we encounter a break or continue
  std::vector<std::pair<mlir::Block *, mlir::Block *>> loopBlocks;
  // if we encounter a break, we have an early return
  // this boolean is true when we are inside a loop and we encounter a break
  bool earlyReturn = false;
  bool returnDropped = false;
  bool fetchRaw = false; // we do casting, which creates a copy. to assign we don't want copy
  std::vector<mlir::Value> inferenceContext;
  mlir::Value methodStack;
  std::shared_ptr<ASTNode> returnType;

  std::any visitAssign(std::shared_ptr<AssignNode> tree) override;
  std::any visitDecl(std::shared_ptr<DeclNode> tree) override;
  std::any visitStreamOut(std::shared_ptr<StreamOut> tree) override;
  std::any visitStreamIn(std::shared_ptr<StreamIn> tree) override;

  std::any visitType(std::shared_ptr<TypeNode> tree) override;

  // === EXPRESSION AST NODES ===
  std::any visitID(std::shared_ptr<IDNode> tree) override;
  std::any visitInt(std::shared_ptr<IntNode> tree) override;
  std::any visitReal(std::shared_ptr<RealNode> tree) override;
  std::any visitChar(std::shared_ptr<CharNode> tree) override;
  std::any visitString(std::shared_ptr<VectorNode> tree) override;
  std::any visitBool(std::shared_ptr<BoolNode> tree) override;
  std::any visitTuple(std::shared_ptr<TupleNode> tree) override;
  std::any visitVector(std::shared_ptr<VectorNode> tree) override;
  std::any visitStdInputNode(std::shared_ptr<StdInputNode> tree) override;


  // Expr/Binary
  std::any visitArith(std::shared_ptr<BinaryArithNode> tree) override;
  std::any visitCmp(std::shared_ptr<BinaryCmpNode> tree) override;
  std::any visitUnaryArith(std::shared_ptr<UnaryArithNode> tree) override;
  std::any visitIndex(std::shared_ptr<IndexNode> tree) override;
  std::any visitStride(std::shared_ptr<StrideNode> tree) override;
  std::any visitCast(std::shared_ptr<CastNode> tree) override;
  std::any visitTupleIndex(std::shared_ptr<TupleIndexNode> tree) override;
  std::any visitConcat(std::shared_ptr<ConcatNode> tree) override;

  // Expr/Vector
  std::any visitFilter(std::shared_ptr<FilterNode> tree) override;
  std::any visitGenerator(std::shared_ptr<GeneratorNode> tree) override;
  std::any visitRangeVec(std::shared_ptr<RangeVecNode> tree) override;

  // === BLOCK AST NODES ===
  std::any visitConditional(std::shared_ptr<ConditionalNode> tree) override;
  std::any visitInfiniteLoop(std::shared_ptr<InfiniteLoopNode> tree) override;
  std::any visitPredicatedLoop(std::shared_ptr<PredicatedLoopNode> tree) override;
  std::any visitPostPredicatedLoop(std::shared_ptr<PostPredicatedLoopNode> tree) override;
  std::any visitIteratorLoop(std::shared_ptr<IteratorLoopNode> tree) override;
  std::any visitBreak(std::shared_ptr<BreakNode> tree) override;
  std::any visitContinue(std::shared_ptr<ContinueNode> tree) override;

  std::any visitBlock(std::shared_ptr<BlockNode> tree) override;

  // method definitions
  std::any visitProcedure(std::shared_ptr<ProcedureNode> tree) override;

  std::any visitFunction(std::shared_ptr<FunctionNode> tree) override;

  std::any visitCall(std::shared_ptr<CallNode> tree) override;

  std::any visitReturn(std::shared_ptr<ReturnNode> tree) override;
  // === Null and identity
  std::any visitNull(std::shared_ptr<NullNode> tree) override;
  std::any visitIdentity(std::shared_ptr<IdentityNode> tree) override;

  mlir::Value castType(mlir::Value, std::shared_ptr<Type> type);
public:
  explicit BackendWalker(std::ofstream &out) : codeGenerator(out){};

  void generateCode(std::shared_ptr<ASTNode> tree);
};
