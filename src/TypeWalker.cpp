#include "TypeWalker.h"
#include "BuiltInTypeSymbol.h"
#include <stdexcept>
#define DEBUG

int TypeWalker::getTypeIndex(const std::string type) {
  if (type == "boolean") {
    return this->boolIndex;
  } else if (type == "character") {
    return this->charIndex;
  } else if (type == "integer") {
    return this->integerIndex;
  } else if (type == "real") {
    return this->realIndex;
  } else if (type == "vector") {
    return this->vectorIndex;
  } else {
    throw std::runtime_error("Unknown type");
  }
}

std::shared_ptr<Type> TypeWalker::getPromotedType(const std::shared_ptr<Type> left, const std::shared_ptr<Type> right) {
  auto leftIndex = this->getTypeIndex(left->getName());
  auto rightIndex = this->getTypeIndex(right->getName());
  auto promotedString = this->promotionTable[leftIndex][rightIndex];

  auto promotedType = std::make_shared<BuiltInTypeSymbol>(this->promotionTable[leftIndex][rightIndex]);

#ifdef DEBUG
  std::cout << "type promotions between " <<  left->getName() << ", " << right->getName();
  std::cout << "result :" <<  promotedType->getName() << std::endl;
#endif 

  return promotedType;
}

std::any TypeWalker::visitArith(std::shared_ptr<ArithNode> tree) {
  walk(tree->getRHS());
  walk(tree->getLHS());

  auto rightType = tree->getRHS()->type;
  auto leftType = tree->getLHS()->type;

  auto promoteLeft = this->getPromotedType(leftType, rightType);
  auto promoteRight = this->getPromotedType(rightType, leftType);

  tree->getRHS()->promoteTo = promoteRight;
  tree->getLHS()->promoteTo = promoteLeft;

  return 0;
}

std::any TypeWalker::visitCmp(std::shared_ptr<CmpNode> tree) {
  walk(tree->getRHS());
  walk(tree->getLHS());

  auto rightType = tree->getRHS()->type;
  auto leftType = tree->getLHS()->type;

  auto promoteLeft = this->getPromotedType(leftType, rightType);
  auto promoteRight = this->getPromotedType(rightType, leftType);

  tree->getRHS()->promoteTo = promoteRight;
  tree->getLHS()->promoteTo = promoteLeft;

  return 0;
}

std::any TypeWalker::visitIndex(std::shared_ptr<IndexNode> tree) {
  // TODO this
  auto right = tree->getRHS()->type;
  auto type = std::make_shared<BuiltInTypeSymbol>("integer");
  tree->type = type;

#ifdef DEBUG
  std::cout << "Visit index resulting type is\n";
#endif // DEBUG
  return 0;
}

std::any TypeWalker::visitID(std::shared_ptr<IDNode> tree) {
  //TODO fix this, we need to pull from defref walk
  auto type = std::make_shared<BuiltInTypeSymbol>("integer");
#ifdef DEBUG
  std::cout << "Visint identifier, resolve to\n";
#endif // DEBUG
  tree->type = type;
  return 0;
}

std::any TypeWalker::visitInt(std::shared_ptr<IntNode> tree) {
  auto type = std::make_shared<BuiltInTypeSymbol>("integer");
  tree->type = type;
  return 0;
}

std::any TypeWalker::visitFilter(std::shared_ptr<FilterNode> tree) {
  auto type = std::make_shared<BuiltInTypeSymbol>("vector");
  tree->type = type;
  return 0;
}

std::any TypeWalker::visitGenerator(std::shared_ptr<GeneratorNode> tree) {
  auto type = std::make_shared<BuiltInTypeSymbol>("vector");
  tree->type = type;
  return 0;
}

std::any TypeWalker::visitRangeVec(std::shared_ptr<RangeVecNode> tree) {
  auto type = std::make_shared<BuiltInTypeSymbol>("vector");
  tree->type = type;
  return 0;
}



