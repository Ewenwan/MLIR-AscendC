//===- TranslateToCpp.cpp - Translating to C++ calls ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-ascendc/Target/Cpp/translateAscendCToCpp.h"
#include "mlir-ascendc/Dialect/AscendC/IR/AscendC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

#define DEBUG_TYPE "ascendc-to-cpp"

using namespace mlir;
using namespace ascendc;

//===----------------------------------------------------------------------===//
// CppEmitter Struct
//===----------------------------------------------------------------------===//

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  explicit CppEmitter(raw_ostream &os, bool toggleAscendCSpecifics);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Return the element type as a string
  std::string printElementType(Type type);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits AscendC specific types, called inside emitType
  LogicalResult emitAscendCTypes(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits value as an operands of an operation
  LogicalResult emitOperand(Value value);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    explicit Scope(CppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    CppEmitter &emitter;
  };

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Return if currently adds AscendC specific contents
  bool removeAscendCInfo() { return toggleAscendCSpecifics; };

  /// Update current AscendC tensor element type
  void updateTensorElementType(std::string &elementType) {
    tensorElementType = elementType;
  }

  /// Returns current AscendC tensor element type
  std::string getTensorElementType() { return tensorElementType; }

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// If remove AscendC specific contents
  bool toggleAscendCSpecifics;

  /// Store the current AscendC tensorElementType
  std::string tensorElementType = {};

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
} // namespace

//===----------------------------------------------------------------------===//
// CppEmitter Member Fucntions
//===----------------------------------------------------------------------===//

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

/// Return the existing or a new name for a Value.
StringRef CppEmitter::getOrCreateName(Value val) {

  if (!valueMapper.count(val)) {
    if (dyn_cast<LocalTensorType>(val.getType())) {
      valueMapper.insert(
          val, llvm::formatv("localTensor{0}", ++valueInScopeCount.top()));
    } else if (auto queType = dyn_cast<TQueType>(val.getType())) {
      if (queType.getTposition() == TPosition::VECIN)
        valueMapper.insert(
            val, llvm::formatv("inQueue{0}", ++valueInScopeCount.top()));
      else if (queType.getTposition() == TPosition::VECOUT)
        valueMapper.insert(
            val, llvm::formatv("outQueue{0}", ++valueInScopeCount.top()));
      else
        valueMapper.insert(val,
                           llvm::formatv("que{0}", ++valueInScopeCount.top()));
    } else if (dyn_cast<TPipeType>(val.getType())) {
      valueMapper.insert(val, llvm::formatv("pipe"));
    } else if (dyn_cast<IntegerType>(val.getType())) {
      valueMapper.insert(val, llvm::formatv("i{0}", ++valueInScopeCount.top()));
    } else if (dyn_cast<GlobalTensorType>(val.getType())) {
      valueMapper.insert(
          val, llvm::formatv("gmTensor{0}", ++valueInScopeCount.top()));
    } else {
      valueMapper.insert(val, llvm::formatv("v{0}", ++valueInScopeCount.top()));
    }
  }
  return *valueMapper.begin(val);
}

CppEmitter::CppEmitter(raw_ostream &os, bool toggleAscendCSpecifics)
    : os(os), toggleAscendCSpecifics(toggleAscendCSpecifics) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  default:
    llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

std::string CppEmitter::printElementType(Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return "bool";
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return llvm::formatv("uint{0}_t", iType.getWidth());
      else
        return llvm::formatv("int{0}_t", iType.getWidth());
    default:
      return nullptr;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 16:
      return "half";
    case 32:
      return "float";
    case 64:
      return "double";
    default:
      return nullptr;
    }
  }
  return nullptr;
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 16:
      return (os << "half"), success();
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }

  if (failed(emitAscendCTypes(loc, type)))
    return emitError(loc, "cannot emit type ") << type;
  return success();
}

LogicalResult CppEmitter::emitAscendCTypes(Location loc, Type type) {
  if (auto localTensor = dyn_cast<LocalTensorType>(type)) {
    os << "LocalTensor<";
    if (failed(emitType(loc, localTensor.getElementType())))
      return failure();
    os << ">";
    return success();
  }
  if (auto tpipe = dyn_cast<TPipeType>(type)) {
    os << "TPipe";
    return success();
  }
  if (auto tque = dyn_cast<TQueType>(type)) {
    os << "TQue<QuePosition::" << tque.getTposition() << ", "
       << tque.getBufNum() << ">";
    return success();
  }
  if (auto tque = dyn_cast<GM_ADDRType>(type)) {
    os << "GM_ADDR";
    return success();
  }
  if (auto globalTensor = dyn_cast<GlobalTensorType>(type)) {
    os << "GlobalTensor<";
    if (failed(emitType(loc, globalTensor.getElementType())))
      return failure();
    os << ">";
    return success();
  }
  return failure();
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitError(loc, "unsupported multi results function");
  }
}

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
  raw_ostream &os = this->ostream();
  auto printInt = [&os](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&os](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "(float)";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        os << "(double)";
        break;
      default:
        break;
      };
      os << strValue;
    }
    // TODO: consider printing NAN/(+-)INFINITY here
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
  if (failed(emitType(result.getOwner()->getLoc(), result.getType())))
    return failure();
  os << " " << getOrCreateName(result);
  if (trailingSemicolon)
    os << ";";
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {

  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
      return failure();
    os << " = ";
    break;
  }
  default:
    for (OpResult result : op.getResults()) {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
        return failure();
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CppEmitter::emitOperand(Value value) {
  os << getOrCreateName(value);
  return success();
}

//===----------------------------------------------------------------------===//
// Arith Operation Translation
//===----------------------------------------------------------------------===//

static LogicalResult printBinaryOperation(CppEmitter &emitter,
                                          Operation *operation,
                                          StringRef binaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  if (failed(emitter.emitOperand(operation->getOperand(0))))
    return failure();

  os << " " << binaryOperator << " ";

  if (failed(emitter.emitOperand(operation->getOperand(1))))
    return failure();

  return success();
}

static LogicalResult printOperation(CppEmitter &emitter, arith::AddFOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "+");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::AddIOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "+");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivFOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivUIOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::DivSIOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "/");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MulFOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "*");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::MulIOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "*");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SubFOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "-");
}

static LogicalResult printOperation(CppEmitter &emitter, arith::SubIOp op) {
  Operation *operation = op.getOperation();

  return printBinaryOperation(emitter, operation, "-");
}

static LogicalResult printConstantOp(CppEmitter &emitter, Operation *operation,
                                     Attribute value) {
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(CppEmitter &emitter,
                                    arith::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  if (failed(printConstantOp(emitter, operation, value)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Scope Related Translation
//===----------------------------------------------------------------------===//

// func::FuncOp
static LogicalResult printOperation(CppEmitter &emitter,
                                    func::FuncOp functionOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  // FYI: EmitC funcOp is developing a method of adding extern "C"
  if (!emitter.removeAscendCInfo())
    os << "extern \"C\" __global__ __aicore__ ";

  if (failed(emitter.emitTypes(functionOp.getLoc(),
                               functionOp.getFunctionType().getResults())))
    return failure();
  os << " " << functionOp.getName();

  os << "(";
  if (failed(interleaveCommaWithError(
          functionOp.getArguments(), os,
          [&](BlockArgument arg) -> LogicalResult {
            if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
              return failure();
            os << " " << emitter.getOrCreateName(arg);
            return success();
          })))
    return failure();
  os << ") {\n";
  os.indent();

  for (Block &block : functionOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      bool trailingSemicolon = !isa<scf::ForOp>(op);

      if (failed(emitter.emitOperation(op, trailingSemicolon)))
        return failure();
    }
  }
  os.unindent() << "}\n";
  return success();
}

// func::ReturnOp
static LogicalResult printOperation(CppEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    return success();
  case 1:
    os << " ";
    if (failed(emitter.emitOperand(returnOp.getOperand(0))))
      return failure();
    return success();
  default:
    return emitError(returnOp->getLoc(), "Cannot return with multiple results");
  }
}

// scf::ForOp
static LogicalResult printOperation(CppEmitter &emitter, scf::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  os << "for (";
  if (failed(
          emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType())))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if (failed(emitter.emitOperand(forOp.getLowerBound())))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  Value upperBound = forOp.getUpperBound();
  if (failed(emitter.emitOperand(upperBound)))
    return failure();
  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if (failed(emitter.emitOperand(forOp.getStep())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  return success();
}

// Can remove yieldOp translation as yieldOps are skipped
static LogicalResult printOperation(CppEmitter &emitter,
                                    scf::YieldOp returnOp) {
  return success();
}

//===----------------------------------------------------------------------===//
// AscendC Related Translation
//===----------------------------------------------------------------------===//

// Create Pipe Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::CreatePipeOp createPipeOp) {
  if (failed(
          emitter.emitVariableDeclaration(createPipeOp->getResult(0), false)))
    return failure();
  return success();
}

// Init buffers after creating a queue
static LogicalResult initQueue(CppEmitter &emitter, ascendc::CreateQueueOp op) {
  raw_ostream &os = emitter.ostream();
  std::string elementType = {};
  if (emitter.getTensorElementType().empty()) {
    elementType = "half";
  } else {
    elementType = emitter.getTensorElementType();
  }
  os << "\n";

  os << emitter.getOrCreateName(op.getTpipe()) << ".InitBuffer(";
  os << emitter.getOrCreateName(op.getQueue()) << ", ";
  os << op.getQueue().getType().getBufNum() << ", ";
  os << emitter.getOrCreateName(op.getTileLength()) << " * ";
  os << "sizeof(" << elementType << "))";

  return success();
}

// Create Queue Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::CreateQueueOp createQueueOp) {
  if (failed(
          emitter.emitVariableDeclaration(createQueueOp->getResult(0), true)))
    return failure();
  if (failed(initQueue(emitter, createQueueOp)))
    return failure();
  return success();
}

// Deque Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::DequeOp dequeOp) {
  raw_ostream &os = emitter.ostream();

  Type elemType = dequeOp.getLocalTensor().getType().getElementType();
  Operation *operation = dequeOp.getOperation();
  Location loc = operation->getLoc();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << emitter.getOrCreateName(dequeOp.getQueue());

  os << ".DeQue<";
  if (failed(emitter.emitType(loc, elemType)))
    return emitError(loc, "cannot LocalTensor element type ") << elemType;
  os << ">()";

  return success();
}

// Enque Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::EnqueOp enqueOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(enqueOp.getQueue())))
    return failure();
  os << ".EnQue(" << emitter.getOrCreateName(enqueOp.getLocalTensor()) << ")";
  return success();
}

// Add Op
static LogicalResult printOperation(CppEmitter &emitter, ascendc::AddOp addOp) {
  raw_ostream &os = emitter.ostream();
  os << "Add(";
  if (failed(emitter.emitOperand(addOp.getDst())))
    return failure();
  os << ", ";
  if (failed(emitter.emitOperand(addOp.getSrc0())))
    return failure();
  os << ", ";
  if (failed(emitter.emitOperand(addOp.getSrc1())))
    return failure();
  if (auto tileLen = dyn_cast_or_null<Value>(addOp.getTileLength())) {
    os << ", ";
    if (failed(emitter.emitOperand(tileLen)))
      return failure();
  }
  os << ")";
  return success();
}

// FreeTensor Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::FreeTensorOp freeOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitOperand(freeOp.getQueue())))
    return failure();
  os << ".FreeTensor(";
  if (failed(emitter.emitOperand(freeOp.getLocalTensor())))
    return failure();
  os << ")";
  return success();
}

// AllocTensor Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::AllocTensorOp allocOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*allocOp.getOperation())))
    return failure();
  if (failed(emitter.emitOperand(allocOp.getQueue())))
    return failure();
  os << ".AllocTensor<";
  if (failed(emitter.emitType(allocOp->getLoc(),
                              allocOp.getLocalTensorType().getElementType())))
    return failure();
  os << ">()";
  return success();
}

// DataCopyOp
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::DataCopyOp dataCopyOp) {
  raw_ostream &os = emitter.ostream();
  os << "DataCopy(";
  if (failed(emitter.emitOperand(dataCopyOp.getDstTensor())))
    return failure();
  if (auto dstOffset = dyn_cast_or_null<Value>(dataCopyOp.getDstOffset())) {
    os << "[";
    if (failed(emitter.emitOperand(dstOffset)))
      return failure();
    os << "]";
  }
  os << ", ";
  if (failed(emitter.emitOperand(dataCopyOp.getSrcTensor())))
    return failure();
  if (auto srcOffset = dyn_cast_or_null<Value>(dataCopyOp.getSrcOffset())) {
    os << "[";
    if (failed(emitter.emitOperand(srcOffset)))
      return failure();
    os << "]";
  }
  os << ", ";
  if (failed(emitter.emitOperand(dataCopyOp.getTileLength())))
    return failure();
  os << ")";
  return success();
}

// GetBlockIdx Op
static LogicalResult printOperation(CppEmitter &emitter,
                                    ascendc::GetBlockIdxOp blockIdOp) {
  raw_ostream &os = emitter.ostream();
  if (failed(emitter.emitAssignPrefix(*blockIdOp.getOperation())))
    return failure();
  os << "GetBlockIdx()";
  return success();
}

// Set global buffer after declaring global tensor
static LogicalResult setGlobalBuffer(CppEmitter &emitter,
                                     ascendc::CreateGlobalTensorOp op) {
  raw_ostream &os = emitter.ostream();
  os << "\n";
  if (failed(emitter.emitOperand(op.getGlobalTensor())))
    return failure();
  os << ".SetGlobalBuffer((";
  if (!emitter.removeAscendCInfo())
    os << "__gm__ ";
  if (failed(emitter.emitType(op->getLoc(),
                              op.getGlobalTensor().getType().getElementType())))
    return failure();
  os << "*)";
  os << emitter.getOrCreateName(op.getGmAddress());
  if (op.getOffset())
    os << " + " << emitter.getOrCreateName(op.getOffset());
  os << ", " << emitter.getOrCreateName(op.getBlockLen());
  os << ")";

  return success();
}

// CreateGlobalTensorOp
static LogicalResult
printOperation(CppEmitter &emitter,
               ascendc::CreateGlobalTensorOp globalTensorOp) {
  if (emitter.getTensorElementType().empty()) {
    std::string elementType = emitter.printElementType(
        globalTensorOp.getGlobalTensor().getType().getElementType());
    emitter.updateTensorElementType(elementType);
  }
  if (failed(
          emitter.emitVariableDeclaration(globalTensorOp->getResult(0), true)))
    return failure();
  if (failed(setGlobalBuffer(emitter, globalTensorOp)))
    return failure();
  return success();
}

// Module Op
static LogicalResult printOperation(CppEmitter &emitter, ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

// Op translation registration
LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // AscendC ops
          .Case<ascendc::DequeOp, ascendc::EnqueOp, ascendc::AddOp,
                ascendc::FreeTensorOp, ascendc::AllocTensorOp,
                ascendc::CreateQueueOp, ascendc::CreatePipeOp,
                ascendc::GetBlockIdxOp, ascendc::CreateGlobalTensorOp,
                ascendc::DataCopyOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // SCF ops
          .Case<scf::ForOp, scf::YieldOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Arithmetic ops
          .Case<arith::ConstantOp, arith::AddFOp, arith::AddIOp, arith::DivFOp,
                arith::DivUIOp, arith::DivSIOp, arith::MulFOp, arith::MulIOp,
                arith::SubFOp, arith::SubIOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Unsupported
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();

  os << (trailingSemicolon ? ";\n" : "\n");

  return success();
}

LogicalResult mlir::translateAscendCToCpp(Operation *op, raw_ostream &os,
                                          bool toggleAscendCSpecifics) {
  CppEmitter emitter(os, toggleAscendCSpecifics);
  if (!emitter.removeAscendCInfo()) {
    emitter.ostream()
        << "/*\n* Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All "
           "rights reserved."
        << "\n* This code is generated by AscendC EmitC Builder."
        << "\n*/\n\n";
    emitter.ostream()
        << "#include \"kernel_operator.h\"\nusing namespace AscendC;\n\n";
  }
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}
