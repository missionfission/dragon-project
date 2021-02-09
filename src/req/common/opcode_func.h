#include "power_func.h"

#ifndef OPCODE_FUNC_H
#define OPCODE_FUNC_H

// clang-format off
#define LLVM_IR_Move 0
#define LLVM_IR_Ret 1
#define LLVM_IR_Br 2
#define LLVM_IR_Switch 3
#define LLVM_IR_IndirectBr 4
#define LLVM_IR_Invoke 5
#define LLVM_IR_Resume 6
#define LLVM_IR_Unreachable 7
#define LLVM_IR_Add 8
#define LLVM_IR_FAdd 9
#define LLVM_IR_Sub 10
#define LLVM_IR_FSub 11
#define LLVM_IR_Mul 12
#define LLVM_IR_FMul 13
#define LLVM_IR_UDiv 14
#define LLVM_IR_SDiv 15
#define LLVM_IR_FDiv 16
#define LLVM_IR_URem 17
#define LLVM_IR_SRem 18
#define LLVM_IR_FRem 19
#define LLVM_IR_Shl 20
#define LLVM_IR_LShr 21
#define LLVM_IR_AShr 22
#define LLVM_IR_And 23
#define LLVM_IR_Or 24
#define LLVM_IR_Xor 25
#define LLVM_IR_Alloca 26
#define LLVM_IR_Load 27
#define LLVM_IR_Store 28
#define LLVM_IR_GetElementPtr 29
#define LLVM_IR_Fence 30
#define LLVM_IR_AtomicCmpXchg 31
#define LLVM_IR_AtomicRMW 32
#define LLVM_IR_Trunc 33
#define LLVM_IR_ZExt 34
#define LLVM_IR_SExt 35
#define LLVM_IR_FPToUI 36
#define LLVM_IR_FPToSI 37
#define LLVM_IR_UIToFP 38
#define LLVM_IR_SIToFP 39
#define LLVM_IR_FPTrunc 40
#define LLVM_IR_FPExt 41
#define LLVM_IR_PtrToInt 42
#define LLVM_IR_IntToPtr 43
#define LLVM_IR_BitCast 44
#define LLVM_IR_AddrSpaceCast 45
#define LLVM_IR_ICmp 46
#define LLVM_IR_FCmp 47
#define LLVM_IR_PHI 48
#define LLVM_IR_Call 49
#define LLVM_IR_Select 50
#define LLVM_IR_VAArg 53
#define LLVM_IR_ExtractElement 54
#define LLVM_IR_InsertElement 55
#define LLVM_IR_ShuffleVector 56
#define LLVM_IR_ExtractValue 57
#define LLVM_IR_InsertValue 58
#define LLVM_IR_LandingPad 59
// Custom opcodes for Aladdin.
#define LLVM_IR_SetReadyBits 95
#define LLVM_IR_EntryDecl 96
#define LLVM_IR_DMAFence 97
#define LLVM_IR_DMAStore 98
#define LLVM_IR_DMALoad 99
#define LLVM_IR_IndexAdd 100
#define LLVM_IR_SilentStore 101
#define LLVM_IR_SpecialMathOp 102
#define LLVM_IR_Intrinsic 104
// clang-format on

#endif
