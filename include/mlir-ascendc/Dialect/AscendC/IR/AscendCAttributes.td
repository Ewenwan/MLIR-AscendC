//===- AscendCAttrs.td - AscendC attributes definitions ---*- tablegen -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR AscendC dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDCATTRIBUTES
#define MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDCATTRIBUTES

include "mlir-ascendc/Dialect/AscendC/IR/AscendCBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

//===----------------------------------------------------------------------===//
// Attribute Declarations
//===----------------------------------------------------------------------===//

class AscendC_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AscendC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class AscendC_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::ascendc";
}

class AscendC_I32EnumAttr<string mnemonic, AscendC_I32Enum enumInfo,
               list <Trait> traits = []> :
    EnumAttr<AscendC_Dialect, enumInfo, mnemonic, traits> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Tposition
//===----------------------------------------------------------------------===//

def TPosition_GM : I32EnumAttrCase<"GM", 0, "GM">;
def TPosition_VECIN : I32EnumAttrCase<"VECIN", 1, "VECIN">;
def TPosition_VECCALC : I32EnumAttrCase<"VECCALC", 2, "VECCALC">;
def TPosition_VECOUT : I32EnumAttrCase<"VECOUT", 3, "VECOUT">;
def TPosition_A1 : I32EnumAttrCase<"A1", 4, "A1">;
def TPosition_A2 : I32EnumAttrCase<"A2", 5, "A2">;
def TPosition_B1 : I32EnumAttrCase<"B1", 6, "B1">;
def TPosition_B2 : I32EnumAttrCase<"B2", 7, "B2">;
def TPosition_C1 : I32EnumAttrCase<"C1", 8, "C1">;
def TPosition_C2 : I32EnumAttrCase<"C2", 9, "C2">;
def TPosition_CO1 : I32EnumAttrCase<"CO1", 10, "CO1">;
def TPosition_CO2 : I32EnumAttrCase<"CO2", 11, "CO2">;
def TPosition_LCM : I32EnumAttrCase<"LCM", 12, "LCM">;
def TPosition_SPM : I32EnumAttrCase<"SPM", 13, "SPM">;
def TPosition_SHM : I32EnumAttrCase<"SHM", 14, "SHM">;
def TPosition_TSCM : I32EnumAttrCase<"TSCM", 15, "TSCM">;
def TPosition_MAX : I32EnumAttrCase<"MAX", 16, "MAX">;

def AscendC_TPosition_Enum : AscendC_I32Enum<
  "TPosition", "AscendC TPosition", [
    TPosition_GM,
    TPosition_VECIN,
    TPosition_VECCALC,
    TPosition_VECOUT,
    TPosition_A1,
    TPosition_A2,
    TPosition_B1,
    TPosition_B2,
    TPosition_C1,
    TPosition_C2,
    TPosition_CO1,
    TPosition_CO2,
    TPosition_LCM,
    TPosition_SPM,
    TPosition_SHM,
    TPosition_TSCM,
    TPosition_MAX
  ]>;

def AscendC_TPosition_Attr : AscendC_I32EnumAttr<"TPosition", AscendC_TPosition_Enum,
    [DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let description = [{
    Use TPosition to abstract real physical hardware to manage different
    levels of physical memory. GM is Global Memory, VECIN/VECCALC/VECOUT
    is Unified Buffer, A1 is L1 Buffer, A2 is L0A Buffer, B1 is L1 Buffer,
    B2 is L0B Buffer, CO1 is L0C Buffer, CO2 is UB/GM based on different
    products.
  }];
}

//===----------------------------------------------------------------------===//
// Hardware
//===----------------------------------------------------------------------===//

def Hardware_GM : I32EnumAttrCase<"GM", 0, "GM">;
def Hardware_UB : I32EnumAttrCase<"UB", 1, "UB">;
def Hardware_L1 : I32EnumAttrCase<"L1", 2, "L1">;
def Hardware_L0A : I32EnumAttrCase<"L0A", 3, "L0A">;
def Hardware_L0B : I32EnumAttrCase<"L0B ", 4, "L0B">;
def Hardware_L0C : I32EnumAttrCase<"L0C", 5, "L0C">;
def Hardware_BIAS : I32EnumAttrCase<"BIAS", 6, "BIAS">;
def Hardware_FIXBUF : I32EnumAttrCase<"FIXBUF", 7, "FIXBUF">;
def Hardware_MAX : I32EnumAttrCase<"MAX", 8, "MAX">;

def AscendC_Hardware_Enum : AscendC_I32Enum<
  "Hardware", "AscendC Hardware", [
    Hardware_GM,
    Hardware_UB,
    Hardware_L1,
    Hardware_L0A,
    Hardware_L0B,
    Hardware_L0C,
    Hardware_BIAS,
    Hardware_FIXBUF,
    Hardware_MAX
  ]>;

def AscendC_Hardware_Attr : AscendC_I32EnumAttr<"Hardware", AscendC_Hardware_Enum>  {
  let description = [{
    Use Hardware to represent actual physical hardware. Unified Buffer requires 32 Byte
    aligned; L1 Buffer needs 512 Byte aligned; L1OUT buffer needs 512 Byte aligned for
    half type and 1024 Byte aligned for float/int32_t/uint32_t.
  }];
}

#endif // MLIRASCENDC_DIALECT_ASCENDC_IR_ASCENDCATTRIBUTES
