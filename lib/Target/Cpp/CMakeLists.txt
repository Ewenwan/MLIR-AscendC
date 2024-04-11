add_mlir_translation_library(MLIRAscendCToCpp
    TranslateRegistration.cpp
    TranslateAscendCToCpp.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRAscendCDialect
    MLIRFuncDialect
    MLIRIR
    MLIRSCFDialect
    MLIRSupport
    MLIRTranslateLib
)
