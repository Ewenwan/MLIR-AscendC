configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MLIR_ASCENDC_TEST_DEPENDS
  FileCheck count not
  ascendc-mlir-opt
  ascendc-mlir-translate
)

add_lit_testsuite(check-mlir-ascendc "Running the mlir-ascendc regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${MLIR_ASCENDC_TEST_DEPENDS}
)
set_target_properties(check-mlir-ascendc PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_ASCENDC ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${MLIR_ASCENDC_TEST_DEPENDS}
)
