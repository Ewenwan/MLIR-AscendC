#===--- lit.site.cfg.py ----------------------------------------------------===#
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# MLIR_ASCENDC is under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MLIR-AscendC'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.toy', '.ll', '.tc', '.py', '.yaml', '.test', '.pdll', '.c']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(("%host_cxx", config.host_cxx))
config.substitutions.append(("%host_cc", config.host_cc))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt',
                   'lit.cfg.py', 'lit.site.cfg.py']

# if config.enable_bspriv_common_mlir_ascend:
#     config.available_features.add('enable_bspriv_common_mlir_ascend')

# if config.enable_bspriv_common_mlir_kunpeng:
#     config.available_features.add('enable_bspriv_common_mlir_kunpeng')

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.mlir_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.mlir_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.mlir_ascendc_tools_dir, append_path=True)

tool_dirs = [config.mlir_tools_dir, config.llvm_tools_dir, config.mlir_ascendc_tools_dir]
tools = [
    'ascendc-mlir-opt'
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment['FILECHECK_OPTS'] = "-enable-var-scope --allow-unused-prefixes=false"

if config.enable_assertions:
    config.available_features.add('asserts')
else:
    config.available_features.add('noasserts')
