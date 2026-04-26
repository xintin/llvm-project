# -*- Python -*-
"""Lit configuration for Salmon transpiler FileCheck tests (loaded by lit.site.cfg.py)."""

import os

import lit.formats

config.name = "Salmon Transpiler"
config.test_format = lit.formats.ShTest(execute_external=True)
# Each AMDGPU assembly fixture is self-testing: the top of every `.s`
# carries the lit RUN / CHECK directives, followed by the actual kernel
# bytes. The RUN line invokes `llvm-mc` + `ld.lld` itself to assemble
# the source into an amdhsa code object, then hands the hsaco to
# `raise_cli` for lifting. There are no separate `.ll` test files —
# `%s` in a RUN line expands to the assembly fixture itself, which
# doubles as the FileCheck input.
config.suffixes = [".s"]

# `lit.site.cfg.py` is the intended entry point and sets both roots. Only
# fall back to source-tree defaults when `lit.cfg.py` is loaded directly
# (rare — e.g. running `llvm-lit lit_tests/` without reconfiguring). In
# that fallback path we intentionally route `test_exec_root` under the
# user's temp dir so an ad-hoc run never writes `Output/` into the
# source tree.
if not getattr(config, "test_source_root", None):
    config.test_source_root = os.path.dirname(os.path.abspath(__file__))
if not getattr(config, "test_exec_root", None):
    import tempfile
    config.test_exec_root = os.path.join(
        tempfile.gettempdir(), "salmon-transpiler-lit"
    )

config.substitutions.append(("%raise_cli", config.raise_cli))
config.substitutions.append(("%FileCheck", config.file_check))
config.substitutions.append(("%not", config.not_tool))
# `%llvm_mc` bakes in the fixed triple + filetype so each RUN line only
# has to vary `-mcpu=<isa>`. `%ld_lld` is the raw linker path; the test
# supplies `-shared` explicitly so the hsaco shape stays visible.
config.substitutions.append(
    ("%llvm_mc", config.llvm_mc + " -triple=amdgcn-amd-amdhsa -filetype=obj")
)
config.substitutions.append(("%ld_lld", config.ld_lld))
