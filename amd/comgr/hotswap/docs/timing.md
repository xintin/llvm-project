# HotSwap Timing Diagnostics

`HSA_HOTSWAP_TIMING=1` enables opt-in diagnostic timing for the COMGR-backed
HotSwap load path.

When enabled through `amd_comgr_hotswap_transpile_with_options`, the returned
HotSwap result may expose `AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TIMING_JSON`.
That field is a JSON object encoded as a string. Its keys are best-effort
diagnostic counters for cache lookup, cache-key construction, pipeline stages
such as raise/`llc`/`llvm-mc`/link, cache write, and output wrapping.

The timing field is intentionally not emitted by default. Production cache-hit
loads should not pay instrumentation overhead, and callers must treat the JSON
keys as diagnostic telemetry rather than semantic ABI.
