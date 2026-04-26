; RUN: %raise_cli %global_load_async_to_lds_offset_co --isa=gfx1250 \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_offset_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Cross-target fixture for the non-zero `flat_offset` arm of
; `GLOBAL_LOAD_ASYNC_TO_LDS_B32`.  Companion to
; `global_load_async_to_lds/global_load_async_to_lds.ll`, which pins the
; zero-offset b32/b64/b128 shapes.  This fixture narrows in on the
; semantics-bearing branch where `handle_flat.cpp` must GEP the decoded
; `flat_offset` onto BOTH the global pointer and the LDS pointer before
; emitting the synchronous `load i32` + `store i32` pair.
;
; === ISA source-of-truth ===
;
; `instruction_manual.pdf §13.6.10`:
;
;   dsaddr = LDS_BASE.b32 + VGPR[laneId][VDST.u32] + INST_OFFSET.b32;
;   memaddr = ADDR;
;   // Address computed the same as for other GLOBAL instructions
;
; `programming_manual.pdf §4.9.9.1`:
;
;   LDS[VGPR[VDST][lane] + byte + INST_OFFSET] =
;   GLOBAL_MEMORY[VGPR[VADDR][lane] + INST_OFFSET + byte];
;
; The offset therefore is not a global-address-only modifier.  Removing
; the LDS-side GEP would silently write the loaded value into the wrong
; LDS slot for every non-zero `INST_OFFSET`.

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 4
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gptr_off{{[0-9]*}} = getelementptr i8, ptr addrspace(1) %{{[0-9]+}}, i64 16
; IR: %async_lptr_off{{[0-9]*}} = getelementptr i8, ptr addrspace(3) %lds_ptr{{[0-9]*}}, i64 16
; IR: %async_gload{{[0-9]*}} = load i32, ptr addrspace(1) %async_gptr_off{{[0-9]*}}, align 4
; IR: store i32 %async_gload{{[0-9]*}}, ptr addrspace(3) %async_lptr_off{{[0-9]*}}, align 4

; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b32
