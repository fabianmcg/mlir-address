set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Address)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Address)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Address)
add_public_tablegen_target(MLIRAddressPassIncGen)

add_mlir_doc(Passes AddressPasses ./ -gen-pass-doc)
