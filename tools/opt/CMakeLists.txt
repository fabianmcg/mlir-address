get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIRArithDialect MLIROptLib)

add_llvm_executable(dev-opt main.cpp)

llvm_update_compile_flags(dev-opt)
target_link_libraries(dev-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(dev-opt)
