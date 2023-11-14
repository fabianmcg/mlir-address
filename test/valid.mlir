func.func @foo() {
  %null = addr.constant 0
  addr.type_offset f32 : i32
  %memref = addr.to_memref %null base %null : memref<i32>
  %dst = memref.reinterpret_cast %memref to
  offset: [0],
  sizes: [2],
  strides: [1] : memref<i32> to memref<2xi32, strided<[1], offset: 0>>
  %base = addr.from_memref extract_base [%memref : memref<i32>]
  return
}

func.func @bar(%arg : memref<i32>) -> memref<i32> {
  %base = addr.from_memref extract_base [%arg : memref<i32>]
  %aligned = addr.from_memref [%arg : memref<i32>]
  %c10 = arith.constant 10 : i32
  %sizeF32 = addr.type_offset f32 : i32
  %byteOff = arith.muli %c10, %sizeF32 : i32
  %ptr = addr.ptradd %aligned : !addr.address, %byteOff : i32
  %memref = addr.to_memref %ptr base %base : memref<i32>
  return %memref : memref<i32>
}

func.func @baz(%arg : memref<i32>) -> memref<i32> {
  %memref = func.call @bar(%arg) : (memref<i32>) -> memref<i32>
  return %memref : memref<i32>
}

func.func @foz(%arg : memref<i32>) -> !addr.address {
  %memref = func.call @bar(%arg) : (memref<i32>) -> memref<i32>
  %base = addr.from_memref extract_base [%memref : memref<i32>]
  return %base : !addr.address
}