#map = affine_map<()[s0] -> (s0 + 1)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @phase_1 {
    stream.executable.export public @phase_1 workgroups(%arg0: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      stream.return %c1, %c16, %arg0 : index, index, index
    }
    builtin.module {
      func.func @phase_1(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: index) attributes {translation_info = #translation} {
        %cst = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %block_id_y = gpu.block_id  y upper_bound 16
        %block_id_z = gpu.block_id  z
        %thread_id_x = gpu.thread_id  x upper_bound 64
        %0 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<?xi32, strided<[1], offset: ?>>{%arg4}
        %1 = memref.load %0[%block_id_z] : memref<?xi32, strided<[1], offset: ?>>
        %2 = affine.apply #map()[%block_id_z]
        %3 = memref.load %0[%2] : memref<?xi32, strided<[1], offset: ?>>
        %4 = arith.subi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = arith.minsi %5, %c8 : index
        %7 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<8x?x16x64xf32, strided<[?, 1024, 64, 1], offset: ?>>{%arg4}
        %8 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x?x16xf32, strided<[?, 16, 1], offset: ?>>{%arg4}
        %9:3 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst_0, %arg8 = %cst_0) -> (vector<1xf32>, vector<1xf32>, vector<1xf32>) {
          %13 = vector.load %7[%arg5, %block_id_z, %block_id_y, %thread_id_x] : memref<8x?x16x64xf32, strided<[?, 1024, 64, 1], offset: ?>>, vector<1xf32>
          %14 = vector.load %8[%arg5, %block_id_z, %block_id_y] : memref<8x?x16xf32, strided<[?, 16, 1], offset: ?>>, vector<1xf32>
          %15 = arith.maximumf %14, %arg6 : vector<1xf32>
          %16 = arith.subf %arg6, %15 : vector<1xf32>
          %17 = math.exp2 %16 : vector<1xf32>
          %18 = arith.subf %14, %15 : vector<1xf32>
          %19 = math.exp2 %18 : vector<1xf32>
          %20 = arith.mulf %arg7, %17 : vector<1xf32>
          %21 = arith.addf %20, %19 : vector<1xf32>
          %22 = arith.mulf %arg8, %17 : vector<1xf32>
          %23 = arith.mulf %19, %13 : vector<1xf32>
          %24 = arith.addf %22, %23 : vector<1xf32>
          scf.yield %15, %21, %24 : vector<1xf32>, vector<1xf32>, vector<1xf32>
        }
        %10 = arith.divf %9#2, %9#1 : vector<1xf32>
        %11 = arith.truncf %10 : vector<1xf32> to vector<1xf16>
        %12 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>{%arg4}
        vector.store %11, %12[%block_id_z, %block_id_y, %thread_id_x] : memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<8x?x16x64xf32>, %arg1: tensor<8x?x16xf32>, %arg2: tensor<?xi32>, %arg3: tensor<?x16x64xf16>, %arg4: index) -> tensor<?x16x64xf16> {
    %0 = flow.dispatch @phase_1::@phase_1[%arg4](%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<8x?x16x64xf32>{%arg4}, tensor<8x?x16xf32>{%arg4}, tensor<?xi32>{%arg4}, tensor<?x16x64xf16>{%arg4}, index) -> %arg3{%arg4}
    return %0 : tensor<?x16x64xf16>
  }
}
