#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0] -> (s0 * 16)>
#map3 = affine_map<()[s0] -> (s0 * 16 + 1)>
#map4 = affine_map<()[s0] -> (s0 * 16 + 2)>
#map5 = affine_map<()[s0] -> (s0 * 16 + 3)>
#map6 = affine_map<()[s0] -> (s0 * 16 + 4)>
#map7 = affine_map<()[s0] -> (s0 * 16 + 5)>
#map8 = affine_map<()[s0] -> (s0 * 16 + 6)>
#map9 = affine_map<()[s0] -> (s0 * 16 + 7)>
#map10 = affine_map<()[s0] -> (s0 * 16 + 8)>
#map11 = affine_map<()[s0] -> (s0 * 16 + 9)>
#map12 = affine_map<()[s0] -> (s0 * 16 + 10)>
#map13 = affine_map<()[s0] -> (s0 * 16 + 11)>
#map14 = affine_map<()[s0] -> (s0 * 16 + 12)>
#map15 = affine_map<()[s0] -> (s0 * 16 + 13)>
#map16 = affine_map<()[s0] -> (s0 * 16 + 14)>
#map17 = affine_map<()[s0] -> (s0 * 16 + 15)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @phase_1 {
    stream.executable.export public @phase_1 workgroups(%arg0: index) -> (index, index, index) {
      %c2 = arith.constant 2 : index
      %c5 = arith.constant 5 : index
      stream.return %c2, %c5, %arg0 : index, index, index
    }
    builtin.module {
      func.func @phase_1(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: index) attributes {translation_info = #translation} {
        %cst = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %block_id_x = gpu.block_id  x upper_bound 2
        %block_id_y = gpu.block_id  y upper_bound 5
        %block_id_z = gpu.block_id  z
        %thread_id_x = gpu.thread_id  x upper_bound 64
        %0 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<?xi32, strided<[1], offset: ?>>{%arg4}
        %1 = memref.load %0[%block_id_z] : memref<?xi32, strided<[1], offset: ?>>
        %2 = affine.apply #map()[%block_id_z]
        %3 = memref.load %0[%2] : memref<?xi32, strided<[1], offset: ?>>
        %4 = arith.subi %3, %1 : i32
        %5 = arith.index_cast %4 : i32 to index
        %6 = arith.minsi %5, %c8 : index
        %7 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>{%arg4}
        %8 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<8x?x128xf32, strided<[?, 128, 1], offset: ?>>{%arg4}
        %9 = affine.apply #map1()[%thread_id_x, %block_id_x]
        %10 = affine.apply #map2()[%block_id_y]
        %11 = affine.apply #map3()[%block_id_y]
        %12 = affine.apply #map4()[%block_id_y]
        %13 = affine.apply #map5()[%block_id_y]
        %14 = affine.apply #map6()[%block_id_y]
        %15 = affine.apply #map7()[%block_id_y]
        %16 = affine.apply #map8()[%block_id_y]
        %17 = affine.apply #map9()[%block_id_y]
        %18 = affine.apply #map10()[%block_id_y]
        %19 = affine.apply #map11()[%block_id_y]
        %20 = affine.apply #map12()[%block_id_y]
        %21 = affine.apply #map13()[%block_id_y]
        %22 = affine.apply #map14()[%block_id_y]
        %23 = affine.apply #map15()[%block_id_y]
        %24 = affine.apply #map16()[%block_id_y]
        %25 = affine.apply #map17()[%block_id_y]
        %26:18 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %cst, %arg7 = %cst_0, %arg8 = %cst_0, %arg9 = %cst_0, %arg10 = %cst_0, %arg11 = %cst_0, %arg12 = %cst_0, %arg13 = %cst_0, %arg14 = %cst_0, %arg15 = %cst_0, %arg16 = %cst_0, %arg17 = %cst_0, %arg18 = %cst_0, %arg19 = %cst_0, %arg20 = %cst_0, %arg21 = %cst_0, %arg22 = %cst_0, %arg23 = %cst_0) -> (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) {
          %77 = vector.load %7[%arg5, %block_id_z, %9, %10] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %78 = vector.load %7[%arg5, %block_id_z, %9, %11] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %79 = vector.load %7[%arg5, %block_id_z, %9, %12] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %80 = vector.load %7[%arg5, %block_id_z, %9, %13] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %81 = vector.load %7[%arg5, %block_id_z, %9, %14] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %82 = vector.load %7[%arg5, %block_id_z, %9, %15] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %83 = vector.load %7[%arg5, %block_id_z, %9, %16] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %84 = vector.load %7[%arg5, %block_id_z, %9, %17] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %85 = vector.load %7[%arg5, %block_id_z, %9, %18] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %86 = vector.load %7[%arg5, %block_id_z, %9, %19] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %87 = vector.load %7[%arg5, %block_id_z, %9, %20] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %88 = vector.load %7[%arg5, %block_id_z, %9, %21] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %89 = vector.load %7[%arg5, %block_id_z, %9, %22] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %90 = vector.load %7[%arg5, %block_id_z, %9, %23] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %91 = vector.load %7[%arg5, %block_id_z, %9, %24] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %92 = vector.load %7[%arg5, %block_id_z, %9, %25] : memref<8x?x128x80xf32, strided<[?, 10240, 80, 1], offset: ?>>, vector<1xf32>
          %93 = vector.load %8[%arg5, %block_id_z, %9] : memref<8x?x128xf32, strided<[?, 128, 1], offset: ?>>, vector<1xf32>
          %94 = arith.maximumf %93, %arg6 : vector<1xf32>
          %95 = arith.subf %arg6, %94 : vector<1xf32>
          %96 = math.exp2 %95 : vector<1xf32>
          %97 = arith.subf %93, %94 : vector<1xf32>
          %98 = math.exp2 %97 : vector<1xf32>
          %99 = arith.mulf %arg7, %96 : vector<1xf32>
          %100 = arith.addf %99, %98 : vector<1xf32>
          %101 = arith.mulf %arg8, %96 : vector<1xf32>
          %102 = arith.mulf %arg9, %96 : vector<1xf32>
          %103 = arith.mulf %arg10, %96 : vector<1xf32>
          %104 = arith.mulf %arg11, %96 : vector<1xf32>
          %105 = arith.mulf %arg12, %96 : vector<1xf32>
          %106 = arith.mulf %arg13, %96 : vector<1xf32>
          %107 = arith.mulf %arg14, %96 : vector<1xf32>
          %108 = arith.mulf %arg15, %96 : vector<1xf32>
          %109 = arith.mulf %arg16, %96 : vector<1xf32>
          %110 = arith.mulf %arg17, %96 : vector<1xf32>
          %111 = arith.mulf %arg18, %96 : vector<1xf32>
          %112 = arith.mulf %arg19, %96 : vector<1xf32>
          %113 = arith.mulf %arg20, %96 : vector<1xf32>
          %114 = arith.mulf %arg21, %96 : vector<1xf32>
          %115 = arith.mulf %arg22, %96 : vector<1xf32>
          %116 = arith.mulf %arg23, %96 : vector<1xf32>
          %117 = arith.mulf %98, %77 : vector<1xf32>
          %118 = arith.mulf %98, %78 : vector<1xf32>
          %119 = arith.mulf %98, %79 : vector<1xf32>
          %120 = arith.mulf %98, %80 : vector<1xf32>
          %121 = arith.mulf %98, %81 : vector<1xf32>
          %122 = arith.mulf %98, %82 : vector<1xf32>
          %123 = arith.mulf %98, %83 : vector<1xf32>
          %124 = arith.mulf %98, %84 : vector<1xf32>
          %125 = arith.mulf %98, %85 : vector<1xf32>
          %126 = arith.mulf %98, %86 : vector<1xf32>
          %127 = arith.mulf %98, %87 : vector<1xf32>
          %128 = arith.mulf %98, %88 : vector<1xf32>
          %129 = arith.mulf %98, %89 : vector<1xf32>
          %130 = arith.mulf %98, %90 : vector<1xf32>
          %131 = arith.mulf %98, %91 : vector<1xf32>
          %132 = arith.mulf %98, %92 : vector<1xf32>
          %133 = arith.addf %101, %117 : vector<1xf32>
          %134 = arith.addf %102, %118 : vector<1xf32>
          %135 = arith.addf %103, %119 : vector<1xf32>
          %136 = arith.addf %104, %120 : vector<1xf32>
          %137 = arith.addf %105, %121 : vector<1xf32>
          %138 = arith.addf %106, %122 : vector<1xf32>
          %139 = arith.addf %107, %123 : vector<1xf32>
          %140 = arith.addf %108, %124 : vector<1xf32>
          %141 = arith.addf %109, %125 : vector<1xf32>
          %142 = arith.addf %110, %126 : vector<1xf32>
          %143 = arith.addf %111, %127 : vector<1xf32>
          %144 = arith.addf %112, %128 : vector<1xf32>
          %145 = arith.addf %113, %129 : vector<1xf32>
          %146 = arith.addf %114, %130 : vector<1xf32>
          %147 = arith.addf %115, %131 : vector<1xf32>
          %148 = arith.addf %116, %132 : vector<1xf32>
          scf.yield %94, %100, %133, %134, %135, %136, %137, %138, %139, %140, %141, %142, %143, %144, %145, %146, %147, %148 : vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>
        }
        %27 = arith.divf %26#2, %26#1 : vector<1xf32>
        %28 = arith.divf %26#3, %26#1 : vector<1xf32>
        %29 = arith.divf %26#4, %26#1 : vector<1xf32>
        %30 = arith.divf %26#5, %26#1 : vector<1xf32>
        %31 = arith.divf %26#6, %26#1 : vector<1xf32>
        %32 = arith.divf %26#7, %26#1 : vector<1xf32>
        %33 = arith.divf %26#8, %26#1 : vector<1xf32>
        %34 = arith.divf %26#9, %26#1 : vector<1xf32>
        %35 = arith.divf %26#10, %26#1 : vector<1xf32>
        %36 = arith.divf %26#11, %26#1 : vector<1xf32>
        %37 = arith.divf %26#12, %26#1 : vector<1xf32>
        %38 = arith.divf %26#13, %26#1 : vector<1xf32>
        %39 = arith.divf %26#14, %26#1 : vector<1xf32>
        %40 = arith.divf %26#15, %26#1 : vector<1xf32>
        %41 = arith.divf %26#16, %26#1 : vector<1xf32>
        %42 = arith.divf %26#17, %26#1 : vector<1xf32>
        %43 = arith.truncf %27 : vector<1xf32> to vector<1xf16>
        %44 = arith.truncf %28 : vector<1xf32> to vector<1xf16>
        %45 = arith.truncf %29 : vector<1xf32> to vector<1xf16>
        %46 = arith.truncf %30 : vector<1xf32> to vector<1xf16>
        %47 = arith.truncf %31 : vector<1xf32> to vector<1xf16>
        %48 = arith.truncf %32 : vector<1xf32> to vector<1xf16>
        %49 = arith.truncf %33 : vector<1xf32> to vector<1xf16>
        %50 = arith.truncf %34 : vector<1xf32> to vector<1xf16>
        %51 = arith.truncf %35 : vector<1xf32> to vector<1xf16>
        %52 = arith.truncf %36 : vector<1xf32> to vector<1xf16>
        %53 = arith.truncf %37 : vector<1xf32> to vector<1xf16>
        %54 = arith.truncf %38 : vector<1xf32> to vector<1xf16>
        %55 = arith.truncf %39 : vector<1xf32> to vector<1xf16>
        %56 = arith.truncf %40 : vector<1xf32> to vector<1xf16>
        %57 = arith.truncf %41 : vector<1xf32> to vector<1xf16>
        %58 = arith.truncf %42 : vector<1xf32> to vector<1xf16>
        %59 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>{%arg4}
        %60 = affine.apply #map1()[%thread_id_x, %block_id_x]
        %61 = affine.apply #map2()[%block_id_y]
        vector.store %43, %59[%block_id_z, %60, %61] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %62 = affine.apply #map3()[%block_id_y]
        vector.store %44, %59[%block_id_z, %60, %62] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %63 = affine.apply #map4()[%block_id_y]
        vector.store %45, %59[%block_id_z, %60, %63] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %64 = affine.apply #map5()[%block_id_y]
        vector.store %46, %59[%block_id_z, %60, %64] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %65 = affine.apply #map6()[%block_id_y]
        vector.store %47, %59[%block_id_z, %60, %65] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %66 = affine.apply #map7()[%block_id_y]
        vector.store %48, %59[%block_id_z, %60, %66] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %67 = affine.apply #map8()[%block_id_y]
        vector.store %49, %59[%block_id_z, %60, %67] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %68 = affine.apply #map9()[%block_id_y]
        vector.store %50, %59[%block_id_z, %60, %68] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %69 = affine.apply #map10()[%block_id_y]
        vector.store %51, %59[%block_id_z, %60, %69] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %70 = affine.apply #map11()[%block_id_y]
        vector.store %52, %59[%block_id_z, %60, %70] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %71 = affine.apply #map12()[%block_id_y]
        vector.store %53, %59[%block_id_z, %60, %71] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %72 = affine.apply #map13()[%block_id_y]
        vector.store %54, %59[%block_id_z, %60, %72] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %73 = affine.apply #map14()[%block_id_y]
        vector.store %55, %59[%block_id_z, %60, %73] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %74 = affine.apply #map15()[%block_id_y]
        vector.store %56, %59[%block_id_z, %60, %74] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %75 = affine.apply #map16()[%block_id_y]
        vector.store %57, %59[%block_id_z, %60, %75] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        %76 = affine.apply #map17()[%block_id_y]
        vector.store %58, %59[%block_id_z, %60, %76] : memref<?x128x80xf16, strided<[10240, 80, 1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<8x?x128x80xf32>, %arg1: tensor<8x?x128xf32>, %arg2: tensor<?xi32>, %arg3: tensor<?x128x80xf16>, %arg4: index) -> tensor<?x128x80xf16> {
    %0 = flow.dispatch @phase_1::@phase_1[%arg4](%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<8x?x128x80xf32>{%arg4}, tensor<8x?x128xf32>{%arg4}, tensor<?xi32>{%arg4}, tensor<?x128x80xf16>{%arg4}, index) -> %arg3{%arg4}
    return %0 : tensor<?x128x80xf16>
  }
}
