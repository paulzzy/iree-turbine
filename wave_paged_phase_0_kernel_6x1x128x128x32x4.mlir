#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 mod 16)>
#map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
#map3 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
#map4 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 32)>
#map5 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 48)>
#map6 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 64)>
#map7 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 80)>
#map8 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 96)>
#map9 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 112)>
#map10 = affine_map<()[s0] -> (s0 ceildiv 16)>
#map11 = affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 16) * 128)>
#map12 = affine_map<()[s0] -> ((s0 floordiv 16) mod 16)>
#map13 = affine_map<()[s0] -> (s0 floordiv 16 - ((s0 floordiv 16 + 4) floordiv 16) * 16 + 4)>
#map14 = affine_map<()[s0] -> (s0 floordiv 16 - ((s0 floordiv 16 + 8) floordiv 16) * 16 + 8)>
#map15 = affine_map<()[s0] -> (s0 floordiv 16 - ((s0 floordiv 16 + 12) floordiv 16) * 16 + 12)>
#map16 = affine_map<()[s0] -> (s0 mod 16 + 16)>
#map17 = affine_map<()[s0] -> (s0 mod 16 + 32)>
#map18 = affine_map<()[s0] -> (s0 mod 16 + 48)>
#map19 = affine_map<()[s0] -> (s0 mod 16 + 64)>
#map20 = affine_map<()[s0] -> (s0 mod 16 + 80)>
#map21 = affine_map<()[s0] -> (s0 mod 16 + 96)>
#map22 = affine_map<()[s0] -> (s0 mod 16 + 112)>
#map23 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map24 = affine_map<()[s0, s1, s2, s3] -> (s0 * 16 + s2 + s3 + ((s1 mod 64) floordiv 16) * 4)>
#map25 = affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 + s3 + (s0 floordiv 16) mod 16)>
#map26 = affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 + s3 + s0 floordiv 16 - ((s0 floordiv 16 + 12) floordiv 16) * 16 + 12)>
#map27 = affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 + s3 + s0 floordiv 16 - ((s0 floordiv 16 + 8) floordiv 16) * 16 + 8)>
#map28 = affine_map<()[s0, s1, s2, s3] -> (s1 * 16 + s2 + s3 + s0 floordiv 16 - ((s0 floordiv 16 + 4) floordiv 16) * 16 + 4)>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [64, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @phase_0 {
    stream.executable.export public @phase_0 workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      stream.return %arg2, %c1, %c8 : index, index, index
    }
    builtin.module {
      func.func @phase_0(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding, %arg5: !stream.binding, %arg6: !stream.binding, %arg7: index, %arg8: index, %arg9: index) attributes {translation_info = #translation} {
        %cst = arith.constant dense<0.0231049061> : vector<4xf32>
        %cst_0 = arith.constant dense<112> : vector<4xindex>
        %cst_1 = arith.constant dense<96> : vector<4xindex>
        %cst_2 = arith.constant dense<80> : vector<4xindex>
        %cst_3 = arith.constant dense<64> : vector<4xindex>
        %cst_4 = arith.constant dense<48> : vector<4xindex>
        %cst_5 = arith.constant dense<32> : vector<4xindex>
        %cst_6 = arith.constant dense<16> : vector<4xindex>
        %cst_7 = arith.constant dense<0.000000e+00> : vector<1xbf16>
        %cst_8 = arith.constant dense<128> : vector<4xindex>
        %cst_9 = arith.constant dense<0.000000e+00> : vector<8xbf16>
        %cst_10 = arith.constant dense<0> : vector<1xi32>
        %cst_11 = arith.constant dense<0> : vector<4xi32>
        %cst_12 = arith.constant dense<0.000000e+00> : vector<4xbf16>
        %cst_13 = arith.constant dense<8> : vector<1xindex>
        %cst_14 = arith.constant dense<1.000000e+00> : vector<1xf32>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c16_i32 = arith.constant 16 : i32
        %cst_15 = arith.constant dense<-1.000000e+00> : vector<4xf32>
        %cst_16 = arith.constant dense<-2.885390e+00> : vector<4xf32>
        %cst_17 = arith.constant dense<1.000000e+00> : vector<4xf32>
        %c1073741822 = arith.constant 1073741822 : index
        %cst_18 = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
        %c6 = arith.constant 6 : index
        %c5120 = arith.constant 5120 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %cst_19 = arith.constant dense<-1.000000e+06> : vector<4xf32>
        %cst_20 = arith.constant dense<0.000000e+00> : vector<4xf32>
        %cst_21 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %cst_22 = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_23 = arith.constant dense<43.2808495> : vector<4xf32>
        %cst_24 = arith.constant dense<0.127517432> : vector<4xf32>
        %block_id_x = gpu.block_id  x
        %block_id_z = gpu.block_id  z upper_bound 8
        %thread_id_x = gpu.thread_id  x upper_bound 64
        %alloc = memref.alloc() : memref<9344xi8, #gpu.address_space<workgroup>>
        %0 = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<?xi32, strided<[1], offset: ?>>{%arg9}
        %1 = vector.load %0[%block_id_x] : memref<?xi32, strided<[1], offset: ?>>, vector<1xi32>
        %2 = affine.apply #map()[%block_id_x]
        %3 = vector.load %0[%2] : memref<?xi32, strided<[1], offset: ?>>, vector<1xi32>
        %4 = vector.extract %3[0] : i32 from vector<1xi32>
        %5 = arith.index_cast %4 : i32 to index
        %6 = arith.subi %3, %1 : vector<1xi32>
        %7 = vector.extract %1[0] : i32 from vector<1xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = arith.index_cast %6 : vector<1xi32> to vector<1xindex>
        %10 = arith.ceildivsi %9, %cst_13 : vector<1xindex>
        %11 = arith.index_cast %10 : vector<1xindex> to vector<1xi32>
        %12 = vector.splat %block_id_z : vector<1xindex>
        %13 = arith.index_cast %12 : vector<1xindex> to vector<1xi32>
        %14 = arith.muli %13, %11 : vector<1xi32>
        %15 = vector.extract %14[0] : i32 from vector<1xi32>
        %16 = arith.index_cast %15 : i32 to index
        %17 = vector.extract %11[0] : i32 from vector<1xi32>
        %18 = arith.index_cast %17 : i32 to index
        %19 = vector.extract %6[0] : i32 from vector<1xi32>
        %20 = arith.index_cast %19 : i32 to index
        %21 = arith.subi %20, %16 : index
        %22 = arith.maxsi %21, %c0 : index
        %23 = arith.minsi %22, %18 : index
        %view = memref.view %alloc[%c0][] : memref<9344xi8, #gpu.address_space<workgroup>> to memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>
        %view_25 = memref.view %alloc[%c5120][] : memref<9344xi8, #gpu.address_space<workgroup>> to memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>
        %24 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>{%arg9}
        %25 = affine.apply #map1()[%thread_id_x]
        %26 = affine.apply #map2()[%thread_id_x]
        %27 = arith.cmpi slt, %25, %c6 : index
        %28 = vector.splat %27 : vector<4xi1>
        %29 = vector.maskedload %24[%block_id_x, %25, %26], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %30 = affine.apply #map3()[%thread_id_x]
        %31 = vector.maskedload %24[%block_id_x, %25, %30], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %32 = affine.apply #map4()[%thread_id_x]
        %33 = vector.maskedload %24[%block_id_x, %25, %32], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %34 = affine.apply #map5()[%thread_id_x]
        %35 = vector.maskedload %24[%block_id_x, %25, %34], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %36 = affine.apply #map6()[%thread_id_x]
        %37 = vector.maskedload %24[%block_id_x, %25, %36], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %38 = affine.apply #map7()[%thread_id_x]
        %39 = vector.maskedload %24[%block_id_x, %25, %38], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %40 = affine.apply #map8()[%thread_id_x]
        %41 = vector.maskedload %24[%block_id_x, %25, %40], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %42 = affine.apply #map9()[%thread_id_x]
        %43 = vector.maskedload %24[%block_id_x, %25, %42], %28, %cst_12 : memref<?x6x128xbf16, strided<[768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xbf16> into vector<4xbf16>
        %44 = affine.apply #map10()[%23]
        %45 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>>{%arg8}
        %46 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>>{%arg8}
        %47 = stream.binding.subspan %arg4[%c0] : !stream.binding -> memref<?xi32, strided<[1], offset: ?>>{%arg7}
        %48 = vector.splat %5 : vector<4xindex>
        %49 = affine.apply #map11()[%thread_id_x]
        %50 = affine.apply #map12()[%thread_id_x]
        %51 = affine.apply #map13()[%thread_id_x]
        %52 = affine.apply #map14()[%thread_id_x]
        %53 = affine.apply #map15()[%thread_id_x]
        %54 = vector.splat %25 : vector<4xindex>
        %reinterpret_cast = memref.reinterpret_cast %45 to offset: [%c0], sizes: [%c1073741822], strides: [1] : memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
        %55 = affine.apply #map16()[%thread_id_x]
        %56 = affine.apply #map17()[%thread_id_x]
        %57 = affine.apply #map18()[%thread_id_x]
        %58 = affine.apply #map19()[%thread_id_x]
        %59 = affine.apply #map20()[%thread_id_x]
        %60 = affine.apply #map21()[%thread_id_x]
        %61 = affine.apply #map22()[%thread_id_x]
        %62 = affine.apply #map23()[%16, %23, %8]
        %63 = vector.splat %62 : vector<4xindex>
        %64:10 = scf.for %arg10 = %c0 to %44 step %c1 iter_args(%arg11 = %cst_22, %arg12 = %cst_21, %arg13 = %cst_20, %arg14 = %cst_20, %arg15 = %cst_20, %arg16 = %cst_20, %arg17 = %cst_20, %arg18 = %cst_20, %arg19 = %cst_20, %arg20 = %cst_20) -> (vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
          %66 = affine.apply #map24()[%arg10, %thread_id_x, %16, %8]
          %67 = vector.splat %66 : vector<4xindex>
          %68 = arith.addi %67, %cst_18 overflow<nsw, nuw> : vector<4xindex>
          %69 = arith.cmpi slt, %68, %48 : vector<4xindex>
          %70 = vector.maskedload %47[%66], %69, %cst_11 : memref<?xi32, strided<[1], offset: ?>>, vector<4xi1>, vector<4xi32> into vector<4xi32>
          %71 = affine.apply #map25()[%thread_id_x, %arg10, %16, %8]
          %72 = arith.cmpi slt, %71, %5 : index
          %73 = vector.splat %72 : vector<1xi1>
          %74 = vector.maskedload %47[%71], %73, %cst_10 : memref<?xi32, strided<[1], offset: ?>>, vector<1xi1>, vector<1xi32> into vector<1xi32>
          %75 = affine.apply #map26()[%thread_id_x, %arg10, %16, %8]
          %76 = arith.cmpi slt, %75, %5 : index
          %77 = vector.splat %76 : vector<1xi1>
          %78 = vector.maskedload %47[%75], %77, %cst_10 : memref<?xi32, strided<[1], offset: ?>>, vector<1xi1>, vector<1xi32> into vector<1xi32>
          %79 = affine.apply #map27()[%thread_id_x, %arg10, %16, %8]
          %80 = arith.cmpi slt, %79, %5 : index
          %81 = vector.splat %80 : vector<1xi1>
          %82 = vector.maskedload %47[%79], %81, %cst_10 : memref<?xi32, strided<[1], offset: ?>>, vector<1xi1>, vector<1xi32> into vector<1xi32>
          %83 = affine.apply #map28()[%thread_id_x, %arg10, %16, %8]
          %84 = arith.cmpi slt, %83, %5 : index
          %85 = vector.splat %84 : vector<1xi1>
          %86 = vector.maskedload %47[%83], %85, %cst_10 : memref<?xi32, strided<[1], offset: ?>>, vector<1xi1>, vector<1xi32> into vector<1xi32>
          %87 = vector.extract %74[0] : i32 from vector<1xi32>
          %88 = arith.index_cast %87 : i32 to index
          %89 = vector.splat %72 : vector<8xi1>
          %90 = vector.maskedload %46[%88, %c0, %49], %89, %cst_9 : memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
          amdgpu.lds_barrier
          vector.store %90, %view_25[%c0, %c0, %50, %49] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %91 = vector.extract %86[0] : i32 from vector<1xi32>
          %92 = arith.index_cast %91 : i32 to index
          %93 = vector.splat %84 : vector<8xi1>
          %94 = vector.maskedload %46[%92, %c0, %49], %93, %cst_9 : memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
          vector.store %94, %view_25[%c0, %c0, %51, %49] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %95 = vector.extract %82[0] : i32 from vector<1xi32>
          %96 = arith.index_cast %95 : i32 to index
          %97 = vector.splat %80 : vector<8xi1>
          %98 = vector.maskedload %46[%96, %c0, %49], %97, %cst_9 : memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
          vector.store %98, %view_25[%c0, %c0, %52, %49] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %99 = vector.extract %78[0] : i32 from vector<1xi32>
          %100 = arith.index_cast %99 : i32 to index
          %101 = vector.splat %76 : vector<8xi1>
          %102 = vector.maskedload %46[%100, %c0, %49], %101, %cst_9 : memref<?x1x128xbf16, strided<[128, 128, 1], offset: ?>>, vector<8xi1>, vector<8xbf16> into vector<8xbf16>
          vector.store %102, %view_25[%c0, %c0, %53, %49] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
          %103 = arith.index_cast %70 : vector<4xi32> to vector<4xindex>
          %104 = arith.muli %103, %cst_8 overflow<nsw, nuw> : vector<4xindex>
          %105 = arith.addi %54, %104 overflow<nsw, nuw> : vector<4xindex>
          %106 = vector.extract %69[0] : i1 from vector<4xi1>
          %107 = vector.extract %105[0] : index from vector<4xindex>
          %108 = vector.splat %106 : vector<1xi1>
          %109 = vector.maskedload %reinterpret_cast[%107], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %110 = vector.extract %69[1] : i1 from vector<4xi1>
          %111 = vector.extract %105[1] : index from vector<4xindex>
          %112 = vector.splat %110 : vector<1xi1>
          %113 = vector.maskedload %reinterpret_cast[%111], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %114 = vector.extract %69[2] : i1 from vector<4xi1>
          %115 = vector.extract %105[2] : index from vector<4xindex>
          %116 = vector.splat %114 : vector<1xi1>
          %117 = vector.maskedload %reinterpret_cast[%115], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %118 = vector.extract %69[3] : i1 from vector<4xi1>
          %119 = vector.extract %105[3] : index from vector<4xindex>
          %120 = vector.splat %118 : vector<1xi1>
          %121 = vector.maskedload %reinterpret_cast[%119], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %122 = vector.extract %109[0] : bf16 from vector<1xbf16>
          %123 = vector.extract %113[0] : bf16 from vector<1xbf16>
          %124 = vector.extract %117[0] : bf16 from vector<1xbf16>
          %125 = vector.extract %121[0] : bf16 from vector<1xbf16>
          %126 = vector.from_elements %122, %123, %124, %125 : vector<4xbf16>
          %127 = arith.addi %105, %cst_6 overflow<nsw, nuw> : vector<4xindex>
          %128 = vector.extract %127[0] : index from vector<4xindex>
          %129 = vector.maskedload %reinterpret_cast[%128], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %130 = vector.extract %127[1] : index from vector<4xindex>
          %131 = vector.maskedload %reinterpret_cast[%130], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %132 = vector.extract %127[2] : index from vector<4xindex>
          %133 = vector.maskedload %reinterpret_cast[%132], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %134 = vector.extract %127[3] : index from vector<4xindex>
          %135 = vector.maskedload %reinterpret_cast[%134], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %136 = vector.extract %129[0] : bf16 from vector<1xbf16>
          %137 = vector.extract %131[0] : bf16 from vector<1xbf16>
          %138 = vector.extract %133[0] : bf16 from vector<1xbf16>
          %139 = vector.extract %135[0] : bf16 from vector<1xbf16>
          %140 = vector.from_elements %136, %137, %138, %139 : vector<4xbf16>
          %141 = arith.addi %105, %cst_5 overflow<nsw, nuw> : vector<4xindex>
          %142 = vector.extract %141[0] : index from vector<4xindex>
          %143 = vector.maskedload %reinterpret_cast[%142], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %144 = vector.extract %141[1] : index from vector<4xindex>
          %145 = vector.maskedload %reinterpret_cast[%144], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %146 = vector.extract %141[2] : index from vector<4xindex>
          %147 = vector.maskedload %reinterpret_cast[%146], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %148 = vector.extract %141[3] : index from vector<4xindex>
          %149 = vector.maskedload %reinterpret_cast[%148], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %150 = vector.extract %143[0] : bf16 from vector<1xbf16>
          %151 = vector.extract %145[0] : bf16 from vector<1xbf16>
          %152 = vector.extract %147[0] : bf16 from vector<1xbf16>
          %153 = vector.extract %149[0] : bf16 from vector<1xbf16>
          %154 = vector.from_elements %150, %151, %152, %153 : vector<4xbf16>
          %155 = arith.addi %105, %cst_4 overflow<nsw, nuw> : vector<4xindex>
          %156 = vector.extract %155[0] : index from vector<4xindex>
          %157 = vector.maskedload %reinterpret_cast[%156], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %158 = vector.extract %155[1] : index from vector<4xindex>
          %159 = vector.maskedload %reinterpret_cast[%158], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %160 = vector.extract %155[2] : index from vector<4xindex>
          %161 = vector.maskedload %reinterpret_cast[%160], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %162 = vector.extract %155[3] : index from vector<4xindex>
          %163 = vector.maskedload %reinterpret_cast[%162], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %164 = vector.extract %157[0] : bf16 from vector<1xbf16>
          %165 = vector.extract %159[0] : bf16 from vector<1xbf16>
          %166 = vector.extract %161[0] : bf16 from vector<1xbf16>
          %167 = vector.extract %163[0] : bf16 from vector<1xbf16>
          %168 = vector.from_elements %164, %165, %166, %167 : vector<4xbf16>
          %169 = arith.addi %105, %cst_3 overflow<nsw, nuw> : vector<4xindex>
          %170 = vector.extract %169[0] : index from vector<4xindex>
          %171 = vector.maskedload %reinterpret_cast[%170], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %172 = vector.extract %169[1] : index from vector<4xindex>
          %173 = vector.maskedload %reinterpret_cast[%172], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %174 = vector.extract %169[2] : index from vector<4xindex>
          %175 = vector.maskedload %reinterpret_cast[%174], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %176 = vector.extract %169[3] : index from vector<4xindex>
          %177 = vector.maskedload %reinterpret_cast[%176], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %178 = vector.extract %171[0] : bf16 from vector<1xbf16>
          %179 = vector.extract %173[0] : bf16 from vector<1xbf16>
          %180 = vector.extract %175[0] : bf16 from vector<1xbf16>
          %181 = vector.extract %177[0] : bf16 from vector<1xbf16>
          %182 = vector.from_elements %178, %179, %180, %181 : vector<4xbf16>
          %183 = arith.addi %105, %cst_2 overflow<nsw, nuw> : vector<4xindex>
          %184 = vector.extract %183[0] : index from vector<4xindex>
          %185 = vector.maskedload %reinterpret_cast[%184], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %186 = vector.extract %183[1] : index from vector<4xindex>
          %187 = vector.maskedload %reinterpret_cast[%186], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %188 = vector.extract %183[2] : index from vector<4xindex>
          %189 = vector.maskedload %reinterpret_cast[%188], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %190 = vector.extract %183[3] : index from vector<4xindex>
          %191 = vector.maskedload %reinterpret_cast[%190], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %192 = vector.extract %185[0] : bf16 from vector<1xbf16>
          %193 = vector.extract %187[0] : bf16 from vector<1xbf16>
          %194 = vector.extract %189[0] : bf16 from vector<1xbf16>
          %195 = vector.extract %191[0] : bf16 from vector<1xbf16>
          %196 = vector.from_elements %192, %193, %194, %195 : vector<4xbf16>
          %197 = arith.addi %105, %cst_1 overflow<nsw, nuw> : vector<4xindex>
          %198 = vector.extract %197[0] : index from vector<4xindex>
          %199 = vector.maskedload %reinterpret_cast[%198], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %200 = vector.extract %197[1] : index from vector<4xindex>
          %201 = vector.maskedload %reinterpret_cast[%200], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %202 = vector.extract %197[2] : index from vector<4xindex>
          %203 = vector.maskedload %reinterpret_cast[%202], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %204 = vector.extract %197[3] : index from vector<4xindex>
          %205 = vector.maskedload %reinterpret_cast[%204], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %206 = vector.extract %199[0] : bf16 from vector<1xbf16>
          %207 = vector.extract %201[0] : bf16 from vector<1xbf16>
          %208 = vector.extract %203[0] : bf16 from vector<1xbf16>
          %209 = vector.extract %205[0] : bf16 from vector<1xbf16>
          %210 = vector.from_elements %206, %207, %208, %209 : vector<4xbf16>
          %211 = arith.addi %105, %cst_0 overflow<nsw, nuw> : vector<4xindex>
          %212 = vector.extract %211[0] : index from vector<4xindex>
          %213 = vector.maskedload %reinterpret_cast[%212], %108, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %214 = vector.extract %211[1] : index from vector<4xindex>
          %215 = vector.maskedload %reinterpret_cast[%214], %112, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %216 = vector.extract %211[2] : index from vector<4xindex>
          %217 = vector.maskedload %reinterpret_cast[%216], %116, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %218 = vector.extract %211[3] : index from vector<4xindex>
          %219 = vector.maskedload %reinterpret_cast[%218], %120, %cst_7 : memref<?xbf16, strided<[1], offset: ?>>, vector<1xi1>, vector<1xbf16> into vector<1xbf16>
          %220 = vector.extract %213[0] : bf16 from vector<1xbf16>
          %221 = vector.extract %215[0] : bf16 from vector<1xbf16>
          %222 = vector.extract %217[0] : bf16 from vector<1xbf16>
          %223 = vector.extract %219[0] : bf16 from vector<1xbf16>
          %224 = vector.from_elements %220, %221, %222, %223 : vector<4xbf16>
          vector.store %126, %view[%c0, %c0, %25, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %140, %view[%c0, %c0, %55, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %154, %view[%c0, %c0, %56, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %168, %view[%c0, %c0, %57, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %182, %view[%c0, %c0, %58, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %196, %view[%c0, %c0, %59, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %210, %view[%c0, %c0, %60, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          vector.store %224, %view[%c0, %c0, %61, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          amdgpu.lds_barrier
          %225 = vector.load %view[%c0, %c0, %25, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %226 = vector.load %view[%c0, %c0, %55, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %227 = vector.load %view[%c0, %c0, %56, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %228 = vector.load %view[%c0, %c0, %57, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %229 = vector.load %view[%c0, %c0, %58, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %230 = vector.load %view[%c0, %c0, %59, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %231 = vector.load %view[%c0, %c0, %60, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %232 = vector.load %view[%c0, %c0, %61, %26] : memref<1x1x128x20xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %233 = vector.load %view_25[%c0, %c0, %25, %26] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %234 = vector.load %view_25[%c0, %c0, %25, %30] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %235 = vector.load %view_25[%c0, %c0, %25, %32] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %236 = vector.load %view_25[%c0, %c0, %25, %34] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %237 = vector.load %view_25[%c0, %c0, %25, %36] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %238 = vector.load %view_25[%c0, %c0, %25, %38] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %239 = vector.load %view_25[%c0, %c0, %25, %40] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %240 = vector.load %view_25[%c0, %c0, %25, %42] : memref<1x1x16x132xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
          %241 = amdgpu.mfma %233 * %29 + %cst_20 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %242 = amdgpu.mfma %234 * %31 + %241 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %243 = amdgpu.mfma %235 * %33 + %242 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %244 = amdgpu.mfma %236 * %35 + %243 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %245 = amdgpu.mfma %237 * %37 + %244 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %246 = amdgpu.mfma %238 * %39 + %245 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %247 = amdgpu.mfma %239 * %41 + %246 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %248 = amdgpu.mfma %240 * %43 + %247 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %249 = arith.mulf %248, %cst_24 : vector<4xf32>
          %250 = arith.mulf %249, %cst : vector<4xf32>
          %251 = math.absf %250 : vector<4xf32>
          %252 = arith.mulf %251, %cst_16 : vector<4xf32>
          %253 = math.exp2 %252 : vector<4xf32>
          %254 = arith.addf %253, %cst_17 : vector<4xf32>
          %255 = arith.divf %cst_17, %254 : vector<4xf32>
          %256 = arith.mulf %255, %cst_15 : vector<4xf32>
          %257 = arith.mulf %253, %256 : vector<4xf32>
          %258 = arith.addf %257, %255 : vector<4xf32>
          %259 = math.copysign %258, %250 : vector<4xf32>
          %260 = arith.mulf %259, %cst_23 : vector<4xf32>
          %261 = arith.cmpi slt, %68, %63 : vector<4xindex>
          %262 = arith.select %261, %cst_20, %cst_19 : vector<4xi1>, vector<4xf32>
          %263 = arith.addf %260, %262 : vector<4xf32>
          %264 = vector.extract %263[0] : f32 from vector<4xf32>
          %265 = vector.extract %263[1] : f32 from vector<4xf32>
          %266 = arith.maximumf %264, %265 : f32
          %267 = vector.extract %263[2] : f32 from vector<4xf32>
          %268 = arith.maximumf %266, %267 : f32
          %269 = vector.extract %263[3] : f32 from vector<4xf32>
          %270 = arith.maximumf %268, %269 : f32
          %271 = vector.broadcast %270 : f32 to vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %271, %c16_i32, %c64_i32 : vector<1xf32>
          %272 = arith.maximumf %271, %shuffleResult : vector<1xf32>
          %shuffleResult_26, %valid_27 = gpu.shuffle  xor %272, %c32_i32, %c64_i32 : vector<1xf32>
          %273 = arith.maximumf %272, %shuffleResult_26 : vector<1xf32>
          %274 = arith.maximumf %arg11, %273 : vector<1xf32>
          %275 = arith.subf %arg11, %274 : vector<1xf32>
          %276 = math.exp2 %275 : vector<1xf32>
          %277 = vector.extract %274[0] : f32 from vector<1xf32>
          %278 = vector.splat %277 : vector<4xf32>
          %279 = arith.subf %263, %278 : vector<4xf32>
          %280 = math.exp2 %279 : vector<4xf32>
          %281 = arith.mulf %arg12, %276 : vector<1xf32>
          %282 = vector.extract %280[0] : f32 from vector<4xf32>
          %283 = vector.extract %280[1] : f32 from vector<4xf32>
          %284 = arith.addf %282, %283 : f32
          %285 = vector.extract %280[2] : f32 from vector<4xf32>
          %286 = arith.addf %284, %285 : f32
          %287 = vector.extract %280[3] : f32 from vector<4xf32>
          %288 = arith.addf %286, %287 : f32
          %289 = vector.broadcast %288 : f32 to vector<1xf32>
          %shuffleResult_28, %valid_29 = gpu.shuffle  xor %289, %c16_i32, %c64_i32 : vector<1xf32>
          %290 = arith.addf %289, %shuffleResult_28 : vector<1xf32>
          %shuffleResult_30, %valid_31 = gpu.shuffle  xor %290, %c32_i32, %c64_i32 : vector<1xf32>
          %291 = arith.addf %290, %shuffleResult_30 : vector<1xf32>
          %292 = arith.addf %281, %291 : vector<1xf32>
          %293 = arith.truncf %280 : vector<4xf32> to vector<4xbf16>
          %294 = vector.extract %276[0] : f32 from vector<1xf32>
          %295 = vector.splat %294 : vector<4xf32>
          %296 = arith.mulf %arg13, %295 : vector<4xf32>
          %297 = arith.mulf %arg14, %295 : vector<4xf32>
          %298 = arith.mulf %arg15, %295 : vector<4xf32>
          %299 = arith.mulf %arg16, %295 : vector<4xf32>
          %300 = arith.mulf %arg17, %295 : vector<4xf32>
          %301 = arith.mulf %arg18, %295 : vector<4xf32>
          %302 = arith.mulf %arg19, %295 : vector<4xf32>
          %303 = arith.mulf %arg20, %295 : vector<4xf32>
          %304 = amdgpu.mfma %225 * %293 + %296 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %305 = amdgpu.mfma %226 * %293 + %297 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %306 = amdgpu.mfma %227 * %293 + %298 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %307 = amdgpu.mfma %228 * %293 + %299 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %308 = amdgpu.mfma %229 * %293 + %300 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %309 = amdgpu.mfma %230 * %293 + %301 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %310 = amdgpu.mfma %231 * %293 + %302 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          %311 = amdgpu.mfma %232 * %293 + %303 {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
          scf.yield %274, %292, %304, %305, %306, %307, %308, %309, %310, %311 : vector<1xf32>, vector<1xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
        }
        %65 = arith.cmpi sgt, %23, %c0 : index
        scf.if %65 {
          %66 = stream.binding.subspan %arg6[%c0] : !stream.binding -> memref<8x?x6xf32, strided<[?, 6, 1], offset: ?>>{%arg9}
          %67 = stream.binding.subspan %arg5[%c0] : !stream.binding -> memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>{%arg9}
          %68 = arith.divf %cst_14, %64#1 : vector<1xf32>
          %69 = vector.extract %68[0] : f32 from vector<1xf32>
          %70 = vector.splat %69 : vector<4xf32>
          %71 = arith.mulf %64#2, %70 : vector<4xf32>
          %72 = arith.mulf %64#3, %70 : vector<4xf32>
          %73 = arith.mulf %64#4, %70 : vector<4xf32>
          %74 = arith.mulf %64#5, %70 : vector<4xf32>
          %75 = arith.mulf %64#6, %70 : vector<4xf32>
          %76 = arith.mulf %64#7, %70 : vector<4xf32>
          %77 = arith.mulf %64#8, %70 : vector<4xf32>
          %78 = arith.mulf %64#9, %70 : vector<4xf32>
          %79 = math.log2 %64#1 : vector<1xf32>
          %80 = arith.addf %64#0, %79 : vector<1xf32>
          %81 = vector.splat %27 : vector<1xi1>
          vector.maskedstore %66[%block_id_z, %block_id_x, %25], %81, %80 : memref<8x?x6xf32, strided<[?, 6, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %26], %28, %71 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %30], %28, %72 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %32], %28, %73 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %34], %28, %74 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %36], %28, %75 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %38], %28, %76 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %40], %28, %77 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
          vector.maskedstore %67[%block_id_z, %block_id_x, %25, %42], %28, %78 : memref<8x?x6x128xf32, strided<[?, 768, 128, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
        }
        return
      }
    }
  }
  func.func @isolated_benchmark(%arg0: tensor<?x6x128xbf16>, %arg1: tensor<?x1x128xbf16>, %arg2: tensor<?x1x128xbf16>, %arg3: tensor<?xi32>, %arg4: tensor<?xi32>, %arg5: tensor<8x?x6x128xf32>, %arg6: tensor<8x?x6xf32>, %arg7: index, %arg8: index, %arg9: index) -> (tensor<8x?x6x128xf32>, tensor<8x?x6xf32>) {
    %0:2 = flow.dispatch @phase_0::@phase_0[%arg7, %arg8, %arg9](%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : (tensor<?x6x128xbf16>{%arg9}, tensor<?x1x128xbf16>{%arg8}, tensor<?x1x128xbf16>{%arg8}, tensor<?xi32>{%arg9}, tensor<?xi32>{%arg7}, tensor<8x?x6x128xf32>{%arg9}, tensor<8x?x6xf32>{%arg9}, index, index, index) -> (%arg5{%arg9}, %arg6{%arg9})
    return %0#0, %0#1 : tensor<8x?x6x128xf32>, tensor<8x?x6xf32>
  }
}
