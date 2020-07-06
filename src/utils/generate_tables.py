import os
import subprocess

size = [
    2048,
    4096,
    32768,
    131072,
    262144,
    1048576,
    4194304,
    8388608,
    16777216,
    33554432,
    134217728,
    67108864,
    1073741824,
]


for i in range(len(size)):
    f = open("temp.cfg", "w")

    f.write("-size (bytes) " + str(size[i]) + "\n")
    f.write('-Array Power Gating - "false"\n')
    f.write('-WL Power Gating - "false"\n')
    f.write('-CL Power Gating - "false"\n')
    f.write('-Bitline floating - "false"\n')
    f.write('-Interconnect Power Gating - "false"\n')
    f.write("-block size (bytes) 64 \n")
    f.write(
        "-associativity 1 \n-read-write port 1\n-UCA bank count 1\n-technology (u) 0.040\n"
    )
    f.write(
        '-Data array cell type - "itrs-hp" \n-Data array peripheral type - "itrs-hp" \n-Tag array cell type - "itrs-hp" \n'
    )
    f.write(
        '-output/input bus width 512 \n- cache type "cache" \n-operating temperature (K) 360 \n-tag size (b) "default" \n'
    )
    f.write('-access mode (normal, sequential, fast) - "normal"\n')
    f.write(
        "-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:100:0\n"
    )
    f.write(
        "-deviate (delay, dynamic power, leakage power, cycle time, area) 20:100000:100000:100000:100000\n"
    )
    f.write('-Optimize ED or ED^2 (ED, ED^2, NONE): "ED^2"\n')
    f.write('-Wire signaling (fullswing, lowswing, default) - "Global_30"\n')
    f.write('-Wire inside mat - "semi-global"\n')
    f.write('-Wire outside mat - "semi-global"\n')
    f.write('-Interconnect projection - "conservative"\n')
    f.close()
    os.system("./cacti -infile temp.cfg")


for i in range(len(size)):
    f = open("temp.cfg", "w")
    f.write("-DesignTarget: cache\n")
    f.write("-CacheAccessMode: Normal\n")
    f.write("-Associativity (for cache only): 1\n")
    f.write("-ProcessNode: 65\n")
    f.write("-Capacity (MB): 4\n")
    f.write("-WordWidth (bit): 64\n")
    f.write("-DeviceRoadmap: LOP\n")
    f.write("-LocalWireType: LocalAggressive\n")
    f.write("-LocalWireRepeaterType: RepeatedNone\n")
    f.write("-LocalWireUseLowSwing: No\n")
    f.write("-GlobalWireType: GlobalAggressive\n")
    f.write("-GlobalWireRepeaterType: RepeatedNone\n")
    f.write("-GlobalWireUseLowSwing: No\n")
    f.write("-Routing: H-tree\n")
    f.write("-InternalSensing: true\n")
    f.write("-MemoryCellInputFile: sample_SRAM.cell\n")
    f.write("-Temperature (K): 350\n")
    f.write("-OptimizationTarget: WriteEDP\n")
    f.write("-EnablePruning: Yes\n")
    f.write("-BufferDesignOptimization: latency\n")
    f.write("-StackedDieCount: 2\n")
    f.write("-PartitionGranularity: 0\n")
    f.write("-LocalTSVProjection: 0\n")
    f.write("-GlobalTSVProjection: 0\n")
    f.write("-TSVRedundancy: 1.0\n")
