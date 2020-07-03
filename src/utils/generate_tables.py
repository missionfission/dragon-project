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
