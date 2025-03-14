#!/usr/bin/env python3

from tinyfive.machine import machine
import numpy as np

# Create a machine instance
m = machine(mem_size=4000)

# Test simple multiplication
print("=== Simple Multiplication Test ===")
m.x[11] = 6        # manually load '6' into register x[11]
m.x[12] = 7        # manually load '7' into register x[12]
m.MUL(10, 11, 12)  # x[10] := x[11] * x[12]
print(f"6 * 7 = {m.x[10]}")

# Check available attributes
print("\n=== Machine Attributes ===")
attrs = [attr for attr in dir(m) if not attr.startswith('__')]
print("Attributes:", attrs)

# Test memory operations
print("\n=== Memory Operations Test ===")
addr = 100
value = 42
m.write_i32(addr, value)
read_value = m.read_i32(addr)
print(f"Wrote {value} to address {addr}, read back {read_value}")

# Test vector operations - fix the parameter order
print("\n=== Vector Operations Test ===")
vec_addr = 200
vec = np.array([1, 2, 3, 4, 5], dtype=np.int32)
# The correct order is write_i32_vec(vec, start_addr)
m.write_i32_vec(vec, vec_addr)
read_vec = m.read_i32_vec(vec_addr, 5)
print(f"Wrote vector {vec}, read back {read_vec}")

# Test instruction execution
print("\n=== Instruction Execution Test ===")
# Reset registers
m.x[1] = 0
m.x[2] = 0
m.x[3] = 0

# Store instructions in memory
instr_addr = 1000
m.LI(1, 10)  # Load immediate 10 into x1
m.LI(2, 20)  # Load immediate 20 into x2
m.ADD(3, 1, 2)  # x3 = x1 + x2

# Check if we have an ops attribute that contains the instructions
if hasattr(m, 'ops'):
    print("Machine has 'ops' attribute with instructions:", m.ops)
    
    # Store instructions in memory
    for i, op in enumerate(m.ops):
        m.write_i32(instr_addr + i*4, op)
    
    # Execute from the instruction address
    m.exe(instr_addr)
else:
    print("Machine does not have 'ops' attribute, cannot execute instructions")

# Check result
print(f"10 + 20 = {m.x[3]}")

# Let's try to manually track instructions
print("\n=== Manual Instruction Tracking ===")
# Create a new machine
m2 = machine(mem_size=1000)

# Create a counter dictionary
instr_counter = {}

# Store instructions in memory
instr_addr = 1000
instr_bytes = []

# Add some instructions and track them
m2.LI(1, 5)
instr_counter['LI'] = instr_counter.get('LI', 0) + 1
if hasattr(m2, 'ops'):
    instr_bytes.append(m2.ops[-1])  # Store the last instruction

m2.LI(2, 10)
instr_counter['LI'] = instr_counter.get('LI', 0) + 1
if hasattr(m2, 'ops'):
    instr_bytes.append(m2.ops[-1])  # Store the last instruction

m2.ADD(3, 1, 2)
instr_counter['ADD'] = instr_counter.get('ADD', 0) + 1
if hasattr(m2, 'ops'):
    instr_bytes.append(m2.ops[-1])  # Store the last instruction

m2.MUL(4, 1, 2)
instr_counter['MUL'] = instr_counter.get('MUL', 0) + 1
if hasattr(m2, 'ops'):
    instr_bytes.append(m2.ops[-1])  # Store the last instruction

# Store instructions in memory if we have them
if instr_bytes:
    for i, instr in enumerate(instr_bytes):
        m2.write_i32(instr_addr + i*4, instr)
    
    # Execute from the instruction address
    m2.exe(instr_addr)
else:
    print("Could not get instruction bytes, cannot execute")

# Print results
print("Manual instruction counts:", instr_counter)
print("5 + 10 =", m2.x[3])
print("5 * 10 =", m2.x[4]) 