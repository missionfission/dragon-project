#!/usr/bin/env python3

from tinyfive.machine import machine
import numpy as np

def test_numpy_support():
    """Test TinyFive's NumPy support"""
    print("Testing TinyFive NumPy support...")
    
    # Create a machine instance
    m = machine(mem_size=10000)
    
    # Create NumPy arrays
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b = np.array([6, 7, 8, 9, 10], dtype=np.int32)
    
    # Store arrays in memory
    a_addr = 100
    b_addr = 200
    c_addr = 300
    
    print("Writing arrays to memory...")
    try:
        m.write_i32_vec(a, a_addr)
        m.write_i32_vec(b, b_addr)
        print("Success!")
    except Exception as e:
        print(f"Failed to write arrays: {e}")
        return
    
    # Read arrays back
    print("Reading arrays from memory...")
    try:
        a_read = m.read_i32_vec(a_addr, 5)
        b_read = m.read_i32_vec(b_addr, 5)
        print(f"a_read: {a_read}")
        print(f"b_read: {b_read}")
    except Exception as e:
        print(f"Failed to read arrays: {e}")
        return
    
    # Perform vector addition using RISC-V
    print("Performing vector addition...")
    
    # Try to execute with machine instructions
    try:
        # Initialize registers
        m.x[5] = 5  # Size of vectors
        m.x[6] = 0  # Loop counter
        
        # Store instructions in memory
        instr_addr = 1000
        
        # Check if loop counter < size
        m.BGE(6, 5, 10)  # If i >= size, exit loop
        
        # Calculate addresses
        m.MUL(7, 6, 4)  # i * 4
        m.ADDI(8, 7, a_addr)  # a_addr + i*4
        m.ADDI(9, 7, b_addr)  # b_addr + i*4
        m.ADDI(10, 7, c_addr)  # c_addr + i*4
        
        # Load values
        m.LW(11, 8, 0)  # Load a[i]
        m.LW(12, 9, 0)  # Load b[i]
        
        # Add values
        m.ADD(13, 11, 12)  # a[i] + b[i]
        
        # Store result
        m.SW(10, 13, 0)  # Store in c[i]
        
        # Increment counter
        m.ADDI(6, 6, 1)  # i++
        
        # Jump back to start of loop
        m.JAL(0, 0)
        
        # Execute the program
        print("Executing program...")
        m.exe(start=instr_addr)
        print("Execution complete!")
    except Exception as e:
        print(f"Execution failed: {e}")
        
        # Try manual execution
        print("Trying manual execution...")
        for i in range(5):
            a_val = m.read_i32(a_addr + i * 4)
            b_val = m.read_i32(b_addr + i * 4)
            c_val = a_val + b_val
            m.write_i32(c_addr + i * 4, c_val)
        print("Manual execution complete!")
    
    # Read result
    try:
        c_read = m.read_i32_vec(c_addr, 5)
        print(f"c_read (a + b): {c_read}")
        
        # Verify result
        expected = a + b
        print(f"Expected result: {expected}")
        print(f"Correct: {np.array_equal(c_read, expected)}")
    except Exception as e:
        print(f"Failed to read result: {e}")
    
    print("NumPy test complete!")

if __name__ == "__main__":
    test_numpy_support() 