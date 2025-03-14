#!/usr/bin/env python3

# Try different import paths
try:
    from tinyfive.machine import machine
    print("Successfully imported from tinyfive.machine")
except ImportError:
    print("Failed to import from tinyfive.machine")

try:
    from tinyfive import machine
    print("Successfully imported from tinyfive")
except ImportError:
    print("Failed to import from tinyfive")

# If both fail, try to find the module
import sys
import pkgutil
print("\nSearching for tinyfive module:")
for loader, name, is_pkg in pkgutil.iter_modules():
    if 'tiny' in name.lower():
        print(f"Found module: {name}, is_package: {is_pkg}")

# Try to run a simple example if we found the machine
if 'machine' in locals():
    print("\nRunning simple example:")
    m = machine(mem_size=1000)
    
    # Print machine attributes
    print("Machine attributes:", dir(m))
    
    # Try to use registers
    print("Register x1:", m.x1)
    
    # Try to use memory
    print("Memory type:", type(m.mem))
    print("Memory size:", len(m.mem))
    
    # Try to run a simple program
    print("\nRunning simple program:")
    m.LI(m.x1, 5)  # Load immediate 5 into x1
    m.LI(m.x2, 7)  # Load immediate 7 into x2
    m.ADD(m.x3, m.x1, m.x2)  # x3 = x1 + x2
    
    print("Instructions:", m.instr)
    
    # Execute the program
    m.exe()
    
    # Check the result
    print("x3 should be 12:", m.x3) 