#!/usr/bin/env python3

import ast
import astor
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tinyfive.machine import machine

@dataclass
class Variable:
    """Represents a variable in memory"""
    name: str
    addr: int
    size: int
    type: str

@dataclass
class Function:
    """Represents a function definition"""
    name: str
    params: List[str]
    body: List[ast.stmt]
    return_var: Optional[Variable] = None

class RISCVCompiler:
    """Compiles Python code to RISC-V instructions"""
    
    def __init__(self, mem_size: int = 10000):
        self.mem_size = mem_size
        self.machine = machine(mem_size=mem_size)
        self.variables: Dict[str, Variable] = {}
        self.functions: Dict[str, Function] = {}
        self.next_addr = 4  # Start storing variables at address 4
        self.temp_counter = 0
        self.instr_counter = {}
        self.return_value = None
    
    def get_temp_reg(self) -> int:
        """Get next available temporary register"""
        temps = [5, 6, 7, 28, 29, 30, 31]  # t0-t2, t3-t6
        self.temp_counter = (self.temp_counter + 1) % len(temps)
        return temps[self.temp_counter]
    
    def allocate_variable(self, name: str, size: int = 4, type: str = 'int') -> Variable:
        """Allocate memory for a variable"""
        var = Variable(name, self.next_addr, size, type)
        self.variables[name] = var
        self.next_addr += size
        # Initialize memory location to 0
        self.machine.write_i32(var.addr, 0)
        return var
    
    def compile_binary_op(self, node: ast.BinOp) -> int:
        """Compile binary operation and return register with result"""
        # Compile left and right operands
        left_reg = self.compile_expr(node.left)
        right_reg = self.compile_expr(node.right)
        result_reg = self.get_temp_reg()
        
        # Generate appropriate instruction
        if isinstance(node.op, ast.Add):
            self.machine.ADD(result_reg, left_reg, right_reg)
            self.instr_counter['ADD'] = self.instr_counter.get('ADD', 0) + 1
        elif isinstance(node.op, ast.Sub):
            self.machine.SUB(result_reg, left_reg, right_reg)
            self.instr_counter['SUB'] = self.instr_counter.get('SUB', 0) + 1
        elif isinstance(node.op, ast.Mult):
            self.machine.MUL(result_reg, left_reg, right_reg)
            self.instr_counter['MUL'] = self.instr_counter.get('MUL', 0) + 1
        elif isinstance(node.op, ast.Div):
            self.machine.DIV(result_reg, left_reg, right_reg)
            self.instr_counter['DIV'] = self.instr_counter.get('DIV', 0) + 1
        
        # Store result in memory for debugging
        self.machine.write_i32(0, self.machine.x[result_reg])
        
        return result_reg
    
    def compile_compare(self, node: ast.Compare) -> int:
        """Compile comparison operation and return register with result"""
        left_reg = self.compile_expr(node.left)
        right_reg = self.compile_expr(node.comparators[0])
        result_reg = self.get_temp_reg()
        
        if isinstance(node.ops[0], ast.Lt):
            self.machine.SLT(result_reg, left_reg, right_reg)
            self.instr_counter['SLT'] = self.instr_counter.get('SLT', 0) + 1
        elif isinstance(node.ops[0], ast.Gt):
            self.machine.SLT(result_reg, right_reg, left_reg)
            self.instr_counter['SLT'] = self.instr_counter.get('SLT', 0) + 1
        elif isinstance(node.ops[0], ast.LtE):
            self.machine.SLT(result_reg, right_reg, left_reg)
            self.machine.XORI(result_reg, result_reg, 1)
            self.instr_counter['SLT'] = self.instr_counter.get('SLT', 0) + 1
            self.instr_counter['XORI'] = self.instr_counter.get('XORI', 0) + 1
        elif isinstance(node.ops[0], ast.GtE):
            self.machine.SLT(result_reg, left_reg, right_reg)
            self.machine.XORI(result_reg, result_reg, 1)
            self.instr_counter['SLT'] = self.instr_counter.get('SLT', 0) + 1
            self.instr_counter['XORI'] = self.instr_counter.get('XORI', 0) + 1
        elif isinstance(node.ops[0], ast.Eq):
            # Implement equality using XOR and SLTU
            self.machine.XOR(result_reg, left_reg, right_reg)
            self.machine.SLTU(result_reg, 0, result_reg)
            self.machine.XORI(result_reg, result_reg, 1)
            self.instr_counter['XOR'] = self.instr_counter.get('XOR', 0) + 1
            self.instr_counter['SLTU'] = self.instr_counter.get('SLTU', 0) + 1
            self.instr_counter['XORI'] = self.instr_counter.get('XORI', 0) + 1
        
        return result_reg
    
    def compile_expr(self, node: ast.expr) -> int:
        """Compile expression and return register containing result"""
        if isinstance(node, ast.Num) or isinstance(node, ast.Constant):
            reg = self.get_temp_reg()
            if isinstance(node, ast.Num):
                value = node.n
            else:
                value = node.value
                if isinstance(value, str):
                    # Skip string constants in docstrings
                    return reg
            self.machine.x[reg] = value  # Directly set register value
            return reg
        elif isinstance(node, ast.Name):
            if node.id in self.variables:
                reg = self.get_temp_reg()
                var = self.variables[node.id]
                if isinstance(node.ctx, ast.Store):
                    # For store operations, return the address
                    self.machine.x[reg] = var.addr
                else:
                    # For load operations, return the value
                    value = self.machine.read_i32(var.addr)
                    if var.size > 4:  # List variable
                        self.machine.x[reg] = value  # Return base address
                    else:
                        self.machine.x[reg] = value  # Return value
                return reg
            else:
                raise NameError(f"Variable {node.id} not found")
        elif isinstance(node, ast.List):
            # Allocate memory for the list
            list_addr = self.next_addr
            list_size = len(node.elts)
            self.next_addr += list_size * 4  # Each element is 4 bytes
            
            # Store list elements
            for i, elt in enumerate(node.elts):
                value_reg = self.compile_expr(elt)
                self.machine.write_i32(list_addr + i * 4, self.machine.x[value_reg])
            
            # Return address of list
            reg = self.get_temp_reg()
            self.machine.x[reg] = list_addr
            return reg
        elif isinstance(node, ast.ListComp):
            # Get range bounds from the iterator
            if isinstance(node.generators[0].iter, ast.Call) and \
               isinstance(node.generators[0].iter.func, ast.Name) and \
               node.generators[0].iter.func.id == 'range':
                args = node.generators[0].iter.args
                if len(args) == 1:
                    start_reg = self.get_temp_reg()
                    self.machine.x[start_reg] = 0
                    end_reg = self.compile_expr(args[0])
                elif len(args) == 2:
                    start_reg = self.compile_expr(args[0])
                    end_reg = self.compile_expr(args[1])
                else:
                    raise NotImplementedError("Only range(end) or range(start, end) supported")
                
                # Allocate memory for the list
                list_size = self.machine.x[end_reg] - self.machine.x[start_reg]
                list_addr = self.next_addr
                self.next_addr += list_size * 4  # Each element is 4 bytes
                
                # Store iterator variable
                iter_var = node.generators[0].target.id
                if iter_var not in self.variables:
                    self.allocate_variable(iter_var)
                
                # Generate list elements
                i = self.machine.x[start_reg]
                while i < self.machine.x[end_reg]:
                    # Set iterator variable
                    self.machine.write_i32(self.variables[iter_var].addr, i)
                    
                    # Evaluate element expression
                    value_reg = self.compile_expr(node.elt)
                    self.machine.write_i32(list_addr + (i - self.machine.x[start_reg]) * 4, 
                                         self.machine.x[value_reg])
                    i += 1
                
                # Return address of list
                reg = self.get_temp_reg()
                self.machine.x[reg] = list_addr
                return reg
            else:
                raise NotImplementedError("Only range-based list comprehensions supported")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'range':
                # Handle range call
                args = node.args
                if len(args) == 1:
                    start_reg = self.get_temp_reg()
                    self.machine.x[start_reg] = 0
                    end_reg = self.compile_expr(args[0])
                elif len(args) == 2:
                    start_reg = self.compile_expr(args[0])
                    end_reg = self.compile_expr(args[1])
                else:
                    raise NotImplementedError("Only range(end) or range(start, end) supported")
                
                # Allocate memory for the range list
                list_size = self.machine.x[end_reg] - self.machine.x[start_reg]
                list_addr = self.next_addr
                self.next_addr += list_size * 4  # Each element is 4 bytes
                
                # Generate range elements
                i = self.machine.x[start_reg]
                while i < self.machine.x[end_reg]:
                    self.machine.write_i32(list_addr + (i - self.machine.x[start_reg]) * 4, i)
                    i += 1
                
                # Return address of list
                reg = self.get_temp_reg()
                self.machine.x[reg] = list_addr
                return reg
            else:
                return self.compile_call(node)
        elif isinstance(node, ast.Subscript):
            # Get base address of list
            base_reg = self.compile_expr(node.value)
            base_addr = self.machine.x[base_reg]
            
            # Get index
            if isinstance(node.slice, ast.Index):
                index_reg = self.compile_expr(node.slice.value)
            else:
                index_reg = self.compile_expr(node.slice)
            
            # Calculate element address
            addr_reg = self.get_temp_reg()
            self.machine.MUL(addr_reg, index_reg, 4)  # Multiply by 4 (element size)
            self.machine.ADD(addr_reg, addr_reg, base_reg)  # Add base address
            
            if isinstance(node.ctx, ast.Store):
                # For store operations, return the address
                return addr_reg
            else:
                # For load operations, return the value
                result_reg = self.get_temp_reg()
                value = self.machine.read_i32(self.machine.x[addr_reg])
                self.machine.x[result_reg] = value
                return result_reg
        elif isinstance(node, ast.BinOp):
            return self.compile_binary_op(node)
        elif isinstance(node, ast.Compare):
            return self.compile_compare(node)
        elif isinstance(node, ast.Call):
            return self.compile_call(node)
        else:
            raise NotImplementedError(f"Expression type {type(node)} not supported")
    
    def compile_assign(self, node: ast.Assign):
        """Compile assignment statement"""
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")
        
        target = node.targets[0]
        
        # Compile the value expression
        value_reg = self.compile_expr(node.value)
        
        if isinstance(target, ast.Name):
            # Allocate variable if it doesn't exist
            if target.id not in self.variables:
                # For list values, we need to store the base address
                if isinstance(node.value, ast.List):
                    var = self.allocate_variable(target.id, size=len(node.value.elts) * 4)
                    self.machine.write_i32(var.addr, self.machine.x[value_reg])
                else:
                    var = self.allocate_variable(target.id)
                    value = self.machine.x[value_reg]  # Get value from register
                    self.machine.write_i32(var.addr, value)
            else:
                var = self.variables[target.id]
                if isinstance(node.value, ast.List):
                    self.machine.write_i32(var.addr, self.machine.x[value_reg])
                else:
                    value = self.machine.x[value_reg]  # Get value from register
                    self.machine.write_i32(var.addr, value)
        elif isinstance(target, ast.Subscript):
            # Get target address
            target_reg = self.compile_expr(target)
            
            # Store value at target address
            self.machine.write_i32(self.machine.x[target_reg], self.machine.x[value_reg])
        else:
            raise NotImplementedError("Only simple variable assignments supported")
        
        # Store value in register for chaining
        result_reg = self.get_temp_reg()
        self.machine.x[result_reg] = self.machine.x[value_reg]
        return result_reg
    
    def compile_if(self, node: ast.If):
        """Compile if statement"""
        # Compile condition
        cond_reg = self.compile_expr(node.test)
        
        # Store current state
        state_before = self.machine.x.copy()
        mem_before = {}
        for name, var in self.variables.items():
            mem_before[name] = self.machine.read_i32(var.addr)
        
        # Compile and execute 'if' body
        for stmt in node.body:
            self.compile_statement(stmt)
        
        # Store 'if' body results
        state_if = self.machine.x.copy()
        mem_if = {}
        for name, var in self.variables.items():
            mem_if[name] = self.machine.read_i32(var.addr)
        
        # Restore state before 'if'
        self.machine.x = state_before.copy()
        for name, value in mem_before.items():
            self.machine.write_i32(self.variables[name].addr, value)
        
        # Compile and execute 'else' body
        if node.orelse:
            for stmt in node.orelse:
                self.compile_statement(stmt)
        
        # Store 'else' body results
        state_else = self.machine.x.copy()
        mem_else = {}
        for name, var in self.variables.items():
            mem_else[name] = self.machine.read_i32(var.addr)
        
        # Select results based on condition
        if self.machine.x[cond_reg] != 0:
            self.machine.x = state_if
            for name, value in mem_if.items():
                self.machine.write_i32(self.variables[name].addr, value)
        else:
            self.machine.x = state_else
            for name, value in mem_else.items():
                self.machine.write_i32(self.variables[name].addr, value)
    
    def compile_while(self, node: ast.While):
        """Compile while loop"""
        while True:
            # Compile condition
            cond_reg = self.compile_expr(node.test)
            
            # Exit if condition is false
            if self.machine.x[cond_reg] == 0:
                break
            
            # Compile body
            for stmt in node.body:
                self.compile_statement(stmt)
    
    def compile_return(self, node: ast.Return):
        """Compile return statement"""
        if node.value:
            value_reg = self.compile_expr(node.value)
            value = self.machine.x[value_reg]  # Get value from register
            self.return_value = value
            # Store return value in memory at a fixed address
            self.machine.write_i32(0, value)
    
    def compile_statement(self, node: ast.stmt):
        """Compile a statement"""
        if isinstance(node, ast.Assign):
            self.compile_assign(node)
        elif isinstance(node, ast.If):
            self.compile_if(node)
        elif isinstance(node, ast.While):
            self.compile_while(node)
        elif isinstance(node, ast.Return):
            self.compile_return(node)
        elif isinstance(node, ast.Expr):
            # Just compile the expression and ignore the result
            self.compile_expr(node.value)
        elif isinstance(node, ast.For):
            # Get range bounds from the iterator
            if isinstance(node.iter, ast.Call) and \
               isinstance(node.iter.func, ast.Name) and \
               node.iter.func.id == 'range':
                args = node.iter.args
                if len(args) == 1:
                    start_reg = self.get_temp_reg()
                    self.machine.x[start_reg] = 0
                    end_reg = self.compile_expr(args[0])
                elif len(args) == 2:
                    start_reg = self.compile_expr(args[0])
                    end_reg = self.compile_expr(args[1])
                else:
                    raise NotImplementedError("Only range(end) or range(start, end) supported")
                
                # Store iterator variable
                iter_var = node.target.id
                if iter_var not in self.variables:
                    self.allocate_variable(iter_var)
                
                # Generate loop
                i = self.machine.x[start_reg]
                while i < self.machine.x[end_reg]:
                    # Set iterator variable
                    self.machine.write_i32(self.variables[iter_var].addr, i)
                    
                    # Execute loop body
                    for stmt in node.body:
                        self.compile_statement(stmt)
                    
                    i += 1
            else:
                raise NotImplementedError("Only range-based for loops supported")
        else:
            raise NotImplementedError(f"Statement type {type(node)} not supported")
    
    def compile_function(self, node: ast.FunctionDef):
        """Compile a function definition"""
        # Create function object
        func = Function(
            name=node.name,
            params=[arg.arg for arg in node.args.args],
            body=node.body
        )
        self.functions[node.name] = func
    
    def compile_call(self, node: ast.Call) -> int:
        """Compile function call"""
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple function calls supported")
        
        func_name = node.func.id
        
        # Handle built-in functions
        if func_name == 'len':
            if len(node.args) != 1:
                raise ValueError("len() takes exactly one argument")
            
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id in self.variables:
                var = self.variables[arg.id]
                if var.size > 4:  # List variable
                    # For simplicity, we'll assume the size is a multiple of 4 (element size)
                    list_size = var.size // 4
                    result_reg = self.get_temp_reg()
                    self.machine.x[result_reg] = list_size
                    return result_reg
                else:
                    raise TypeError(f"object of type 'int' has no len()")
            elif isinstance(arg, ast.List):
                result_reg = self.get_temp_reg()
                self.machine.x[result_reg] = len(arg.elts)
                return result_reg
            else:
                # Try to evaluate the argument
                arg_reg = self.compile_expr(arg)
                # For now, we'll just return 0 as we can't determine the length
                result_reg = self.get_temp_reg()
                self.machine.x[result_reg] = 0
                return result_reg
        
        # Handle user-defined functions
        if func_name not in self.functions:
            raise NameError(f"Function {func_name} not found")
        
        func = self.functions[func_name]
        
        # Store current state
        state_before = self.machine.x.copy()
        mem_before = {}
        for name, var in self.variables.items():
            mem_before[name] = self.machine.read_i32(var.addr)
        
        # Create new scope for function parameters
        param_vars = {}
        param_regs = []  # Store parameter registers
        for param, arg in zip(func.params, node.args):
            value_reg = self.compile_expr(arg)
            var = self.allocate_variable(param)
            value = self.machine.x[value_reg]  # Get value from register
            self.machine.write_i32(var.addr, value)
            param_vars[param] = var
            param_regs.append(value_reg)  # Save parameter register
        
        # Reset return value
        self.return_value = None
        self.machine.write_i32(0, 0)  # Clear return value memory
        
        # Compile function body
        for stmt in func.body:
            self.compile_statement(stmt)
        
        # Get return value from memory
        return_value = self.machine.read_i32(0)
        
        # Restore state
        self.machine.x = state_before.copy()
        for name, value in mem_before.items():
            self.machine.write_i32(self.variables[name].addr, value)
        
        # Return value in a fresh register
        result_reg = self.get_temp_reg()
        self.machine.x[result_reg] = return_value
        
        # Remove function parameters from scope
        for var in param_vars.values():
            del self.variables[var.name]
        
        return result_reg
    
    def compile_module(self, node: ast.Module):
        """Compile a module (top-level code)"""
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.compile_function(stmt)
            else:
                self.compile_statement(stmt)

def compile_and_run(source_code: str, mem_size: int = 10000) -> Dict[str, Any]:
    """
    Compile and run Python source code on RISC-V
    
    Args:
        source_code: Python source code as string
        mem_size: Memory size for RISC-V machine
    
    Returns:
        Dictionary containing execution results and statistics
    """
    # Parse Python code
    tree = ast.parse(source_code)
    
    # Create compiler and compile code
    compiler = RISCVCompiler(mem_size=mem_size)
    compiler.compile_module(tree)
    
    # Get final variable values
    variables = {}
    for name, var in compiler.variables.items():
        variables[name] = compiler.machine.read_i32(var.addr)
    
    return {
        'variables': variables,
        'instruction_counts': compiler.instr_counter
    }

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python py2riscv.py <python_file>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        source_code = f.read()
    
    results = compile_and_run(source_code)
    print("\nFinal variable values:")
    for name, value in results['variables'].items():
        print(f"{name} = {value}")
    
    print("\nInstruction counts:")
    for instr, count in results['instruction_counts'].items():
        print(f"{instr}: {count}")
