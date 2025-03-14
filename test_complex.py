#!/usr/bin/env python3

def fibonacci(n):
    if n <= 1:
        return n
    a = 0
    b = 1
    i = 2
    while i <= n:
        temp = a + b
        a = b
        b = temp
        i = i + 1
    return b

# Calculate first 10 Fibonacci numbers
n = 10
result = fibonacci(n)

# Calculate sum of numbers from 1 to 10
sum = 0
i = 1
while i <= 10:
    sum = sum + i
    i = i + 1

# Test some arithmetic
x = 5
y = 3
z = x * y + (x - y) 