#!/usr/bin/env python3

# Simple list creation and access
a = [1, 2, 3]
b = [4, 5, 6]

# Access elements
first_a = a[0]
second_b = b[1]

# Simple function with list
def sum_list(lst):
    total = 0
    for i in range(len(lst)):
        total = total + lst[i]
    return total

# Call function
sum_a = sum_list(a)
sum_b = sum_list(b) 