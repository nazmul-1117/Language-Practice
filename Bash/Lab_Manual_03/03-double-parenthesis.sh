#!/bin/bash

# Double Parenthesis

# Type 1 - Correct (Arithmetic Expansion)
echo "Type 1: "
echo $((2+3))

# Type 2 - Corrected (Command Substitution)
echo "Type 2 (Corrected): "
echo "$((10011%11))"

# Type 3 (demonstrating arithmetic expression within variables)
echo "Type 3 (Arithmetic with variables):"
num1=15
num2=7
echo $((num1 - num2))

# Type 4 (using (( )) for assignment without output)
echo "Type 4 (Using (( )) for assignment):"
((result = num1 * num2))
echo "Result of multiplication: $result"

# Type 5
echo "Type 5:"
a=10
echo $(($a + 5))
echo $((a + 5))

# Type 6
# echo "Type 6:"