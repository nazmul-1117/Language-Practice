#!/bin/bash

# Floating-point arithmetic
echo "scale = 5; 123.456789/345.345345" | bc # Expected: .35748

echo 'scale=4;20+5/2' | bc # Expected: 22.5000

echo "-------------------------------------" # Added for separation

# Bash Shell Script to convert Celsius to Fahrenheit
read -p "Enter degree celsius temperature: " celsius

# Recommended modern syntax for command substitution: $(...)
fahrenheit=$(echo "scale=4; $celsius*1.8 + 32" | bc)

# Older, but still valid, syntax for command substitution: `...`
# fahrenheit=`echo "scale=4; $celsius*1.8 + 32" | bc`

echo "$celsius degree celsius is equal to $fahrenheit degree fahrenheit"

# Example: 37 degree celsius is equal to 98.6 degree fahrenheit