#!/bin/bash
# Basic Arithmetic Operations using let

# Type 1
let a=5+6
echo $a

# Type 2
let  "b = 5 + 6"
echo $b

# Type 3
let "c = 10 * 5"
echo $c

# Type 4
let "d = 10 / 5"
echo $d

# Type 5
let "e = $1 + $2"
echo $e