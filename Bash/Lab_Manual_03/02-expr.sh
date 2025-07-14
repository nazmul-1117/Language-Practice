#!/bin/bash
# Basic Arithmetic Operations using expr

# Type 1
expr 1 + 1 #2
expr 2 - 1 #1
expr 2 \* 3 #6
expr 2 / 2 #1
expr 11 % 2 #1

# Type 2
expr 5+6 #5+6
expr "5+6" #5+6
expr "5 + 6" #5 + 6

# Type 3
expr $1+$2 # 10+20

# Type 4
let "a = $(expr $1 + $2)"
echo $a #30

expr "a = $(expr $1 + $2)"
echo $a # a = 30

let "a = $(expr $1 + $2)"
echo $a # 30