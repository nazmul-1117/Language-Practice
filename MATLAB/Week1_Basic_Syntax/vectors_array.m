clc, clearvars

# Array Creation
# Method 1
a = [1 2 -3 4 5];
#disp(a);

# Method 2
a = 1:10;
#disp(a);

# Method 3
a = [1, -2, 3, 4, 5];
#disp(a);

# Transpose Array
b = a';
#disp(b);

# Using Linespace
c = linspace(1, 100);
#disp(c);

# 50-100 (Only 20 Data)
c = linspace(50, 100, 20);
#disp(c);


# Matrix Creation
# Method 1
x = [1 2; 3 4];
#disp(x);

# Method 2
x = [1, 2; 3, 4];
#disp(x);


# Matrix Operations
x = 1:10;

# Method 1
r = x*x';
printf("`X` Multiply by `X Transpose`: %d\n", r);

# Method 2
r = x.^2;
disp("Individually Power: "), disp(r);

# Special Matrix

# Method 1
# 3x3->1 Matrix
x = ones(3);
disp(x);

# Method 2
# 3x2->1 Matrix
x = ones(3, 2);
disp(x);

# Method 3
# 3x3->0 Matrix
x = zeros(3);
disp(x);

# Method 4
# 3x2->0 Matrix
x = zeros(3, 2);
disp(x);


# Method 5
# 3x3 Eye Matrix
x = eye(3);
disp(x);
disp("\n")

# Indexes
x = [1 2 3; 4 5 6; 7 8 9];
disp(x);

second_index_data = x(2, 2) #1 based index(row, column)

# Method 2
# First Row, 2 and 3 Columns
disp(x(1:2, 2:3));

# End Data
dx = 1:10;
printf("End Data: %d\n", dx(end));
















