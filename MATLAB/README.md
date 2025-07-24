# 📘 **MATLAB/Octave Learning Documentation**

### 📝 **1. Introduction**

MATLAB (Matrix Laboratory) and GNU Octave are high-level programming languages primarily intended for numerical computations. MATLAB is proprietary software developed by MathWorks, while Octave is its open-source counterpart. Both are widely used in engineering, data analysis, signal processing, and simulation.

---

### 📌 **2. Basic Syntax and Operations**

#### 2.1 Variables and Data Types

```matlab
a = 5;          % Integer
b = 3.14;       % Float
c = 'Hello';    % String
d = [1 2 3; 4 5 6];  % Matrix
```

#### 2.2 Operators

* Arithmetic: `+`, `-`, `*`, `/`, `.^` (element-wise power)
* Logical: `==`, `~=`, `>`, `<`, `&&`, `||`

#### 2.3 Comments

```matlab
% This is a single-line comment
```

---

### 📊 **3. Vectors and Matrices**

#### 3.1 Creating Vectors and Matrices

```matlab
v = [1, 2, 3];         % Row vector
v2 = [1; 2; 3];        % Column vector
M = [1 2; 3 4];        % 2x2 Matrix
```

#### 3.2 Matrix Operations

```matlab
transpose = M';            % Transpose
invM = inv(M);             % Inverse
detM = det(M);             % Determinant
dotProd = dot(v, v);       % Dot product
```

---

### 🔁 **4. Control Structures**

#### 4.1 Conditional Statements

```matlab
if x > 0
    disp('Positive');
elseif x < 0
    disp('Negative');
else
    disp('Zero');
end
```

#### 4.2 Loops

```matlab
for i = 1:5
    disp(i);
end

while x < 10
    x = x + 1;
end
```

---

### 🧮 **5. Functions**

#### 5.1 Defining Functions

```matlab
function y = square(x)
    y = x^2;
end
```

#### 5.2 Script vs Function Files

* **Script**: Runs commands in a file.
* **Function**: Accepts input, returns output, has its own scope.

---

### 📈 **6. Plotting and Visualization**

#### 6.1 Basic Plotting

```matlab
x = 0:0.1:10;
y = sin(x);
plot(x, y);
title('Sine Wave');
xlabel('x');
ylabel('sin(x)');
grid on;
```

#### 6.2 Multiple Plots

```matlab
plot(x, sin(x), 'r', x, cos(x), 'b');
legend('sin(x)', 'cos(x)');
```

---

### 📁 **7. File I/O**

#### 7.1 Reading and Writing Files

```matlab
data = load('data.txt');             % Load data from a text file
save('output.txt', 'data', '-ascii') % Save data to a text file
```

---

### 🛠️ **8. Useful Built-in Functions**

| Function    | Description                  |
| ----------- | ---------------------------- |
| `length(x)` | Number of elements in vector |
| `size(A)`   | Dimensions of matrix         |
| `sum(x)`    | Sum of vector elements       |
| `mean(x)`   | Mean value                   |
| `max(x)`    | Maximum value                |
| `sort(x)`   | Sort elements                |

---

### 📚 **9. Advanced Topics (Optional)**

* Simulink (MATLAB only)
* Signal Processing
* Image Processing
* Machine Learning Toolbox
* Symbolic Math

---

### ✅ **10. Practice and Exercises**

Try to implement:

1. Matrix multiplication program
2. Plot a sine and cosine curve
3. Write a function to calculate factorial
4. Load a `.csv` file and compute mean of a column

---

### 🧠 **11. Tips for Learning**

* Use the MATLAB/Octave documentation: `help <function>`
* Practice small projects (e.g., calculator, signal plotting, etc.)
* Participate in MATLAB/Octave communities or forums
* If you're using Octave, try using [Octave Online](https://octave-online.net/)

---