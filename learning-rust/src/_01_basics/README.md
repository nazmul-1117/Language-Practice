# 🦀 Rust Basics

Welcome to the **Rust Basics** section of my Rust learning journey.

This directory contains my first steps in learning the Rust programming language, including fundamental syntax, program structure, variables, data types, functions, and control flow.

The goal of this section is to build a strong foundation before moving into advanced Rust concepts like **ownership, borrowing, lifetimes, and memory safety**.

---

## 📚 Topics Covered

### 1. Hello World

File:

```text
main.rs
```

Learning:

- Rust program structure
- `main()` function
- Printing output
- Compilation and execution using Cargo

Example:

```rust
fn main() {
    println!("Hello, Rust!");
}
```

---

## 2. Variables

File:

```text
variables.rs
```

Topics:

- Variable declaration
- Immutable variables
- Mutable variables
- Constants
- Shadowing

Concept:

Rust variables are immutable by default.

Example:

```rust
let x = 10;

let mut y = 20;
y = 30;
```

---

## 3. Data Types

File:

```text
data_types.rs
```

Topics:

### Scalar Types

- Integer
- Floating point
- Boolean
- Character

Example:

```rust
let age: i32 = 25;
let height: f64 = 5.9;
let active: bool = true;
let grade: char = 'A';
```

## Compound Types

- Tuple
- Array

Example:

```rust
let person = ("John", 25);

let numbers = [1,2,3,4];
```

---

## 4. Functions

File:

```text
functions.rs
```

Topics:

- Function declaration
- Parameters
- Return values
- Expressions

Example:

```rust
fn add(a:i32, b:i32) -> i32 {
    a + b
}
```

---

## 5. Statements and Expressions

File:

```text
statements_expressions.rs
```

Topics:

- Difference between statements and expressions
- Returning values without `return`

Example:

```rust
let result = {
    let x = 10;
    x + 5
};
```

---

## 6. Control Flow

File:

```text
control_flow.rs
```

Topics:

- if
- else
- else if
- match

Example:

```rust
if age >= 18 {
    println!("Adult");
} else {
    println!("Minor");
}
```

---

## 7. Loops

File:

```text
loops.rs
```

Topics:

- loop
- while
- for

Example:

```rust
for number in 1..5 {
    println!("{}", number);
}
```

---

## 🛠️ Running Examples

From the project root:

```bash
cargo run
```

---

## 📁 Directory Structure

```text
01_basics/

├── README.md
├── mod.rs
├── variables.rs
├── data_types.rs
├── functions.rs
├── control_flow.rs
└── loops.rs
```

---

## 🎯 Learning Goals

After completing this section, I should understand:

✅ Rust syntax basics  
✅ How Rust programs execute  
✅ Variables and data types  
✅ Writing functions  
✅ Using conditions and loops  
✅ Basic Rust coding style  

---

## 🧠 Notes

Important Rust mindset:

> Rust looks similar to C/C++, but its compiler checks safety rules during compilation.

Before moving forward, I should be comfortable with:

- Variables
- Functions
- Data types
- Control flow

Next topic:

➡️ **02 Ownership & Borrowing**

---

🦀 Learning Rust step by step.
