# 🦀 Rust Learning Roadmap (Beginner → Advanced)

Since you already know **C/C++, Java, and Python**, you can learn Rust faster because many concepts (types, memory, compilation, data structures) are familiar. Your main focus should be understanding **Rust's unique philosophy: ownership, borrowing, lifetimes, and safe systems programming**.

---

## Structure

```text
learning-rust/

├── README.md
├── ROADMAP.md
├── Cargo.toml
├── .gitignore

├── 01_basics
├── 02_ownership
├── 03_structs_enums
├── 04_collections
├── 05_error_handling
├── 06_traits_generics
├── 07_testing
├── 08_concurrency
├── 09_async
├── 10_networking
├── 11_system_programming

├── projects

└── notes
```

## Phase 0: Setup & Rust Ecosystem (1–2 Days)

### Learn

* Installing Rust
* Rust toolchain
* `rustc` compiler
* Cargo package manager
* Rust project structure
* Rust editions

### Practice

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Check installation:

```bash
rustc --version
cargo --version
```

Create project:

```bash
cargo new hello_rust
cd hello_rust
cargo run
```

Understand:

```text
Cargo.toml
src/
 └── main.rs
```

---

## Phase 1: Rust Fundamentals (1–2 Weeks)

## 1. Basic Syntax

Learn:

* Hello World
* Comments
* Variables
* Constants
* Shadowing
* Mutable variables

Example:

```rust
fn main() {
    let x = 10;
    let mut y = 20;

    y = 30;

    println!("{}", y);
}
```

---

## 2. Data Types

Learn:

### Scalar Types

* Integer
* Float
* Boolean
* Character

```rust
let age: i32 = 25;
let pi: f64 = 3.14;
let active: bool = true;
```

### Compound Types

* Tuple
* Array

```rust
let tuple = (10, "Rust");

let numbers = [1,2,3,4];
```

---

## 3. Functions

Learn:

* Parameters
* Return values
* Expressions

```rust
fn add(a:i32,b:i32)->i32{
    a+b
}
```

---

## 4. Control Flow

Learn:

* if/else
* match
* loops

```rust
match number {
    1 => println!("One"),
    _ => println!("Other"),
}
```

---

## Phase 2: Ownership & Memory Model (Most Important) (2–4 Weeks)

This is where Rust becomes different from C++.

## 5. Stack vs Heap

Understand:

```text
Stack
 |
 |-- integers
 |-- function calls


Heap
 |
 |-- dynamic data
 |-- String
 |-- Vec
```

---

## 6. Ownership System ⭐⭐⭐

Master:

* Ownership rules
* Move semantics
* Copy types
* Clone

Example:

```rust
let a = String::from("Rust");

let b = a;

// a is no longer valid
```

Learn why Rust prevents:

* Dangling pointers
* Double free
* Memory leaks

---

## 7. Borrowing & References

Learn:

* Immutable references
* Mutable references
* Borrow checker

Example:

```rust
fn length(s:&String){
    println!("{}",s.len());
}
```

Rules:

```text
Multiple immutable references OR
One mutable reference
```

---

## 8. Lifetimes

Learn:

* Lifetime annotations
* Relationship between references

Example:

```rust
fn longest<'a>(
x:&'a str,
y:&'a str
)-> &'a str
```

---

## Phase 3: Rust Data Structures (2 Weeks)

## 9. Structs

Learn:

* Creating structures
* Methods
* Associated functions

Example:

```rust
struct User{
    name:String,
    age:u32
}
```

---

## 10. Enums

Learn:

* Enum variants
* Option
* Result

Example:

```rust
enum Status{
    Active,
    Inactive
}
```

---

## 11. Pattern Matching

Master:

* match
* if let
* while let

---

## Phase 4: Collections & Error Handling (2 Weeks)

## 12. Collections

Learn:

### Vector

```rust
Vec<T>
```

Example:

```rust
let numbers = vec![1,2,3];
```

---

### String Handling

Learn:

* String
* &str
* UTF-8

---

### HashMap

```rust
HashMap<K,V>
```

---

## 13. Error Handling

Learn:

### Option

```rust
Some(value)
None
```

### Result

```rust
Ok(value)
Err(error)
```

Handle errors without:

```text
try/catch
```

---

## Phase 5: Intermediate Rust (3–5 Weeks)

## 14. Modules

Learn:

* Packages
* Crates
* Modules
* Visibility

Structure:

```text
src/

main.rs

lib.rs

modules/
```

---

## 15. Traits ⭐

Rust equivalent:

```text
Interfaces (Java)
Abstract classes (C++)
```

Example:

```rust
trait Animal{
    fn sound(&self);
}
```

---

## 16. Generics

Learn:

* Generic functions
* Generic structs
* Trait bounds

Example:

```rust
fn largest<T>(list:&[T])
```

---

## 17. Iterators

Learn:

* Iterator trait
* map()
* filter()
* collect()

Example:

```rust
numbers
.iter()
.map(|x|x*2)
.collect();
```

---

## 18. Closures

Rust lambda functions:

```rust
let add = |a,b| a+b;
```

---

## Phase 6: Advanced Rust (1–3 Months)

## 19. Smart Pointers

Learn:

```rust
Box<T>
Rc<T>
Arc<T>
RefCell<T>
```

Understand:

```text
Heap allocation
Reference counting
Interior mutability
```

---

## 20. Multithreading

Learn:

* Threads
* Channels
* Mutex
* Arc

Example:

```rust
thread::spawn(||{
 println!("Thread");
});
```

---

## 21. Async Programming

Learn:

* async/await
* Futures
* Tokio

Example:

```rust
async fn download(){

}
```

---

## 22. File Handling

Learn:

* Reading files
* Writing files
* Serialization

Libraries:

* serde
* serde_json

---

## 23. Networking

Learn:

* TCP
* UDP
* HTTP
* REST APIs

Libraries:

* Tokio
* Hyper
* Axum

---

## Phase 7: Systems Programming (Your C/C++ Background Advantage)

Learn:

## Operating System Concepts

* Processes
* Threads
* Memory mapping
* System calls

---

## Unsafe Rust

Learn:

```rust
unsafe {

}
```

Topics:

* Raw pointers
* FFI
* C interoperability

---

## Embedded Rust

Learn:

* Microcontrollers
* No-std Rust
* Hardware programming

---

## Phase 8: Real Projects

Build projects in increasing difficulty:

## Beginner

✅ Hello World
✅ Calculator
✅ Temperature Converter
✅ Guessing Game
✅ Todo CLI

---

## Intermediate

✅ File search tool (Rust version of grep)

✅ Markdown parser

✅ JSON parser

✅ Password manager

✅ Web scraper

---

## Advanced

✅ HTTP server

✅ Database engine

✅ Operating system kernel module

✅ Blockchain prototype

✅ Game engine components

✅ Embedded firmware

---

## Recommended Learning Order

```text
Rust Basics
     |
     ↓
Ownership ⭐
     |
     ↓
Borrowing ⭐
     |
     ↓
Structs & Enums
     |
     ↓
Collections
     |
     ↓
Error Handling
     |
     ↓
Traits & Generics
     |
     ↓
Async & Threads
     |
     ↓
Networking
     |
     ↓
Systems Programming
```

---

## Best Resources

### Official Book (Must Read)

📘 **The Rust Programming Language ("The Book")**

[https://doc.rust-lang.org/book/](https://doc.rust-lang.org/book/)

#### Practice

* Rustlings
* Exercism Rust Track
* LeetCode (Rust)

#### Advanced

* Rust by Example
* Tokio Documentation
* The Embedded Rust Book

---

## Suggested Timeline (Based on Your Background)

| Month     | Goal                                        |
| --------- | ------------------------------------------- |
| Month 1   | Rust basics + Ownership                     |
| Month 2   | Structs, Enums, Collections, Error Handling |
| Month 3   | Traits, Generics, Testing, Modules          |
| Month 4   | Async, Networking, Multithreading           |
| Month 5–6 | Systems Programming + Real Projects         |

---

For someone coming from **C/C++**, the biggest mindset shift is:

> **C/C++ says: "You control memory."**
> **Rust says: "You prove to the compiler that your memory usage is safe."** 🦀

Master **ownership and borrowing**, and the rest of Rust becomes much easier.
