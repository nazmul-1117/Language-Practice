# 🦀 Learning Rust

Welcome to my **Rust Learning Repository**.

This repository contains my journey of learning **Rust programming language** and adding Rust to my programming skill stack.

I have experience with **C/C++, Java, and Python**, and I am exploring Rust to understand modern systems programming, memory safety, performance, and reliable software development.

---

## 🚀 Why Rust?

Rust is a modern systems programming language focused on:

- ⚡ High performance
- 🛡️ Memory safety without garbage collection
- 🔒 Safe concurrency
- 🧠 Zero-cost abstractions
- 🛠️ Reliable and maintainable software

Rust combines the power and control of C/C++ with strong compiler-enforced safety.

---

## 🧑‍💻 My Programming Journey

Current Language Stack:

```text
C
│
├── C++
│
├── Java
│
├── MySQL/PostgreSQL
│
├── Assembly Language
│
├── Shell Scripting
│
├── MATLAB/Octave
│
├── Python
│
└── Rust 🦀 (Learning)

```

Rust is my next step toward understanding:

- Systems programming
- Low-level memory management
- Performance-oriented software
- Modern programming paradigms

---

## 📚 Learning Goals

### Rust Fundamentals

- [ ] Installing Rust toolchain
- [ ] Cargo package manager
- [ ] Basic syntax
- [ ] Variables and data types
- [ ] Functions
- [ ] Control flow
- [ ] Loops
- [ ] Pattern matching

---

### Core Rust Concepts

- [ ] Ownership system
- [ ] Borrowing and references
- [ ] Lifetimes
- [ ] Stack and heap memory
- [ ] Structs
- [ ] Enums
- [ ] Pattern matching
- [ ] Error handling

---

### Intermediate Rust

- [ ] Modules and packages
- [ ] Traits
- [ ] Generics
- [ ] Collections
- [ ] Iterators
- [ ] Closures
- [ ] Smart pointers
- [ ] Testing

---

### Advanced Topics

- [ ] Multithreading
- [ ] Async programming
- [ ] Networking
- [ ] File handling
- [ ] Web services
- [ ] System programming

---

## 🛠️ Development Environment

### Tools

- Rust
- Cargo
- VS Code
- rust-analyzer
- Git & GitHub

### Rust Version

```bash
rustc --version
````

Example:

```bash
rustc 1.90.0
```

---

## 📂 Repository Structure

```text
learning-rust/
│
├── Cargo.toml
├── Cargo.lock
├── README.md
├── ROADMAP.md
├── .gitignore
│
├── src/
│   │
│   ├── main.rs
│   ├── lib.rs                 # Shared modules (optional but recommended)
│   │
│   ├── 01_basics/
│   │   ├── mod.rs
│   │   ├── variables.rs
│   │   ├── data_types.rs
│   │   ├── functions.rs
│   │   ├── statements_expressions.rs
│   │   ├── control_flow.rs
│   │   ├── loops.rs
│   │   └── pattern_matching.rs
│   │
│   ├── 02_ownership/
│   │   ├── mod.rs
│   │   ├── ownership.rs
│   │   ├── move_semantics.rs
│   │   ├── copy_clone.rs
│   │   ├── borrowing.rs
│   │   ├── references.rs
│   │   ├── mutable_references.rs
│   │   ├── stack_heap.rs
│   │   └── lifetimes.rs
│   │
│   ├── 03_structs_enums/
│   │   ├── mod.rs
│   │   ├── structs.rs
│   │   ├── tuple_structs.rs
│   │   ├── methods.rs
│   │   ├── associated_functions.rs
│   │   ├── enums.rs
│   │   ├── option.rs
│   │   ├── result.rs
│   │   └── pattern_matching.rs
│   │
│   ├── 04_collections/
│   │   ├── mod.rs
│   │   ├── vectors.rs
│   │   ├── strings.rs
│   │   ├── string_slice.rs
│   │   ├── hashmap.rs
│   │   ├── hashset.rs
│   │   └── iterators.rs
│   │
│   ├── 05_error_handling/
│   │   ├── mod.rs
│   │   ├── panic.rs
│   │   ├── option_handling.rs
│   │   ├── result_handling.rs
│   │   ├── question_mark.rs
│   │   └── custom_errors.rs
│   │
│   ├── 06_modules_packages/
│   │   ├── mod.rs
│   │   ├── modules.rs
│   │   ├── visibility.rs
│   │   ├── crates.rs
│   │   └── cargo.rs
│   │
│   ├── 07_traits_generics/
│   │   ├── mod.rs
│   │   ├── traits.rs
│   │   ├── trait_implementation.rs
│   │   ├── generics.rs
│   │   ├── trait_bounds.rs
│   │   └── dynamic_dispatch.rs
│   │
│   ├── 08_functional_programming/
│   │   ├── mod.rs
│   │   ├── closures.rs
│   │   ├── iterators.rs
│   │   ├── map_filter.rs
│   │   └── collect.rs
│   │
│   ├── 09_testing/
│   │   ├── mod.rs
│   │   ├── unit_tests.rs
│   │   ├── integration_tests.rs
│   │   └── documentation_tests.rs
│   │
│   ├── 10_smart_pointers/
│   │   ├── mod.rs
│   │   ├── box.rs
│   │   ├── rc.rs
│   │   ├── arc.rs
│   │   └── refcell.rs
│   │
│   ├── 11_concurrency/
│   │   ├── mod.rs
│   │   ├── threads.rs
│   │   ├── channels.rs
│   │   ├── mutex.rs
│   │   └── shared_state.rs
│   │
│   ├── 12_async_programming/
│   │   ├── mod.rs
│   │   ├── async_basics.rs
│   │   ├── futures.rs
│   │   └── tokio.rs
│   │
│   ├── 13_file_handling/
│   │   ├── mod.rs
│   │   ├── read_file.rs
│   │   ├── write_file.rs
│   │   ├── serialization.rs
│   │   └── json.rs
│   │
│   ├── 14_networking/
│   │   ├── mod.rs
│   │   ├── tcp.rs
│   │   ├── udp.rs
│   │   ├── http.rs
│   │   └── rest_api.rs
│   │
│   ├── 15_system_programming/
│   │   ├── mod.rs
│   │   ├── unsafe_rust.rs
│   │   ├── raw_pointers.rs
│   │   ├── ffi.rs
│   │   └── memory_layout.rs
│   │
│   └── projects/
│       │
│       ├── beginner/
│       │   ├── calculator.rs
│       │   ├── guessing_game.rs
│       │   └── todo_cli.rs
│       │
│       ├── intermediate/
│       │   ├── file_search.rs
│       │   ├── text_processor.rs
│       │   └── json_parser.rs
│       │
│       └── advanced/
│           ├── web_server.rs
│           ├── database_engine.rs
│           └── game_engine.rs
│
│
├── tests/
│   ├── integration_test.rs
│   └── project_tests.rs
│
├── examples/
│   ├── example_server.rs
│   └── example_cli.rs
│
├── benches/
│   └── performance.rs
│
├── notes/
│   ├── 01_rust_basics.md
│   ├── 02_ownership.md
│   ├── 03_borrow_checker.md
│   ├── 04_memory_management.md
│   ├── rust_vs_cpp.md
│   ├── compiler_notes.md
│   └── mistakes_and_lessons.md
│
└── target/
    └── (generated by Cargo - ignored by Git)
```

---

## 🧪 Practice Projects

Small projects I will build while learning:

- [ ] Hello World
- [ ] Calculator
- [ ] Number guessing game
- [ ] CLI tools
- [ ] File reader
- [ ] Simple text processor
- [ ] Small networking projects

---

## 📖 Learning Resources

### Official Resources

- The Rust Programming Language Book
  [https://doc.rust-lang.org/book/](https://doc.rust-lang.org/book/)

- Rust Official Website
  [https://www.rust-lang.org/](https://www.rust-lang.org/)

---

## 🎯 Purpose of This Repository

This repository is not only about learning Rust syntax.

The main goal is to understand:

> How modern systems programming can achieve C/C++ level performance while providing stronger memory safety.

---

## 📈 Progress

Started: September 2026

Status:

```text
🦀 Rust Journey Started
```

---

## 💡 Notes

Learning Rust after C/C++ provides a different perspective on:

- Memory ownership
- Compiler-driven development
- Safe abstractions
- Modern systems design

This repository will document my progress, experiments, mistakes, and improvements while learning Rust.

---

⭐ Learning never stops. Every new language brings a new way of thinking.
