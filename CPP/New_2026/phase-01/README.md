# 🎯 Phase 1 Project: Student Management System

We'll build a **console-based Student Management System** from scratch.

By the end, you'll have something roughly like:

```text
========================================
       STUDENT MANAGEMENT SYSTEM
========================================

1. Add Student
2. Show All Students
3. Search Student
4. Update Student
5. Delete Student
6. Show Statistics
7. Exit

Choose: _
```

But **we will NOT build the whole thing at once**.

We'll evolve it as we learn.

---

# 🗺️ Phase 1 Learning Plan

### Part 1 — Basic structure

Learn/review:

```text
variables
data types
functions
scope
```

Build:

```text
Student
```

---

### Part 2 — References

Learn:

```cpp
int& ref
```

Then use references to modify students.

Example:

```cpp
void updateStudent(Student& student)
{
    student.age++;
}
```

You'll understand **why** `Student&` is necessary.

---

### Part 3 — Pointers

Learn:

```cpp
Student* ptr
```

Then implement searching:

```cpp
Student* findStudent(...)
```

Now you'll understand why a function might return a pointer.

---

### Part 4 — `const`

We'll use:

```cpp
const Student&
```

and:

```cpp
const Student*
```

You'll learn the difference between:

```cpp
Student*
const Student*
Student* const
const Student* const
```

This is important.

---

### Part 5 — `nullptr`

We'll make search behave like:

```cpp
Student* student = findStudent(id);

if (student == nullptr)
{
    std::cout << "Student not found";
}
```

Now `nullptr` has a practical purpose instead of being something you memorize.

---

### Part 6 — Stack vs Heap

We'll create students in different ways:

```cpp
Student student;
```

and:

```cpp
Student* student = new Student;
```

Then we'll understand:

```text
Stack
 ↓
automatic lifetime

Heap
 ↓
dynamic lifetime
```

We'll also introduce:

```cpp
delete
```

**only for educational understanding**. We won't build the final project around raw `new/delete`, because later we'll learn RAII and smart pointers.

---

### Part 7 — Parameter Passing

We'll compare:

```cpp
void function(Student student);
```

vs

```cpp
void function(Student& student);
```

vs

```cpp
void function(const Student& student);
```

vs

```cpp
void function(Student* student);
```

This is one of the most important Phase 1 skills.

---

### Part 8 — Function Overloading

Our project might have:

```cpp
searchStudent(int id);
searchStudent(std::string name);
```

You'll understand function overloading naturally.

---

### Part 9 — Default Arguments

For example:

```cpp
void displayStudent(
    const Student& student,
    bool detailed = false
);
```

Then:

```cpp
displayStudent(student);
```

and:

```cpp
displayStudent(student, true);
```

---

### Part 10 — `static`

We'll use a static member for something meaningful, such as:

```cpp
Student::getStudentCount();
```

So you'll understand that `static` can represent **class-level/shared state**, rather than just memorizing the keyword.

---

### Part 11 — `constexpr`

We'll introduce constants such as:

```cpp
constexpr int MAX_STUDENTS = 100;
```

Then understand why `constexpr` exists.

---

### Part 12 — `auto`

We'll gradually use:

```cpp
auto student = ...
```

and understand when `auto` improves readability and when explicit types are better.

---

### Part 13 — Namespaces

We'll structure our project:

```cpp
namespace University
{
    class Student
    {
        ...
    };
}
```

Then:

```cpp
University::Student student;
```

---

### Part 14 — Type Casting

We'll use practical examples involving:

```cpp
static_cast
```

and learn why C-style casting such as:

```cpp
(int)value
```

is generally not what we want in modern C++.

---

# 🏗️ Final Project Structure

By the end of Phase 1, we'll have something like:

```text
StudentManagement/
│
├── main.cpp
│
├── Student.h
├── Student.cpp
│
├── StudentManager.h
├── StudentManager.cpp
│
└── README.md
```

But **we won't start with this structure**.

That's important.

We'll start with a single `.cpp` file so that you're learning C++ rather than fighting project architecture.

Then we'll refactor it.

---

# 🚀 Step 1 — Start the Project

Create:

```text
student_management.cpp
```

Put this in it:

```cpp
#include <iostream>
#include <string>

namespace University
{
    struct Student
    {
        int id;
        std::string name;
        int age;
        double cgpa;
    };
}

int main()
{
    University::Student student;

    student.id = 101;
    student.name = "Rahim";
    student.age = 22;
    student.cgpa = 3.49;

    std::cout << "Student Information\n";
    std::cout << "-------------------\n";

    std::cout << "ID: " << student.id << '\n';
    std::cout << "Name: " << student.name << '\n';
    std::cout << "Age: " << student.age << '\n';
    std::cout << "CGPA: " << student.cgpa << '\n';

    return 0;
}
```

## Your first task

Don't copy-paste and move on.

Try to understand every line.

You should be able to answer:

1. What does `#include <iostream>` do?
2. Why do we need `<string>`?
3. What is `namespace University`?
4. What is `struct Student`?
5. What is `student`?
6. What is `student.id`?
7. Why is `main()` returning `int`?
8. What does `return 0` mean?
9. What's the difference between `Student` and `student`?
10. Where does the `student` object live in memory?

### Then modify the program yourself:

Create **three students** and print their information.

Don't use arrays or STL yet. Just:

```cpp
Student student1;
Student student2;
Student student3;
```

Once you've done that, **send me your code**.

I'll review it like a C++ mentor, point out anything wrong, and then we'll move to **Phase 1 → References**.
