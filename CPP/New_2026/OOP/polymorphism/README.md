# C++ Polymorphism — Zero to Advanced

A complete guide to **Polymorphism in C++**, starting from the basic concept and progressing to function overloading, operator overloading, function overriding, virtual functions, runtime polymorphism, abstract classes, interfaces, virtual destructors, casting, multiple inheritance, static polymorphism, CRTP, and practical object-oriented design.

---

# Table of Contents

- [C++ Polymorphism — Zero to Advanced](#c-polymorphism--zero-to-advanced)
- [Table of Contents](#table-of-contents)
- [1. What is Polymorphism?](#1-what-is-polymorphism)
- [2. Why Do We Need Polymorphism?](#2-why-do-we-need-polymorphism)
- [3. Types of Polymorphism in C++](#3-types-of-polymorphism-in-c)
    - [Compile-time polymorphism](#compile-time-polymorphism)
    - [Runtime polymorphism](#runtime-polymorphism)
    - [Static polymorphism](#static-polymorphism)
- [4. Compile-Time Polymorphism](#4-compile-time-polymorphism)
- [5. Function Overloading](#5-function-overloading)
    - [Important](#important)
- [6. Operator Overloading](#6-operator-overloading)
- [7. Runtime Polymorphism](#7-runtime-polymorphism)
- [8. Function Overriding](#8-function-overriding)
- [9. Virtual Functions](#9-virtual-functions)
- [10. `override`](#10-override)
    - [Recommended](#recommended)
- [11. Base Class Pointer](#11-base-class-pointer)
- [12. Base Class Reference](#12-base-class-reference)
- [13. How Runtime Polymorphism Works](#13-how-runtime-polymorphism-works)
- [14. Virtual Table — vtable](#14-virtual-table--vtable)
    - [Important](#important-1)
- [15. Virtual Pointer — vptr](#15-virtual-pointer--vptr)
- [16. Virtual Destructor](#16-virtual-destructor)
- [17. Pure Virtual Functions](#17-pure-virtual-functions)
- [18. Abstract Classes](#18-abstract-classes)
- [19. Interfaces](#19-interfaces)
- [20. Polymorphic Collections](#20-polymorphic-collections)
- [21. Smart Pointers and Polymorphism](#21-smart-pointers-and-polymorphism)
- [22. Object Slicing](#22-object-slicing)
    - [Avoid slicing](#avoid-slicing)
- [23. Upcasting](#23-upcasting)
- [24. Downcasting](#24-downcasting)
- [25. `dynamic_cast`](#25-dynamic_cast)
    - [Requirement](#requirement)
- [26. `static_cast`](#26-static_cast)
- [27. Polymorphism with Multiple Inheritance](#27-polymorphism-with-multiple-inheritance)
- [28. Static Polymorphism](#28-static-polymorphism)
- [29. Templates as Polymorphism](#29-templates-as-polymorphism)
- [30. CRTP](#30-crtp)
- [31. Compile-Time vs Runtime Polymorphism](#31-compile-time-vs-runtime-polymorphism)
    - [Compile-time](#compile-time)
    - [Runtime](#runtime)
- [32. Polymorphism vs Inheritance](#32-polymorphism-vs-inheritance)
    - [Inheritance](#inheritance)
    - [Polymorphism](#polymorphism)
- [33. Polymorphism vs Composition](#33-polymorphism-vs-composition)
- [34. Common Mistakes](#34-common-mistakes)
  - [Mistake 1 — Forgetting `virtual`](#mistake-1--forgetting-virtual)
  - [Mistake 2 — Not using `override`](#mistake-2--not-using-override)
  - [Mistake 3 — Non-virtual destructor](#mistake-3--non-virtual-destructor)
  - [Mistake 4 — Object slicing](#mistake-4--object-slicing)
  - [Mistake 5 — Unsafe downcasting](#mistake-5--unsafe-downcasting)
  - [Mistake 6 — Excessive `dynamic_cast`](#mistake-6--excessive-dynamic_cast)
- [35. Best Practices](#35-best-practices)
    - [1. Use `virtual` for runtime polymorphism](#1-use-virtual-for-runtime-polymorphism)
    - [2. Always use `override` for overriding](#2-always-use-override-for-overriding)
    - [3. Use virtual destructors for polymorphic bases](#3-use-virtual-destructors-for-polymorphic-bases)
    - [4. Prefer smart pointers for ownership](#4-prefer-smart-pointers-for-ownership)
    - [5. Avoid object slicing](#5-avoid-object-slicing)
    - [6. Prefer interfaces for behavior contracts](#6-prefer-interfaces-for-behavior-contracts)
    - [7. Don't use inheritance only for code reuse](#7-dont-use-inheritance-only-for-code-reuse)
    - [8. Keep base interfaces small](#8-keep-base-interfaces-small)
    - [9. Avoid unnecessary downcasting](#9-avoid-unnecessary-downcasting)
- [36. Real-World Example](#36-real-world-example)
- [37. Game Development Example](#37-game-development-example)
- [38. Polymorphism in Networking](#38-polymorphism-in-networking)
- [39. Polymorphism in Unreal Engine](#39-polymorphism-in-unreal-engine)
- [40. Learning Checklist](#40-learning-checklist)
  - [Beginner](#beginner)
  - [Intermediate](#intermediate)
  - [Advanced](#advanced)
- [41. Final Mental Model](#41-final-mental-model)
- [The Most Important Rules](#the-most-important-rules)
- [Recommended Practice Projects](#recommended-practice-projects)
  - [Project 1 — Animal Polymorphism](#project-1--animal-polymorphism)
  - [Project 2 — Shape Renderer](#project-2--shape-renderer)
  - [Project 3 — Payment System](#project-3--payment-system)
  - [Project 4 — Game Character System](#project-4--game-character-system)
  - [Project 5 — Network Message System](#project-5--network-message-system)
- [Final Goal](#final-goal)

---

# 1. What is Polymorphism?

The word **polymorphism** comes from:

```text
Poly = many
Morph = forms
```

So polymorphism means:

> **One interface, many possible forms or behaviors.**

In C++, polymorphism allows the same function call or interface to produce different behavior depending on the context or actual object.

For example:

```text
          Animal
             │
      ┌──────┼──────┐
      │      │      │
     Dog    Cat    Bird
      │      │      │
     bark   meow   fly
```

We can have:

```cpp
Animal* animal;
```

and that pointer can refer to:

```cpp
Dog
Cat
Bird
```

Then:

```cpp
animal->speak();
```

can produce different behavior.

---

# 2. Why Do We Need Polymorphism?

Without polymorphism, we may need code like:

```cpp
if (type == DOG)
{
    dog.speak();
}
else if (type == CAT)
{
    cat.speak();
}
else if (type == BIRD)
{
    bird.speak();
}
```

This becomes difficult to maintain as the number of types grows.

With polymorphism:

```cpp
animal->speak();
```

The object itself determines which implementation should execute.

This allows us to write code that works with a common interface rather than knowing every concrete type.

---

# 3. Types of Polymorphism in C++

C++ supports several forms of polymorphism.

```text
                    POLYMORPHISM
                         │
              ┌──────────┴──────────┐
              │                     │
       Compile-Time            Runtime
              │                     │
       ┌──────┴──────┐              │
       │             │              │
 Function       Operator        Virtual
 Overloading    Overloading      Functions
                                  │
                            Inheritance
                                  │
                            Base Pointer/
                            Base Reference
```

The major categories are:

### Compile-time polymorphism

* Function overloading
* Operator overloading
* Templates

### Runtime polymorphism

* Inheritance
* Virtual functions
* Function overriding
* Base class pointers/references

### Static polymorphism

Often implemented using:

* Templates
* Function overloading
* CRTP

---

# 4. Compile-Time Polymorphism

Compile-time polymorphism means the compiler determines which function or operation should be used.

Example:

```cpp
void print(int value)
{
    std::cout << "Integer\n";
}

void print(double value)
{
    std::cout << "Double\n";
}
```

Now:

```cpp
print(10);
```

calls:

```cpp
print(int);
```

while:

```cpp
print(3.14);
```

calls:

```cpp
print(double);
```

The decision happens during compilation.

---

# 5. Function Overloading

Function overloading means having multiple functions with the same name but different parameter lists.

```cpp
class Calculator
{
public:

    int add(int a, int b)
    {
        return a + b;
    }

    double add(double a, double b)
    {
        return a + b;
    }
};
```

Usage:

```cpp
Calculator calculator;

calculator.add(10, 20);
calculator.add(10.5, 20.5);
```

The compiler determines which function to call.

### Important

Changing only the return type is not enough.

This is invalid:

```cpp
int add(int a, int b);

double add(int a, int b);
```

The parameter lists are identical.

---

# 6. Operator Overloading

C++ allows operators to be overloaded for user-defined types.

Example:

```cpp
class Vector2
{
public:

    float x;
    float y;

    Vector2(float x, float y)
        : x(x), y(y)
    {
    }

    Vector2 operator+(const Vector2& other) const
    {
        return Vector2(
            x + other.x,
            y + other.y
        );
    }
};
```

Now:

```cpp
Vector2 a(10, 20);
Vector2 b(5, 10);

Vector2 c = a + b;
```

The `+` operator now has meaning for `Vector2`.

Conceptually:

```text
a + b
 ↓
operator+
 ↓
Vector2 result
```

---

# 7. Runtime Polymorphism

Runtime polymorphism occurs when the function to execute is determined at runtime.

It usually involves:

```text
Inheritance
      +
Virtual function
      +
Base pointer/reference
      +
Function overriding
```

Example:

```cpp
class Animal
{
public:

    virtual void speak()
    {
        std::cout << "Animal speaks\n";
    }

    virtual ~Animal() = default;
};

class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog barks\n";
    }
};
```

Now:

```cpp
Animal* animal = new Dog();

animal->speak();
```

Output:

```text
Dog barks
```

The pointer type is:

```cpp
Animal*
```

but the actual object is:

```cpp
Dog
```

Therefore the `Dog` implementation executes.

---

# 8. Function Overriding

Function overriding occurs when a derived class provides a new implementation of a virtual base function.

Base:

```cpp
class Animal
{
public:

    virtual void speak()
    {
        std::cout << "Animal speaks\n";
    }
};
```

Derived:

```cpp
class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog barks\n";
    }
};
```

The derived class overrides the base implementation.

---

# 9. Virtual Functions

A virtual function tells C++ that derived classes may provide their own implementation.

```cpp
class Animal
{
public:

    virtual void speak()
    {
        std::cout << "Animal\n";
    }
};
```

Without `virtual`:

```cpp
void speak();
```

the function call through a base pointer is resolved differently.

With:

```cpp
virtual void speak();
```

C++ can perform runtime dispatch.

---

# 10. `override`

Use `override` when overriding a virtual function.

```cpp
class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog\n";
    }
};
```

The compiler verifies that `speak()` actually overrides a virtual function from the base class.

This catches mistakes such as:

```cpp
void speek() override
```

The compiler will report an error.

### Recommended

Always prefer:

```cpp
void speak() override;
```

when overriding a virtual function.

---

# 11. Base Class Pointer

A base class pointer can point to a derived object.

```cpp
Dog dog;

Animal* animal = &dog;
```

This is called:

> **Upcasting**

Now:

```cpp
animal->speak();
```

If `speak()` is virtual, the derived implementation executes.

Example:

```cpp
class Animal
{
public:

    virtual void speak()
    {
        std::cout << "Animal\n";
    }

    virtual ~Animal() = default;
};

class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog\n";
    }
};
```

```cpp
Dog dog;

Animal* animal = &dog;

animal->speak();
```

Output:

```text
Dog
```

---

# 12. Base Class Reference

Runtime polymorphism also works through references.

```cpp
Dog dog;

Animal& animal = dog;

animal.speak();
```

Output:

```text
Dog
```

References are useful because they:

* Do not require `nullptr` checks
* Avoid pointer syntax
* Do not require ownership

---

# 13. How Runtime Polymorphism Works

Consider:

```cpp
class Animal
{
public:

    virtual void speak()
    {
        std::cout << "Animal\n";
    }
};
```

and:

```cpp
class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog\n";
    }
};
```

Then:

```cpp
Animal* animal = new Dog();

animal->speak();
```

Conceptually, C++ needs to determine:

```text
What is the actual object?

Animal*
   │
   ▼
actual object
   │
   ▼
Dog
   │
   ▼
Dog::speak()
```

This is called:

> **Dynamic dispatch / dynamic binding**

The exact implementation mechanism is compiler-dependent, but common C++ implementations use a virtual table.

---

# 14. Virtual Table — vtable

A class containing virtual functions is commonly implemented using a structure called a:

> **Virtual Table (vtable)**

Conceptually:

```text
Dog object
┌──────────────────┐
│ vptr             │
├──────────────────┤
│ Dog data         │
└──────────────────┘

vtable
┌──────────────────┐
│ Dog::speak()     │
├──────────────────┤
│ Dog::~Dog()      │
└──────────────────┘
```

When:

```cpp
animal->speak();
```

is executed, the runtime mechanism can locate the appropriate overridden function.

### Important

The C++ standard specifies virtual dispatch behavior, but it does not require a particular `vtable`/`vptr` implementation.

The `vtable` model is the common implementation strategy.

---

# 15. Virtual Pointer — vptr

A common implementation uses a hidden pointer called a:

> **vptr**

Conceptually:

```text
Object
┌───────────────┐
│ vptr          │ ──────► vtable
├───────────────┤
│ object data   │
└───────────────┘
```

Again, `vptr` is an implementation concept, not a C++ language keyword.

You should understand the concept without assuming that every compiler implements virtual dispatch identically.

---

# 16. Virtual Destructor

A polymorphic base class should generally have a virtual destructor.

Bad:

```cpp
class Animal
{
public:

    virtual void speak() = 0;

    ~Animal()
    {
    }
};
```

Better:

```cpp
class Animal
{
public:

    virtual void speak() = 0;

    virtual ~Animal() = default;
};
```

Now:

```cpp
Animal* animal = new Dog();

delete animal;
```

The derived object can be destroyed correctly through the base interface.

---

# 17. Pure Virtual Functions

A pure virtual function is declared using:

```cpp
= 0;
```

Example:

```cpp
class Animal
{
public:

    virtual void speak() = 0;
};
```

This means the base class defines an interface but does not provide the required concrete implementation.

---

# 18. Abstract Classes

A class containing at least one pure virtual function is abstract.

```cpp
class Animal
{
public:

    virtual void speak() = 0;
};
```

You cannot create:

```cpp
Animal animal;
```

But you can derive:

```cpp
class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog\n";
    }
};
```

Now:

```cpp
Dog dog;

dog.speak();
```

works.

---

# 19. Interfaces

C++ does not have a special `interface` keyword.

An interface-like class is commonly created using pure virtual functions.

```cpp
class IRenderable
{
public:

    virtual void render() = 0;

    virtual ~IRenderable() = default;
};
```

A class can implement it:

```cpp
class Player : public IRenderable
{
public:

    void render() override
    {
        std::cout << "Rendering player\n";
    }
};
```

The interface defines **what** a class must do.

The derived class defines **how** it does it.

---

# 20. Polymorphic Collections

One of the biggest advantages of runtime polymorphism is storing different derived objects through a common base type.

Example:

```cpp
class Animal
{
public:

    virtual void speak() = 0;

    virtual ~Animal() = default;
};
```

Derived:

```cpp
class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog\n";
    }
};

class Cat : public Animal
{
public:

    void speak() override
    {
        std::cout << "Cat\n";
    }
};
```

Now:

```cpp
std::vector<Animal*> animals;

Dog dog;
Cat cat;

animals.push_back(&dog);
animals.push_back(&cat);
```

Then:

```cpp
for (Animal* animal : animals)
{
    animal->speak();
}
```

Output:

```text
Dog
Cat
```

The collection doesn't need to know the exact derived types.

---

# 21. Smart Pointers and Polymorphism

For dynamically allocated polymorphic objects, smart pointers are generally preferable to raw owning pointers.

Use:

```cpp
std::unique_ptr
```

for single ownership.

Example:

```cpp
std::vector<std::unique_ptr<Animal>> animals;

animals.push_back(std::make_unique<Dog>());
animals.push_back(std::make_unique<Cat>());
```

Then:

```cpp
for (const auto& animal : animals)
{
    animal->speak();
}
```

Output:

```text
Dog
Cat
```

The smart pointers automatically manage object lifetime.

---

# 22. Object Slicing

Object slicing happens when a derived object is copied into a base object by value.

Example:

```cpp
class Animal
{
public:

    virtual void speak()
    {
        std::cout << "Animal\n";
    }
};

class Dog : public Animal
{
public:

    void speak() override
    {
        std::cout << "Dog\n";
    }
};
```

Now:

```cpp
Dog dog;

Animal animal = dog;
```

The derived portion is removed from the copied object.

Then:

```cpp
animal.speak();
```

produces:

```text
Animal
```

### Avoid slicing

Use:

```cpp
Animal& animal = dog;
```

or:

```cpp
Animal* animal = &dog;
```

---

# 23. Upcasting

Upcasting means converting:

```text
Derived → Base
```

Example:

```cpp
Dog dog;

Animal* animal = &dog;
```

This is generally safe with public inheritance.

The derived object can be treated as its base interface.

---

# 24. Downcasting

Downcasting means converting:

```text
Base → Derived
```

Example:

```cpp
Animal* animal = new Dog();

Dog* dog = dynamic_cast<Dog*>(animal);
```

Downcasting should be used carefully.

If the object isn't actually a `Dog`, the cast may fail.

---

# 25. `dynamic_cast`

`dynamic_cast` performs runtime-checked casting.

Example:

```cpp
class Animal
{
public:

    virtual ~Animal() = default;
};

class Dog : public Animal
{
};

class Cat : public Animal
{
};
```

Then:

```cpp
Animal* animal = new Dog();

Dog* dog = dynamic_cast<Dog*>(animal);
```

The cast succeeds.

But:

```cpp
Cat* cat = dynamic_cast<Cat*>(animal);
```

results in:

```cpp
nullptr
```

because the actual object is a `Dog`.

### Requirement

For the usual polymorphic downcast case, the source class needs to be polymorphic, typically by having at least one virtual function.

---

# 26. `static_cast`

You can also perform downcasting with:

```cpp
Dog* dog = static_cast<Dog*>(animal);
```

But `static_cast` does not perform the same runtime check as `dynamic_cast`.

Therefore, if the object is not actually a compatible `Dog`, using the resulting pointer can lead to undefined behavior.

Use `static_cast` only when the type relationship is already known and guaranteed by your program logic.

---

# 27. Polymorphism with Multiple Inheritance

Polymorphism can also work with multiple interfaces.

Example:

```cpp
class Printable
{
public:

    virtual void print() = 0;

    virtual ~Printable() = default;
};

class Serializable
{
public:

    virtual void serialize() = 0;

    virtual ~Serializable() = default;
};
```

Now:

```cpp
class Document :
    public Printable,
    public Serializable
{
public:

    void print() override
    {
        std::cout << "Printing\n";
    }

    void serialize() override
    {
        std::cout << "Serializing\n";
    }
};
```

The same object can be viewed through either interface:

```cpp
Document document;

Printable* printable = &document;
Serializable* serializable = &document;

printable->print();
serializable->serialize();
```

---

# 28. Static Polymorphism

Runtime polymorphism is not the only way to achieve polymorphic behavior.

C++ can also use **static polymorphism**.

The compiler determines the implementation at compile time.

Common tools include:

```text
Function overloading
Templates
CRTP
```

Example:

```cpp
template<typename T>
void print(T value)
{
    std::cout << value << '\n';
}
```

Now:

```cpp
print(10);
print(3.14);
print("Hello");
```

The compiler generates the appropriate version.

---

# 29. Templates as Polymorphism

Templates allow generic code to work with different types.

Example:

```cpp
template<typename T>
T add(T a, T b)
{
    return a + b;
}
```

Usage:

```cpp
int a = add(10, 20);

double b = add(10.5, 20.5);
```

The function works with multiple types without requiring a common inheritance hierarchy.

This is often called:

> **Parametric polymorphism**

---

# 30. CRTP

CRTP stands for:

> **Curiously Recurring Template Pattern**

Example:

```cpp
template<typename Derived>
class Base
{
public:

    void interface()
    {
        static_cast<Derived*>(this)->implementation();
    }
};
```

Derived class:

```cpp
class Player : public Base<Player>
{
public:

    void implementation()
    {
        std::cout << "Player implementation\n";
    }
};
```

Usage:

```cpp
Player player;

player.interface();
```

Output:

```text
Player implementation
```

The derived type is known at compile time.

Therefore CRTP can provide static polymorphism.

---

# 31. Compile-Time vs Runtime Polymorphism

| Feature              | Compile-Time       | Runtime                   |
| -------------------- | ------------------ | ------------------------- |
| Decision             | Compile time       | Runtime                   |
| Common mechanism     | Templates          | Virtual functions         |
| Inheritance required | Not necessarily    | Usually                   |
| Virtual function     | No                 | Yes                       |
| Runtime dispatch     | No                 | Yes                       |
| Flexibility          | Compile-time types | Runtime types             |
| Typical overhead     | Usually lower      | Virtual dispatch overhead |
| Example              | Templates          | Base pointer + virtual    |

### Compile-time

```cpp
template<typename T>
void process(T& object)
{
    object.run();
}
```

### Runtime

```cpp
Base* object = new Derived();

object->run();
```

---

# 32. Polymorphism vs Inheritance

Inheritance and polymorphism are related, but they are not the same thing.

### Inheritance

Describes a relationship:

```text
Dog IS-A Animal
```

### Polymorphism

Allows one interface to represent different behaviors:

```text
Animal*
   │
   ├── Dog
   ├── Cat
   └── Bird
```

Inheritance can be used to implement runtime polymorphism, but polymorphism can also be achieved through:

* Templates
* Function overloading
* Operator overloading
* CRTP

---

# 33. Polymorphism vs Composition

Polymorphism:

```cpp
class Player : public Character
{
};
```

Composition:

```cpp
class Player
{
private:

    MovementComponent movement;
    WeaponComponent weapon;
};
```

In a large system, both are useful.

A good design might use:

```text
Inheritance
    ↓
common identity/interface

Composition
    ↓
individual functionality
```

For example:

```text
Character
    │
    ├── Player
    └── Enemy

Player
 ├── MovementComponent
 ├── HealthComponent
 └── InventoryComponent
```

---

# 34. Common Mistakes

## Mistake 1 — Forgetting `virtual`

```cpp
class Animal
{
public:
    void speak();
};
```

If runtime polymorphism is required, this should normally be:

```cpp
virtual void speak();
```

---

## Mistake 2 — Not using `override`

Instead of:

```cpp
void speak();
```

prefer:

```cpp
void speak() override;
```

---

## Mistake 3 — Non-virtual destructor

Bad for a polymorphic base:

```cpp
~Animal();
```

Prefer:

```cpp
virtual ~Animal() = default;
```

---

## Mistake 4 — Object slicing

Avoid:

```cpp
Animal animal = dog;
```

when polymorphic behavior is expected.

---

## Mistake 5 — Unsafe downcasting

Avoid unnecessary:

```cpp
static_cast<Dog*>(animal);
```

when the actual type is uncertain.

---

## Mistake 6 — Excessive `dynamic_cast`

If your code constantly does:

```cpp
if (auto dog = dynamic_cast<Dog*>(animal))
{
}
else if (auto cat = dynamic_cast<Cat*>(animal))
{
}
```

you may have a design problem.

Often the base interface should expose the operation you actually need.

---

# 35. Best Practices

### 1. Use `virtual` for runtime polymorphism

```cpp
virtual void update();
```

### 2. Always use `override` for overriding

```cpp
void update() override;
```

### 3. Use virtual destructors for polymorphic bases

```cpp
virtual ~Base() = default;
```

### 4. Prefer smart pointers for ownership

```cpp
std::unique_ptr<Base>
```

### 5. Avoid object slicing

Use:

```cpp
Base&
```

or:

```cpp
Base*
```

when polymorphism is required.

### 6. Prefer interfaces for behavior contracts

```cpp
class IRenderable
{
public:
    virtual void render() = 0;
    virtual ~IRenderable() = default;
};
```

### 7. Don't use inheritance only for code reuse

Use composition when appropriate.

### 8. Keep base interfaces small

A base class should expose meaningful common behavior.

### 9. Avoid unnecessary downcasting

Good polymorphic design usually minimizes the need for it.

---

# 36. Real-World Example

Consider a payment system.

```text
                Payment
                   │
       ┌───────────┼───────────┐
       │           │           │
     Card        Bkash       PayPal
```

Base:

```cpp
class Payment
{
public:

    virtual void pay(double amount) = 0;

    virtual ~Payment() = default;
};
```

Card:

```cpp
class CardPayment : public Payment
{
public:

    void pay(double amount) override
    {
        std::cout << "Paid by card: "
                  << amount << '\n';
    }
};
```

Bkash:

```cpp
class BkashPayment : public Payment
{
public:

    void pay(double amount) override
    {
        std::cout << "Paid using Bkash: "
                  << amount << '\n';
    }
};
```

PayPal:

```cpp
class PayPalPayment : public Payment
{
public:

    void pay(double amount) override
    {
        std::cout << "Paid using PayPal: "
                  << amount << '\n';
    }
};
```

Now:

```cpp
std::vector<std::unique_ptr<Payment>> payments;

payments.push_back(
    std::make_unique<CardPayment>()
);

payments.push_back(
    std::make_unique<BkashPayment>()
);

payments.push_back(
    std::make_unique<PayPalPayment>()
);
```

Then:

```cpp
for (auto& payment : payments)
{
    payment->pay(1000);
}
```

The caller doesn't need to know which payment system is being used.

---

# 37. Game Development Example

Polymorphism is particularly useful in game architecture.

Consider:

```text
                   Character
                       │
             ┌─────────┴─────────┐
             │                   │
           Player              Enemy
                                 │
                       ┌─────────┴─────────┐
                       │                   │
                    Zombie               Boss
```

Base:

```cpp
class Character
{
public:

    virtual void attack() = 0;

    virtual void update() = 0;

    virtual ~Character() = default;
};
```

Player:

```cpp
class Player : public Character
{
public:

    void attack() override
    {
        std::cout << "Player attacks\n";
    }

    void update() override
    {
        std::cout << "Updating player\n";
    }
};
```

Zombie:

```cpp
class Zombie : public Character
{
public:

    void attack() override
    {
        std::cout << "Zombie attacks\n";
    }

    void update() override
    {
        std::cout << "Updating zombie\n";
    }
};
```

Now:

```cpp
std::vector<std::unique_ptr<Character>> characters;

characters.push_back(
    std::make_unique<Player>()
);

characters.push_back(
    std::make_unique<Zombie>()
);
```

Game loop:

```cpp
for (auto& character : characters)
{
    character->update();
    character->attack();
}
```

The game loop doesn't need:

```cpp
if player...
if zombie...
if boss...
```

Each object provides its own implementation.

---

# 38. Polymorphism in Networking

Polymorphism can also be useful when designing networking systems.

For example:

```text
                NetworkMessage
                     │
          ┌──────────┼──────────┐
          │          │          │
       Login       Chat       Movement
       Message     Message     Message
```

Base:

```cpp
class NetworkMessage
{
public:

    virtual void serialize() = 0;

    virtual ~NetworkMessage() = default;
};
```

Derived:

```cpp
class LoginMessage : public NetworkMessage
{
public:

    void serialize() override
    {
        std::cout << "Serialize login\n";
    }
};
```

```cpp
class MovementMessage : public NetworkMessage
{
public:

    void serialize() override
    {
        std::cout << "Serialize movement\n";
    }
};
```

Now a system can work with:

```cpp
NetworkMessage*
```

without needing to know the concrete message type.

This becomes useful when designing larger networking architectures.

---

# 39. Polymorphism in Unreal Engine

Polymorphism is heavily relevant to Unreal Engine C++.

A simplified hierarchy might look like:

```text
UObject
   │
   └── AActor
         │
         ├── APawn
         │
         └── Other Actor classes
```

Your custom class can derive from an Unreal base class:

```cpp
class AMyCharacter : public ACharacter
{
    GENERATED_BODY()

public:

    virtual void BeginPlay() override;
};
```

Here:

```cpp
virtual void BeginPlay() override;
```

uses C++ virtual-function polymorphism.

Unreal gameplay systems also make extensive use of:

* Base classes
* Derived classes
* Virtual functions
* Interfaces
* Components
* Delegates
* Events
* Runtime object types

Understanding C++ polymorphism therefore becomes an important foundation for Unreal C++.

---

# 40. Learning Checklist

## Beginner

* [ ] Understand the meaning of polymorphism
* [ ] Understand "one interface, many behaviors"
* [ ] Understand function overloading
* [ ] Understand operator overloading
* [ ] Understand inheritance and polymorphism relationship

## Intermediate

* [ ] Understand runtime polymorphism
* [ ] Understand virtual functions
* [ ] Understand overriding
* [ ] Understand `override`
* [ ] Understand base pointers
* [ ] Understand base references
* [ ] Understand dynamic dispatch
* [ ] Understand virtual destructors
* [ ] Understand pure virtual functions
* [ ] Understand abstract classes
* [ ] Understand interfaces
* [ ] Understand object slicing

## Advanced

* [ ] Understand vtable concept
* [ ] Understand vptr concept
* [ ] Understand upcasting
* [ ] Understand downcasting
* [ ] Understand `dynamic_cast`
* [ ] Understand `static_cast`
* [ ] Understand polymorphism with multiple inheritance
* [ ] Understand smart-pointer-based polymorphism
* [ ] Understand static polymorphism
* [ ] Understand template-based polymorphism
* [ ] Understand CRTP
* [ ] Understand composition vs polymorphism
* [ ] Understand polymorphic architecture

---

# 41. Final Mental Model

Think about polymorphism as:

```text
                         POLYMORPHISM
                              │
              ┌───────────────┴───────────────┐
              │                               │
       COMPILE-TIME                      RUNTIME
              │                               │
      ┌───────┼───────┐              ┌────────┼────────┐
      │       │       │              │        │        │
 Function Operator Templates      Inheritance virtual  override
 Overload  Overload                         │
                                            │
                                    Base Pointer
                                            │
                                    Base Reference
                                            │
                                    Dynamic Dispatch
                                            │
                                    Derived Behavior
```

The central idea is:

```text
One interface
      ↓
Many implementations
      ↓
Caller doesn't need to know
the exact concrete type
```

For example:

```cpp
Animal* animal;
```

could represent:

```text
Dog
Cat
Bird
```

and:

```cpp
animal->speak();
```

can produce:

```text
Dog  → bark
Cat  → meow
Bird → chirp
```

---

# The Most Important Rules

```text
1. Polymorphism means one interface can represent many behaviors.

2. C++ has both compile-time and runtime polymorphism.

3. Function overloading is compile-time polymorphism.

4. Operator overloading is compile-time polymorphism.

5. Templates provide a powerful form of compile-time polymorphism.

6. Runtime polymorphism commonly uses inheritance + virtual functions.

7. Use override when overriding virtual functions.

8. Polymorphic base classes should generally have virtual destructors.

9. Avoid object slicing when polymorphism is required.

10. Use smart pointers when they should own polymorphic objects.

11. Avoid unnecessary downcasting.

12. Prefer a clean base interface instead of repeatedly checking
    derived types.

13. Inheritance and polymorphism are related but are not the same concept.

14. Composition and polymorphism can be used together.

15. Good polymorphism allows systems to work with abstractions
    instead of concrete implementations.
```

---

# Recommended Practice Projects

## Project 1 — Animal Polymorphism

```text
Animal
├── Dog
├── Cat
└── Bird
```

Practice:

* `virtual`
* `override`
* Base pointer
* Base reference

---

## Project 2 — Shape Renderer

```text
Shape
├── Circle
├── Rectangle
└── Triangle
```

Implement:

```cpp
virtual void draw() = 0;
```

Store objects using:

```cpp
std::vector<std::unique_ptr<Shape>>
```

---

## Project 3 — Payment System

```text
Payment
├── CardPayment
├── BkashPayment
└── PayPalPayment
```

Practice:

* Abstract classes
* Interfaces
* Runtime polymorphism
* Smart pointers

---

## Project 4 — Game Character System

```text
Character
├── Player
├── Enemy
│   ├── Zombie
│   └── Boss
└── NPC
```

Practice:

* Virtual functions
* Abstract classes
* Runtime polymorphism
* Composition
* Smart pointers

---

## Project 5 — Network Message System

```text
NetworkMessage
├── LoginMessage
├── ChatMessage
├── MovementMessage
└── DisconnectMessage
```

Practice:

* Abstract interfaces
* Serialization
* Polymorphic collections
* Networking architecture

---

# Final Goal

You should not measure your polymorphism knowledge by how many keywords you remember.

You should eventually be able to look at:

```cpp
std::vector<std::unique_ptr<Character>> characters;

for (auto& character : characters)
{
    character->update();
    character->attack();
}
```

and explain:

* Why `Character` is a base abstraction
* Why `update()` and `attack()` are virtual
* Why `override` is used
* Why the vector stores `unique_ptr<Character>`
* How different derived objects can exist in the same collection
* How runtime dispatch selects the correct implementation
* Why the base destructor should be virtual
* Why object slicing is avoided
* When `dynamic_cast` might be needed
* When composition would be better than inheritance
* How this architecture can scale into a game or networking system

The ultimate goal is:

```text
C++ Fundamentals
       ↓
OOP
       ↓
Inheritance
       ↓
Polymorphism
       ↓
RAII + Smart Pointers
       ↓
Templates
       ↓
Multithreading
       ↓
Networking
       ↓
Unreal Engine
```

Polymorphism becomes especially important once you move from small C++ programs into **larger systems where many different objects must be handled through common interfaces**.
