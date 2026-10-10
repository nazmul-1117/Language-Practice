# C++ Inheritance — Zero to Advanced

A complete guide to **Inheritance in C++**, starting from the fundamentals and progressing to advanced concepts such as virtual functions, overriding, abstract classes, multiple inheritance, the diamond problem, virtual inheritance, object slicing, and polymorphic design.

---

## Table of Contents

- [C++ Inheritance — Zero to Advanced](#c-inheritance--zero-to-advanced)
  - [Table of Contents](#table-of-contents)
- [1. What is Inheritance?](#1-what-is-inheritance)
    - [Basic idea](#basic-idea)
- [2. Why Do We Need Inheritance?](#2-why-do-we-need-inheritance)
- [3. Basic Syntax](#3-basic-syntax)
- [4. Base Class and Derived Class](#4-base-class-and-derived-class)
- [5. Simple Example](#5-simple-example)
- [6. What Gets Inherited?](#6-what-gets-inherited)
- [7. What Does Not Get Inherited?](#7-what-does-not-get-inherited)
- [8. Access Specifiers](#8-access-specifiers)
    - [Public](#public)
    - [Protected](#protected)
    - [Private](#private)
- [9. Public, Protected and Private Inheritance](#9-public-protected-and-private-inheritance)
- [10. Public Inheritance](#10-public-inheritance)
- [11. Protected Inheritance](#11-protected-inheritance)
- [12. Private Inheritance](#12-private-inheritance)
- [13. Constructor and Destructor Order](#13-constructor-and-destructor-order)
- [14. Calling Base Class Constructors](#14-calling-base-class-constructors)
- [15. Calling Base Class Functions](#15-calling-base-class-functions)
- [16. Function Overriding](#16-function-overriding)
- [17. `virtual` Functions](#17-virtual-functions)
- [18. `override`](#18-override)
    - [Recommended](#recommended)
- [19. `final`](#19-final)
- [20. Runtime Polymorphism](#20-runtime-polymorphism)
- [21. Base Class Pointer](#21-base-class-pointer)
- [22. Base Class Reference](#22-base-class-reference)
- [23. Virtual Destructor](#23-virtual-destructor)
    - [Rule](#rule)
- [24. Pure Virtual Functions](#24-pure-virtual-functions)
- [25. Abstract Classes](#25-abstract-classes)
- [26. Interfaces in C++](#26-interfaces-in-c)
- [27. Multilevel Inheritance](#27-multilevel-inheritance)
- [28. Hierarchical Inheritance](#28-hierarchical-inheritance)
- [29. Multiple Inheritance](#29-multiple-inheritance)
- [30. The Diamond Problem](#30-the-diamond-problem)
- [31. Virtual Inheritance](#31-virtual-inheritance)
- [32. Object Slicing](#32-object-slicing)
    - [Avoid slicing](#avoid-slicing)
- [33. Name Hiding](#33-name-hiding)
- [34. `using` with Inheritance](#34-using-with-inheritance)
- [35. Static Members and Inheritance](#35-static-members-and-inheritance)
- [36. `protected` Members](#36-protected-members)
    - [Design note](#design-note)
- [37. Inheritance and Friend Functions](#37-inheritance-and-friend-functions)
- [38. Inheritance and Templates](#38-inheritance-and-templates)
- [39. Inheritance vs Composition](#39-inheritance-vs-composition)
- [40. "IS-A" vs "HAS-A"](#40-is-a-vs-has-a)
    - [Inheritance](#inheritance)
    - [Composition](#composition)
- [41. Upcasting](#41-upcasting)
- [42. Downcasting](#42-downcasting)
- [43. `dynamic_cast`](#43-dynamic_cast)
- [44. `static_cast` and Downcasting](#44-static_cast-and-downcasting)
- [45. CRTP](#45-crtp)
- [46. Multiple Inheritance Best Practices](#46-multiple-inheritance-best-practices)
- [47. Common Mistakes](#47-common-mistakes)
  - [Mistake 1 — Forgetting `virtual`](#mistake-1--forgetting-virtual)
  - [Mistake 2 — Forgetting `override`](#mistake-2--forgetting-override)
  - [Mistake 3 — Non-virtual destructor](#mistake-3--non-virtual-destructor)
  - [Mistake 4 — Excessive inheritance](#mistake-4--excessive-inheritance)
  - [Mistake 5 — Deep inheritance hierarchies](#mistake-5--deep-inheritance-hierarchies)
  - [Mistake 6 — Object slicing](#mistake-6--object-slicing)
- [48. Best Practices](#48-best-practices)
    - [1. Prefer `override`](#1-prefer-override)
    - [2. Use virtual destructors for polymorphic bases](#2-use-virtual-destructors-for-polymorphic-bases)
    - [3. Prefer public inheritance for genuine IS-A relationships](#3-prefer-public-inheritance-for-genuine-is-a-relationships)
    - [4. Don't expose data unnecessarily](#4-dont-expose-data-unnecessarily)
    - [5. Prefer composition when the relationship is HAS-A](#5-prefer-composition-when-the-relationship-is-has-a)
    - [6. Avoid unnecessary inheritance](#6-avoid-unnecessary-inheritance)
    - [7. Keep inheritance hierarchies understandable](#7-keep-inheritance-hierarchies-understandable)
- [49. Real-World Example](#49-real-world-example)
- [50. Inheritance in Large C++ Projects](#50-inheritance-in-large-c-projects)
- [51. Inheritance and Unreal Engine](#51-inheritance-and-unreal-engine)
- [52. Learning Checklist](#52-learning-checklist)
  - [Beginner](#beginner)
  - [Intermediate](#intermediate)
  - [Advanced](#advanced)
- [Final Mental Model](#final-mental-model)
  - [The Most Important Rules](#the-most-important-rules)
  - [Recommended Practice Projects](#recommended-practice-projects)
    - [Project 1 — Animal System](#project-1--animal-system)
    - [Project 2 — Vehicle System](#project-2--vehicle-system)
    - [Project 3 — Employee System](#project-3--employee-system)
    - [Project 4 — Game Character System](#project-4--game-character-system)
    - [Project 5 — Multiplayer Entity System](#project-5--multiplayer-entity-system)
  - [Final Goal](#final-goal)

---

# 1. What is Inheritance?

**Inheritance** is an object-oriented programming mechanism that allows one class to acquire properties and behaviors from another class.

The existing class is called the:

> **Base class / Parent class / Superclass**

The new class is called the:

> **Derived class / Child class / Subclass**

### Basic idea

```text
        Animal
          │
     ┌────┴────┐
     ▼         ▼
    Dog       Cat
```

`Dog` and `Cat` inherit common characteristics from `Animal`.

---

# 2. Why Do We Need Inheritance?

Inheritance can help us:

* Reuse code
* Represent relationships between objects
* Create specialized versions of existing classes
* Implement runtime polymorphism
* Build extensible class hierarchies

For example:

```cpp
class Animal
{
public:
    void eat()
    {
        std::cout << "Animal is eating\n";
    }
};
```

Instead of rewriting `eat()` for every animal:

```cpp
class Dog : public Animal
{
};

class Cat : public Animal
{
};
```

Both classes can use `eat()`.

---

# 3. Basic Syntax

```cpp
class Derived : public Base
{
    // derived class members
};
```

Example:

```cpp
class Animal
{
public:
    void eat()
    {
        std::cout << "Eating\n";
    }
};

class Dog : public Animal
{
};
```

Now:

```cpp
Dog dog;

dog.eat();
```

Output:

```text
Eating
```

---

# 4. Base Class and Derived Class

Consider:

```cpp
class Animal
{
public:
    void eat()
    {
        std::cout << "Animal eats\n";
    }
};

class Dog : public Animal
{
public:
    void bark()
    {
        std::cout << "Dog barks\n";
    }
};
```

Here:

```text
Animal
  ↑
  │ inheritance
  │
 Dog
```

`Animal` is the **base class**.

`Dog` is the **derived class**.

`Dog` has access to its own members and the accessible members inherited from `Animal`.

---

# 5. Simple Example

```cpp
#include <iostream>

class Animal
{
public:

    void eat()
    {
        std::cout << "Animal is eating\n";
    }
};

class Dog : public Animal
{
public:

    void bark()
    {
        std::cout << "Dog is barking\n";
    }
};

int main()
{
    Dog dog;

    dog.eat();
    dog.bark();

    return 0;
}
```

Output:

```text
Animal is eating
Dog is barking
```

The `Dog` object can use `eat()` because `Dog` inherits from `Animal`.

---

# 6. What Gets Inherited?

A derived class can access inherited members depending on their access level and the type of inheritance.

For example:

```cpp
class Animal
{
public:
    int age;

protected:
    int weight;

private:
    int secret;
};
```

A derived class can directly access:

```text
public      → yes
protected   → yes
private     → no
```

Example:

```cpp
class Dog : public Animal
{
public:

    void test()
    {
        age = 10;       // OK
        weight = 20;    // OK

        // secret = 30; // ERROR
    }
};
```

---

# 7. What Does Not Get Inherited?

Private members are not directly accessible from the derived class.

```cpp
class Parent
{
private:
    int x;
};

class Child : public Parent
{
public:

    void test()
    {
        // x = 10; // ERROR
    }
};
```

However, the base class's private data still exists as part of the derived object.

It simply cannot be accessed directly by the derived class.

---

# 8. Access Specifiers

C++ provides three main access specifiers:

```text
public
protected
private
```

### Public

Accessible from anywhere where the object is accessible.

### Protected

Accessible from:

* The class itself
* Derived classes
* Friends

### Private

Accessible from:

* The class itself
* Friends

Not directly accessible from derived classes.

---

# 9. Public, Protected and Private Inheritance

Inheritance itself can be:

```cpp
class Child : public Parent
{
};
```

or:

```cpp
class Child : protected Parent
{
};
```

or:

```cpp
class Child : private Parent
{
};
```

These control how the base class's `public` and `protected` members appear inside the derived class.

---

# 10. Public Inheritance

Public inheritance is the most common form.

```cpp
class Dog : public Animal
{
};
```

Access transformation:

| Base Member | In Derived Class |
| ----------- | ---------------- |
| `public`    | `public`         |
| `protected` | `protected`      |
| `private`   | inaccessible     |

Example:

```cpp
class Animal
{
public:
    void eat() {}
    
protected:
    int age;
    
private:
    int secret;
};

class Dog : public Animal
{
};
```

Conceptually:

```text
Animal public    → Dog public
Animal protected → Dog protected
Animal private   → inaccessible
```

Public inheritance normally represents an:

> **IS-A relationship**

Example:

```text
Dog IS-A Animal
Car IS-A Vehicle
Cat IS-A Animal
```

---

# 11. Protected Inheritance

```cpp
class Dog : protected Animal
{
};
```

Transformation:

| Base Member | Derived      |
| ----------- | ------------ |
| `public`    | `protected`  |
| `protected` | `protected`  |
| `private`   | inaccessible |

Example:

```cpp
class Animal
{
public:
    void eat() {}
};

class Dog : protected Animal
{
};
```

Now:

```cpp
Dog dog;

// dog.eat(); // ERROR
```

`eat()` became protected inside `Dog`.

---

# 12. Private Inheritance

```cpp
class Dog : private Animal
{
};
```

Transformation:

| Base Member | Derived      |
| ----------- | ------------ |
| `public`    | `private`    |
| `protected` | `private`    |
| `private`   | inaccessible |

Example:

```cpp
class Animal
{
public:
    void eat() {}
};

class Dog : private Animal
{
};
```

Now:

```cpp
Dog dog;

// dog.eat(); // ERROR
```

---

# 13. Constructor and Destructor Order

When a derived object is created:

```text
Base constructor
       ↓
Derived constructor
```

When it is destroyed:

```text
Derived destructor
       ↓
Base destructor
```

Example:

```cpp
#include <iostream>

class Base
{
public:

    Base()
    {
        std::cout << "Base constructor\n";
    }

    ~Base()
    {
        std::cout << "Base destructor\n";
    }
};

class Derived : public Base
{
public:

    Derived()
    {
        std::cout << "Derived constructor\n";
    }

    ~Derived()
    {
        std::cout << "Derived destructor\n";
    }
};
```

```cpp
int main()
{
    Derived obj;
}
```

Output:

```text
Base constructor
Derived constructor
Derived destructor
Base destructor
```

---

# 14. Calling Base Class Constructors

A derived class can explicitly call a base constructor through the initializer list.

```cpp
class Animal
{
public:

    Animal(int age)
    {
        std::cout << "Age: " << age << '\n';
    }
};

class Dog : public Animal
{
public:

    Dog(int age)
        : Animal(age)
    {
    }
};
```

Now:

```cpp
Dog dog(5);
```

The `Animal(int)` constructor executes before the `Dog` constructor body.

---

# 15. Calling Base Class Functions

You can explicitly access a base implementation using:

```cpp
Base::function();
```

Example:

```cpp
class Animal
{
public:

    void speak()
    {
        std::cout << "Animal speaks\n";
    }
};

class Dog : public Animal
{
public:

    void speak()
    {
        Animal::speak();

        std::cout << "Dog barks\n";
    }
};
```

Output:

```text
Animal speaks
Dog barks
```

---

# 16. Function Overriding

A derived class can provide a new implementation of a base class function.

```cpp
class Animal
{
public:

    void speak()
    {
        std::cout << "Animal speaks\n";
    }
};

class Dog : public Animal
{
public:

    void speak()
    {
        std::cout << "Dog barks\n";
    }
};
```

```cpp
Dog dog;

dog.speak();
```

Output:

```text
Dog barks
```

However, this is **not runtime polymorphism** because the base function is not virtual.

---

# 17. `virtual` Functions

A virtual function enables runtime polymorphism.

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

Now:

```cpp
Animal* animal = new Dog();

animal->speak();

delete animal;
```

Output:

```text
Dog barks
```

The function is selected according to the **actual object type at runtime**.

---

# 18. `override`

Use `override` when you intentionally override a virtual function.

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

`override` allows the compiler to verify that you are actually overriding a base virtual function.

For example:

```cpp
void speek() override
```

would produce a compiler error because `speek()` does not override `speak()`.

### Recommended

Prefer:

```cpp
void speak() override;
```

over:

```cpp
void speak();
```

when overriding a virtual function.

---

# 19. `final`

`final` prevents further overriding.

```cpp
class Animal
{
public:

    virtual void speak() final
    {
        std::cout << "Animal\n";
    }
};
```

Now:

```cpp
class Dog : public Animal
{
public:

    // ERROR
    void speak() override;
};
```

You can also make an entire class non-inheritable:

```cpp
class Dog final
{
};
```

Then:

```cpp
class Puppy : public Dog
{
};
```

is invalid.

---

# 20. Runtime Polymorphism

Runtime polymorphism means the program determines which overridden function to call at runtime.

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
Animal* a1 = new Dog();
Animal* a2 = new Cat();

a1->speak();
a2->speak();

delete a1;
delete a2;
```

Output:

```text
Dog
Cat
```

One base pointer can refer to different derived types.

---

# 21. Base Class Pointer

A pointer to a base class can point to a derived object.

```cpp
Dog dog;

Animal* animal = &dog;
```

This is called:

> **Upcasting**

It is generally safe with public inheritance.

Example:

```cpp
Animal* animal = new Dog();
```

---

# 22. Base Class Reference

The same concept works with references.

```cpp
Dog dog;

Animal& animal = dog;
```

Then:

```cpp
animal.speak();
```

If `speak()` is virtual, the derived implementation executes.

---

# 23. Virtual Destructor

This is extremely important when using polymorphism.

Consider:

```cpp
class Animal
{
public:
    ~Animal() {}
};

class Dog : public Animal
{
public:
    ~Dog() {}
};
```

Then:

```cpp
Animal* animal = new Dog();

delete animal;
```

If the base destructor is not virtual, deleting through the base pointer can result in incorrect destruction behavior and undefined behavior.

Use:

```cpp
class Animal
{
public:

    virtual ~Animal() = default;
};
```

Then:

```cpp
class Dog : public Animal
{
public:

    ~Dog() override = default;
};
```

### Rule

If a class is intended to be used polymorphically, its destructor should generally be virtual.

---

# 24. Pure Virtual Functions

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

This means:

> Derived classes must provide an implementation if they are to be concrete.

---

# 25. Abstract Classes

A class containing at least one pure virtual function is an **abstract class**.

```cpp
class Animal
{
public:

    virtual void speak() = 0;
};
```

You cannot create:

```cpp
Animal animal; // ERROR
```

But you can create:

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

Then:

```cpp
Dog dog;
dog.speak();
```

---

# 26. Interfaces in C++

C++ does not have a dedicated `interface` keyword like some languages.

An interface-like class is commonly created using pure virtual functions.

```cpp
class IPrintable
{
public:

    virtual void print() = 0;

    virtual ~IPrintable() = default;
};
```

A class can implement it:

```cpp
class Document : public IPrintable
{
public:

    void print() override
    {
        std::cout << "Printing document\n";
    }
};
```

The `I` prefix is a common naming convention, not a C++ requirement.

---

# 27. Multilevel Inheritance

Inheritance can form multiple levels.

```text
Animal
  ↓
Mammal
  ↓
Dog
```

Example:

```cpp
class Animal
{
public:

    void eat()
    {
        std::cout << "Eating\n";
    }
};

class Mammal : public Animal
{
public:

    void breathe()
    {
        std::cout << "Breathing\n";
    }
};

class Dog : public Mammal
{
public:

    void bark()
    {
        std::cout << "Barking\n";
    }
};
```

Now:

```cpp
Dog dog;

dog.eat();
dog.breathe();
dog.bark();
```

---

# 28. Hierarchical Inheritance

Multiple classes derive from one base class.

```text
        Animal
       /      \
      /        \
    Dog        Cat
```

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
};

class Cat : public Animal
{
};
```

---

# 29. Multiple Inheritance

A class can inherit from multiple base classes.

```cpp
class Printer
{
public:

    void print()
    {
        std::cout << "Printing\n";
    }
};

class Scanner
{
public:

    void scan()
    {
        std::cout << "Scanning\n";
    }
};

class AllInOne : public Printer, public Scanner
{
};
```

Now:

```cpp
AllInOne device;

device.print();
device.scan();
```

---

# 30. The Diamond Problem

Multiple inheritance can create the **diamond problem**.

```text
          A
        /   \
       B     C
        \   /
          D
```

Example:

```cpp
class A
{
public:

    int value;
};

class B : public A
{
};

class C : public A
{
};

class D : public B, public C
{
};
```

Now:

```cpp
D object;

// object.value = 10; // ERROR
```

Why?

Because `D` contains two separate copies of `A`:

```text
D
├── B
│   └── A
│
└── C
    └── A
```

Which `value` should be used?

The compiler cannot choose.

---

# 31. Virtual Inheritance

Virtual inheritance solves the diamond problem.

```cpp
class A
{
public:

    int value;
};

class B : virtual public A
{
};

class C : virtual public A
{
};

class D : public B, public C
{
};
```

Now `D` contains one shared `A` subobject.

```text
        A
       / \
      B   C
       \ /
        D
```

Now:

```cpp
D object;

object.value = 10;
```

is unambiguous.

---

# 32. Object Slicing

Object slicing happens when a derived object is copied into a base object **by value**.

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

The derived portion is sliced away.

```cpp
animal.speak();
```

Output:

```text
Animal
```

### Avoid slicing

Use a reference:

```cpp
Animal& animal = dog;
```

or pointer:

```cpp
Animal* animal = &dog;
```

---

# 33. Name Hiding

A derived class can hide overloaded functions from the base class.

```cpp
class Base
{
public:

    void print(int)
    {
        std::cout << "int\n";
    }

    void print(double)
    {
        std::cout << "double\n";
    }
};

class Derived : public Base
{
public:

    void print(std::string)
    {
        std::cout << "string\n";
    }
};
```

Now:

```cpp
Derived obj;

// obj.print(10); // ERROR
```

The derived `print()` hides the base overloads.

---

# 34. `using` with Inheritance

You can bring hidden base overloads into the derived scope.

```cpp
class Derived : public Base
{
public:

    using Base::print;

    void print(std::string)
    {
        std::cout << "string\n";
    }
};
```

Now:

```cpp
Derived obj;

obj.print(10);
obj.print(3.14);
obj.print("Hello");
```

All overloads are available.

---

# 35. Static Members and Inheritance

Static members belong to the class rather than individual objects.

```cpp
class Animal
{
public:

    static int count;
};

int Animal::count = 0;
```

Derived classes can access them:

```cpp
class Dog : public Animal
{
};
```

Then:

```cpp
Dog::count++;
```

The static member belongs to the base class unless separately declared in the derived class.

---

# 36. `protected` Members

`protected` is accessible inside the base and derived classes.

```cpp
class Animal
{
protected:

    int age;
};

class Dog : public Animal
{
public:

    void setAge(int value)
    {
        age = value;
    }
};
```

However, outside code cannot directly access it:

```cpp
Dog dog;

// dog.age = 10; // ERROR
```

### Design note

Don't automatically make everything `protected`.

Often this is preferable:

```cpp
private:
    int age;

public:
    void setAge(int);
    int getAge() const;
```

Encapsulation is usually easier to maintain with private data.

---

# 37. Inheritance and Friend Functions

Friendship is not automatically inherited.

Example:

```cpp
class Base
{
private:

    int value;

    friend void show(Base&);
};
```

The friend function can access `Base::value`.

But friendship does not automatically extend to derived classes.

---

# 38. Inheritance and Templates

Inheritance can be combined with templates.

```cpp
template<typename T>
class Base
{
public:

    void show()
    {
        std::cout << "Base\n";
    }
};

class Derived : public Base<int>
{
};
```

Another example:

```cpp
template<typename T>
class Animal
{
public:

    T value;
};

class Dog : public Animal<int>
{
};
```

Now:

```cpp
Dog dog;

dog.value = 10;
```

---

# 39. Inheritance vs Composition

This is one of the most important design decisions in C++.

Inheritance:

```cpp
class Dog : public Animal
{
};
```

Composition:

```cpp
class Engine
{
};

class Car
{
private:

    Engine engine;
};
```

Inheritance means:

```text
IS-A
```

Composition means:

```text
HAS-A
```

---

# 40. "IS-A" vs "HAS-A"

### Inheritance

```text
Dog IS-A Animal
```

Therefore:

```cpp
class Dog : public Animal
{
};
```

### Composition

```text
Car HAS-A Engine
```

Therefore:

```cpp
class Car
{
    Engine engine;
};
```

A useful question is:

> "Can I honestly say the derived class IS a base class?"

If not, inheritance may be the wrong design.

---

# 41. Upcasting

Converting:

```text
Derived → Base
```

is called upcasting.

Example:

```cpp
Dog dog;

Animal* animal = &dog;
```

This is generally safe with public inheritance.

With polymorphism:

```cpp
animal->speak();
```

the derived implementation can execute if `speak()` is virtual.

---

# 42. Downcasting

Downcasting converts:

```text
Base → Derived
```

Example:

```cpp
Animal* animal = new Dog();

Dog* dog = dynamic_cast<Dog*>(animal);
```

Downcasting should be used carefully.

A base pointer does not necessarily point to the derived type you expect.

---

# 43. `dynamic_cast`

`dynamic_cast` performs runtime-checked casting in polymorphic class hierarchies.

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

Now:

```cpp
Animal* animal = new Dog();

Dog* dog = dynamic_cast<Dog*>(animal);
```

The cast succeeds.

But:

```cpp
Cat* cat = dynamic_cast<Cat*>(animal);
```

returns:

```cpp
nullptr
```

because the object is actually a `Dog`.

---

# 44. `static_cast` and Downcasting

You can also use:

```cpp
Dog* dog = static_cast<Dog*>(animal);
```

But `static_cast` does **not** perform the same runtime type check as `dynamic_cast`.

If the object is not actually a compatible `Dog`, using the resulting pointer can cause undefined behavior.

Therefore:

```cpp
dynamic_cast
```

is useful when runtime checking is required.

---

# 45. CRTP

CRTP stands for:

> **Curiously Recurring Template Pattern**

The derived class passes itself as a template argument to the base.

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

class Derived : public Base<Derived>
{
public:

    void implementation()
    {
        std::cout << "Derived implementation\n";
    }
};
```

Usage:

```cpp
Derived obj;

obj.interface();
```

CRTP is commonly used for:

* Static polymorphism
* Generic programming
* Compile-time interfaces
* Avoiding some runtime virtual dispatch

It is an advanced C++ technique and should not be learned before the fundamentals.

---

# 46. Multiple Inheritance Best Practices

Multiple inheritance is not automatically bad, but it requires careful design.

A common reasonable use is combining independent interfaces:

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

This can be clearer than creating a deep inheritance hierarchy.

---

# 47. Common Mistakes

## Mistake 1 — Forgetting `virtual`

Bad:

```cpp
class Animal
{
public:
    void speak();
};
```

when runtime polymorphism is required.

Better:

```cpp
virtual void speak();
```

---

## Mistake 2 — Forgetting `override`

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

Bad:

```cpp
class Base
{
public:
    ~Base() {}
};
```

when deleting derived objects through base pointers.

Prefer:

```cpp
virtual ~Base() = default;
```

---

## Mistake 4 — Excessive inheritance

Don't create inheritance simply to reuse a few functions.

Consider composition instead.

---

## Mistake 5 — Deep inheritance hierarchies

Avoid unnecessarily complex structures such as:

```text
A
 ↓
B
 ↓
C
 ↓
D
 ↓
E
 ↓
F
```

Deep hierarchies can become difficult to maintain.

---

## Mistake 6 — Object slicing

Avoid:

```cpp
Base b = derived;
```

when polymorphic behavior is expected.

Prefer:

```cpp
Base& b = derived;
```

or:

```cpp
Base* b = &derived;
```

---

# 48. Best Practices

### 1. Prefer `override`

```cpp
void update() override;
```

### 2. Use virtual destructors for polymorphic bases

```cpp
virtual ~Base() = default;
```

### 3. Prefer public inheritance for genuine IS-A relationships

```cpp
class Dog : public Animal
```

### 4. Don't expose data unnecessarily

Prefer:

```cpp
private:
    int value;
```

with controlled access.

### 5. Prefer composition when the relationship is HAS-A

```cpp
class Car
{
    Engine engine;
};
```

### 6. Avoid unnecessary inheritance

Inheritance creates coupling between classes.

### 7. Keep inheritance hierarchies understandable

A good hierarchy should communicate a clear conceptual relationship.

---

# 49. Real-World Example

Consider a game.

```text
                 Character
                     │
            ┌────────┴────────┐
            │                 │
         Player            Enemy
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  Zombie              Boss
```

Base:

```cpp
class Character
{
public:

    virtual void attack() = 0;

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
};
```

Boss:

```cpp
class Boss : public Character
{
public:

    void attack() override
    {
        std::cout << "Boss attacks\n";
    }
};
```

Now:

```cpp
std::vector<std::unique_ptr<Character>> characters;

characters.push_back(std::make_unique<Player>());
characters.push_back(std::make_unique<Zombie>());
characters.push_back(std::make_unique<Boss>());
```

Then:

```cpp
for (auto& character : characters)
{
    character->attack();
}
```

Output:

```text
Player attacks
Zombie attacks
Boss attacks
```

The code doesn't need to know the exact derived type.

This is one of the major benefits of polymorphism.

---

# 50. Inheritance in Large C++ Projects

In a large project, inheritance is usually combined with:

```text
Interfaces
        +
Composition
        +
Polymorphism
        +
Dependency management
```

For example:

```text
Entity
  │
  ├── Player
  ├── Enemy
  └── NPC
```

But each entity may also contain components:

```text
Player
 ├── MovementComponent
 ├── HealthComponent
 ├── WeaponComponent
 └── InventoryComponent
```

This demonstrates an important design principle:

> Inheritance and composition are not competitors that must always be used separately.

A large C++ architecture may use both.

---

# 51. Inheritance and Unreal Engine

Inheritance becomes especially important when working with Unreal Engine.

A simplified Unreal-style hierarchy can look like:

```text
UObject
   │
   ├── Actor
   │     │
   │     ├── Pawn
   │     │    │
   │     │    └── Character
   │     │
   │     └── Other Actors
   │
   └── Other UObject-based classes
```

For example, Unreal gameplay classes commonly derive from existing engine classes.

A simplified example:

```cpp
class MyCharacter : public ACharacter
{
    GENERATED_BODY()

public:

    void Jump();
};
```

Here:

```text
ACharacter
     ↑
     │
MyCharacter
```

Your class receives behavior and infrastructure from the Unreal base class while adding your own gameplay logic.

Unreal also makes extensive use of:

* Inheritance
* Virtual functions
* Polymorphism
* Components
* Interfaces
* Composition

This is why understanding normal C++ inheritance before learning Unreal C++ is valuable.

---

# 52. Learning Checklist

Use this checklist to verify your understanding.

## Beginner

* [ ] What is inheritance?
* [ ] What is a base class?
* [ ] What is a derived class?
* [ ] Basic inheritance syntax
* [ ] `public` inheritance
* [ ] `protected` members
* [ ] `private` members
* [ ] Constructor order
* [ ] Destructor order
* [ ] Base constructor initialization
* [ ] Calling base functions

## Intermediate

* [ ] Function overriding
* [ ] `virtual`
* [ ] `override`
* [ ] `final`
* [ ] Runtime polymorphism
* [ ] Base pointers
* [ ] Base references
* [ ] Virtual destructors
* [ ] Pure virtual functions
* [ ] Abstract classes
* [ ] Interfaces
* [ ] Multilevel inheritance
* [ ] Hierarchical inheritance
* [ ] Multiple inheritance

## Advanced

* [ ] Diamond problem
* [ ] Virtual inheritance
* [ ] Object slicing
* [ ] Name hiding
* [ ] `using Base::function`
* [ ] Upcasting
* [ ] Downcasting
* [ ] `dynamic_cast`
* [ ] `static_cast`
* [ ] Multiple interface inheritance
* [ ] CRTP
* [ ] Inheritance vs composition
* [ ] Polymorphic ownership
* [ ] Designing maintainable class hierarchies

---

# Final Mental Model

When thinking about inheritance, remember this progression:

```text
                    INHERITANCE
                         │
                         ▼
                 Base / Derived
                         │
                         ▼
                Access Control
                         │
                         ▼
                  Constructors
                         │
                         ▼
                    Overriding
                         │
                         ▼
                     virtual
                         │
                         ▼
                   Polymorphism
                         │
                         ▼
                Abstract Classes
                         │
                         ▼
                    Interfaces
                         │
                         ▼
             Multiple Inheritance
                         │
                         ▼
                 Diamond Problem
                         │
                         ▼
                Virtual Inheritance
                         │
                         ▼
                Advanced Casting
                         │
                         ▼
              Design: IS-A vs HAS-A
                         │
                         ▼
              Composition + Inheritance
                         │
                         ▼
                Large C++ Systems
                         │
                         ▼
                  Unreal Engine
```

## The Most Important Rules

```text
1. Inheritance represents an IS-A relationship.

2. Composition represents a HAS-A relationship.

3. Use public inheritance for normal polymorphic IS-A relationships.

4. Use virtual functions when runtime polymorphism is required.

5. Use override when overriding virtual functions.

6. Give polymorphic base classes virtual destructors.

7. Avoid object slicing when polymorphism is required.

8. Don't use inheritance just for code reuse.

9. Prefer composition when the relationship is HAS-A.

10. Keep advanced inheritance techniques for situations where
    they actually solve a design problem.
```

---

## Recommended Practice Projects

After learning inheritance, build these in order:

### Project 1 — Animal System

```text
Animal
├── Dog
├── Cat
└── Bird
```

Implement:

* Constructors
* Inheritance
* Virtual functions
* Overriding

### Project 2 — Vehicle System

```text
Vehicle
├── Car
├── Bike
└── Truck
```

Implement:

* Abstract base class
* Pure virtual functions
* Runtime polymorphism

### Project 3 — Employee System

```text
Employee
├── Developer
├── Manager
└── Designer
```

Use:

```cpp
std::vector<std::unique_ptr<Employee>>
```

to practice polymorphic ownership.

### Project 4 — Game Character System

```text
Character
├── Player
├── Enemy
│   ├── Zombie
│   └── Boss
└── NPC
```

Practice:

* Abstract classes
* Virtual functions
* Interfaces
* Composition
* Smart pointers
* Polymorphism

### Project 5 — Multiplayer Entity System

```text
NetworkEntity
├── Player
├── Enemy
├── Projectile
└── Vehicle
```

This bridges your C++ inheritance knowledge toward your next roadmap stage:

```text
C++
 ↓
OOP
 ↓
Inheritance
 ↓
Polymorphism
 ↓
Memory / RAII
 ↓
Multithreading
 ↓
Networking
 ↓
Unreal Engine
```

---

## Final Goal

Don't measure your inheritance knowledge by how many keywords you remember.

You should eventually be able to look at:

```cpp
class Player : public Character
{
public:
    void attack() override;
};
```

and explain:

* Why `Player` derives from `Character`
* What `public` inheritance means
* What `override` means
* Why `attack()` may be virtual
* What happens when a `Character*` points to a `Player`
* Why the base destructor should be virtual
* When composition would be better than inheritance
* How this design can scale to a larger C++ or Unreal Engine project

That is the level of understanding you should aim for before moving deeper into **C++ networking and Unreal Engine**.
