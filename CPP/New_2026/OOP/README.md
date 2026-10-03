# C++ Classes

A **class** in C++ is a user-defined data type that combines **data (attributes)** and **functions (methods)** into a single unit.

A class is one of the fundamental concepts of **Object-Oriented Programming (OOP)**.

---

## 1. Basic Class

```cpp
class Hero {
public:
    int health;
    char level;

    void display() {
        std::cout << health << " " << level << std::endl;
    }
};
```

Create an object:

```cpp
Hero hero;

hero.health = 100;
hero.level = 'A';

hero.display();
```

### Class vs Object

* **Class** → Blueprint/template
* **Object** → Actual instance created from the class

```text
Class
  ↓
Hero
  ↓
┌──────────────┐
│ health       │
│ level        │
│ display()    │
└──────────────┘
      ↓
   Objects
   ├── hero1
   ├── hero2
   └── hero3
```

---

# 2. Access Specifiers

C++ provides three main access specifiers:

```cpp
class Hero {
private:
    int health;

protected:
    char level;

public:
    void display();
};
```

### `private`

Accessible only inside the class.

```cpp
class Hero {
private:
    int health;
};
```

### `public`

Accessible from outside the class.

```cpp
Hero hero;
hero.display();
```

### `protected`

Accessible inside the class and derived classes.

```text
private
   ↓
Class only

protected
   ↓
Class + Derived classes

public
   ↓
Anywhere through the object
```

---

# 3. Encapsulation

**Encapsulation** means keeping data and the functions that operate on that data together while controlling access to the data.

Instead of:

```cpp
class Hero {
public:
    int health;
};
```

we normally use:

```cpp
class Hero {
private:
    int health;

public:
    void setHealth(int h) {
        health = h;
    }

    int getHealth() {
        return health;
    }
};
```

Usage:

```cpp
Hero hero;

hero.setHealth(100);

std::cout << hero.getHealth();
```

This prevents direct modification:

```cpp
hero.health = -500; // ❌
```

---

# 4. Constructor

A **constructor** is a special member function that is automatically called when an object is created.

Characteristics:

* Same name as the class
* No return type
* Called automatically
* Used to initialize objects

```cpp
class Hero {
private:
    int health;

public:
    Hero() {
        health = 100;
    }
};
```

Usage:

```cpp
Hero hero;
```

The constructor automatically runs.

---

# 5. Default Constructor

A constructor that takes no arguments is called a **default constructor**.

```cpp
class Hero {
public:
    Hero() {
        std::cout << "Default Constructor\n";
    }
};
```

```cpp
Hero hero;
```

Output:

```text
Default Constructor
```

---

# 6. Parameterized Constructor

A constructor that accepts parameters is called a **parameterized constructor**.

```cpp
class Hero {
private:
    int health;
    char level;

public:
    Hero(int h, char l) {
        health = h;
        level = l;
    }
};
```

Usage:

```cpp
Hero hero(100, 'A');
```

---

# 7. Constructor Initialization List

An **initialization list** initializes class members before the constructor body executes.

```cpp
class Hero {
private:
    int health;
    char level;

public:
    Hero(int h, char l)
        : health(h), level(l)
    {
    }
};
```

Instead of:

```cpp
Hero(int h, char l) {
    health = h;
    level = l;
}
```

we can directly initialize:

```cpp
Hero(int h, char l)
    : health(h), level(l)
{
}
```

### Why use initialization lists?

They are:

* Required for `const` members
* Required for reference members
* Required for members without a default constructor
* Useful for direct initialization
* Often more efficient than assignment inside the constructor

---

# 8. `const` Member Initialization

A `const` data member must be initialized when the object is created.

```cpp
class Hero {
private:
    const int health;

public:
    Hero(int h)
        : health(h)
    {
    }
};
```

This is invalid:

```cpp
Hero(int h) {
    health = h; // ❌
}
```

because `health` is `const`.

---

# 9. Reference Member Initialization

Reference members also need initialization.

```cpp
class Hero {
private:
    int& health;

public:
    Hero(int& h)
        : health(h)
    {
    }
};
```

A reference cannot simply be assigned later inside the constructor body.

---

# 10. Initialization Order

Members are initialized according to their **declaration order**, not the order in the initialization list.

```cpp
class Hero {
private:
    int health;
    int level;

public:
    Hero(int h, int l)
        : level(l), health(h)
    {
    }
};
```

Even though `level` appears first in the initialization list:

```cpp
: level(l), health(h)
```

the actual order is:

```text
health
  ↓
level
```

because they were declared in that order.

### Best practice

Keep the initialization list in the same order as member declarations:

```cpp
Hero(int h, int l)
    : health(h), level(l)
{
}
```

---

# 11. `this` Pointer

`this` is a pointer that refers to the **current object**.

Consider:

```cpp
class Hero {
private:
    int health;

public:
    Hero(int health) {
        this->health = health;
    }
};
```

Here there are two `health` variables:

```cpp
Hero(int health)
```

The parameter is:

```text
health
```

The class member is:

```text
this->health
```

Therefore:

```cpp
this->health = health;
```

means:

```text
current object's health = constructor parameter health
```

---

# 12. Why Use `this`?

It is commonly used when a parameter has the same name as a class member.

```cpp
class Hero {
private:
    int health;

public:
    void setHealth(int health) {
        this->health = health;
    }
};
```

Without `this`, the names can become ambiguous.

---

# 13. Object Address and `this`

Because `this` points to the current object:

```cpp
class Hero {
public:
    void showAddress() {
        std::cout << this << std::endl;
    }
};
```

Usage:

```cpp
Hero hero;

std::cout << &hero << std::endl;
hero.showAddress();
```

Both addresses refer to the same object.

---

# 14. Member Functions

A function defined inside a class is called a **member function**.

```cpp
class Hero {
private:
    int health;

public:
    int getHealth() {
        return health;
    }

    void setHealth(int h) {
        health = h;
    }
};
```

Usage:

```cpp
Hero hero;

hero.setHealth(100);

std::cout << hero.getHealth();
```

---

# 15. Defining Member Functions Outside the Class

You can declare a function inside the class and define it outside.

```cpp
class Hero {
private:
    int health;

public:
    int getHealth();
};
```

Definition:

```cpp
int Hero::getHealth() {
    return health;
}
```

The `::` operator is called the **scope resolution operator**.

---

# 16. `const` Member Function

A `const` member function promises not to modify the object's non-mutable data members.

```cpp
class Hero {
private:
    int health;

public:
    int getHealth() const {
        return health;
    }
};
```

A getter is often declared `const`:

```cpp
int getHealth() const;
```

This allows it to be called on a `const` object:

```cpp
const Hero hero;

hero.getHealth();
```

---

# 17. Static Data Members

A `static` data member belongs to the **class**, not to each individual object.

```cpp
class Hero {
private:
    static int count;
};
```

Traditionally, define it outside the class:

```cpp
int Hero::count = 0;
```

All objects share the same `count`.

```text
Hero Class
     │
     └── static count
          ↑
     ┌────┼────┐
     │    │    │
  hero1 hero2 hero3
```

---

# 18. Static Member Function

A `static` member function belongs to the class and can be called without creating an object.

```cpp
class Hero {
private:
    inline static int timeToPlay = 100;

public:
    static int getRemainTime() {
        return timeToPlay;
    }
};
```

Call it using:

```cpp
std::cout << Hero::getRemainTime();
```

Notice:

```cpp
Hero::getRemainTime();
```

No object is required.

### Important

A static member function does not have a `this` pointer because it is not associated with a particular object.

Therefore, it cannot directly access non-static members.

---

# 19. Copy Constructor

A copy constructor creates a new object from an existing object.

```cpp
class Hero {
public:
    Hero() {
    }
};
```

Then:

```cpp
Hero hero1;

Hero hero2(hero1);
```

`hero2` is created as a copy of `hero1`.

You can explicitly define a copy constructor:

```cpp
class Hero {
private:
    int health;

public:
    Hero(int h)
        : health(h)
    {
    }

    Hero(const Hero& other)
        : health(other.health)
    {
    }
};
```

Usage:

```cpp
Hero hero1(100);
Hero hero2(hero1);
```

---

# 20. Destructor

A destructor is automatically called when an object is destroyed.

Syntax:

```cpp
~Hero()
```

Example:

```cpp
class Hero {
public:
    ~Hero() {
        std::cout << "Destructor called\n";
    }
};
```

For a local object:

```cpp
{
    Hero hero;
}
```

When the block ends:

```text
Hero created
     ↓
...
     ↓
Block ends
     ↓
Destructor called
```

---

# 21. Stack vs Heap Objects

### Stack object

```cpp
Hero hero;
```

The object is automatically destroyed when it goes out of scope.

### Dynamic/heap object

```cpp
Hero* hero = new Hero();
```

With manual memory management:

```cpp
delete hero;
```

Modern C++ generally prefers RAII and smart pointers over manually managing `new`/`delete`.

---

# 22. Smart Pointer

Instead of:

```cpp
Hero* hero = new Hero();
delete hero;
```

modern C++ can use:

```cpp
#include <memory>

std::unique_ptr<Hero> hero = std::make_unique<Hero>();
```

The object is automatically destroyed when the `unique_ptr` goes out of scope.

---

# 23. `struct` vs `class`

Both can contain:

* Variables
* Functions
* Constructors
* Destructors
* Static members
* Inheritance

The main default-access difference is:

```cpp
struct Hero {
    int health;  // public by default
};
```

while:

```cpp
class Hero {
    int health;  // private by default
};
```

Similarly:

* `struct` → `public` by default
* `class` → `private` by default

---

# 24. Object Creation

There are several ways to create objects.

### Normal object

```cpp
Hero hero;
```

### Parameterized object

```cpp
Hero hero(100, 'A');
```

### Uniform initialization

```cpp
Hero hero{100, 'A'};
```

### Dynamic object

```cpp
Hero* hero = new Hero(100, 'A');
```

### Smart pointer

```cpp
auto hero = std::make_unique<Hero>(100, 'A');
```

---

# 25. Complete Example

Putting the concepts together:

```cpp
#include <iostream>

class Hero {

private:
    const int id;
    int health;
    char level;

    inline static int totalHeroes = 0;

public:

    // Default constructor
    Hero()
        : id(0), health(100), level('C')
    {
        totalHeroes++;
    }

    // Parameterized constructor
    Hero(int id, int health, char level)
        : id(id), health(health), level(level)
    {
        totalHeroes++;
    }

    // Copy constructor
    Hero(const Hero& other)
        : id(other.id),
          health(other.health),
          level(other.level)
    {
        totalHeroes++;
    }

    // Getter
    int getHealth() const {
        return health;
    }

    // Setter
    void setHealth(int health) {
        this->health = health;
    }

    // Static function
    static int getTotalHeroes() {
        return totalHeroes;
    }

    // Destructor
    ~Hero() {
        std::cout << "Hero destroyed\n";
    }
};


int main() {

    Hero hero1;

    Hero hero2(101, 100, 'A');

    Hero hero3(hero2);

    hero1.setHealth(80);

    std::cout << "Hero 1 Health: "
              << hero1.getHealth()
              << '\n';

    std::cout << "Hero 2 Health: "
              << hero2.getHealth()
              << '\n';

    std::cout << "Total Heroes: "
              << Hero::getTotalHeroes()
              << '\n';

    return 0;
}
```

---

# Quick Summary

| Concept                   | Meaning                                   |
| ------------------------- | ----------------------------------------- |
| Class                     | Blueprint for objects                     |
| Object                    | Instance of a class                       |
| Encapsulation             | Bundle data + methods and control access  |
| `private`                 | Class-only access                         |
| `protected`               | Class + derived classes                   |
| `public`                  | Accessible from outside                   |
| Constructor               | Initializes an object                     |
| Default constructor       | Constructor with no parameters            |
| Parameterized constructor | Constructor with parameters               |
| Initialization list       | Directly initializes members              |
| `const` member            | Cannot be modified after initialization   |
| `this`                    | Pointer to current object                 |
| Member function           | Function belonging to a class             |
| Static member             | Shared by all objects                     |
| Static function           | Callable through the class                |
| Copy constructor          | Creates object from another object        |
| Destructor                | Runs when object is destroyed             |
| RAII                      | Resource lifetime tied to object lifetime |
| `unique_ptr`              | Automatic ownership of dynamic object     |

### Core mental model

```text
                    CLASS
                      │
          ┌───────────┴───────────┐
          │                       │
        Data                  Functions
          │                       │
      attributes              methods
          │                       │
          └───────────┬───────────┘
                      │
                    Object
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   Constructor      Methods      Destructor
        │
        ├── Default
        ├── Parameterized
        ├── Copy
        └── Initialization List
                      │
                 `this` pointer
                      │
                Static Members
```

**Recommended learning order:**
`Class → Object → Access Modifiers → Encapsulation → Constructor → Initialization List → this → const → Static → Copy Constructor → Destructor → RAII → Smart Pointers → Inheritance → Polymorphism → Abstraction`
