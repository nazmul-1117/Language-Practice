# C++ Abstraction — Zero to Advanced

Abstraction is one of the four fundamental concepts of **Object-Oriented Programming (OOP)**:

1. **Encapsulation**
2. **Abstraction**
3. **Inheritance**
4. **Polymorphism**

Abstraction is about **hiding unnecessary implementation details and exposing only the essential functionality**.

---

# 1. What is Abstraction?

**Abstraction** means showing only the important information to the user while hiding the internal implementation details.

### Simple Example

When you drive a car, you use:

* Steering wheel
* Accelerator
* Brake
* Gear

You do not need to know:

* How the engine internally works
* How fuel injection works
* How the transmission works
* How the braking mechanism internally operates

You only interact with the **essential interface**.

This is abstraction.

```text
User
  |
  v
+----------------------+
|     Car Interface    |
|                      |
|  start()             |
|  accelerate()        |
|  brake()             |
+----------------------+
          |
          v
+----------------------+
| Internal Engine      |
| Transmission         |
| Fuel System          |
| Braking System       |
+----------------------+
```

The user interacts with the public interface without needing to understand the implementation.

---

# 2. Why Do We Need Abstraction?

Without abstraction, users of a class may need to understand its internal implementation.

Abstraction provides:

* Simplicity
* Security
* Maintainability
* Modularity
* Loose coupling
* Easier code management
* Better software architecture

### Without abstraction

```cpp
class CoffeeMachine {
public:
    void heatWater();
    void grindBeans();
    void pumpWater();
    void mixCoffee();
};
```

The user has to understand all these operations.

### With abstraction

```cpp
class CoffeeMachine {
public:
    void makeCoffee();
};
```

The user only needs:

```cpp
machine.makeCoffee();
```

The internal operations are hidden.

---

# 3. Abstraction in C++

C++ provides abstraction mainly through:

1. **Access specifiers**
2. **Classes**
3. **Encapsulation**
4. **Abstract classes**
5. **Pure virtual functions**
6. **Interfaces**
7. **Inheritance and polymorphism**

The two most important techniques are:

```text
Concrete Class
      |
      v
Access Control
      |
      v
Hide implementation

Abstract Class
      |
      v
Pure Virtual Function
      |
      v
Define required behavior
```

---

# 4. Abstraction vs Encapsulation

These two concepts are closely related but they are not the same.

## Encapsulation

Encapsulation focuses on:

> **Wrapping data and methods together and controlling access to them.**

Example:

```cpp
class BankAccount {
private:
    double balance;

public:
    void deposit(double amount) {
        balance += amount;
    }

    double getBalance() {
        return balance;
    }
};
```

Here, `balance` is hidden using `private`.

---

## Abstraction

Abstraction focuses on:

> **Hiding implementation complexity and exposing only essential functionality.**

Example:

```cpp
class Payment {
public:
    void pay();
};
```

The user does not need to know how payment processing works internally.

---

## Difference

| Encapsulation                         | Abstraction                             |
| ------------------------------------- | --------------------------------------- |
| Controls access to data               | Hides implementation complexity         |
| Focuses on data protection            | Focuses on essential behavior           |
| Uses `private`, `protected`, `public` | Uses abstract classes, interfaces, APIs |
| Implementation + data are bundled     | Unnecessary implementation is hidden    |
| Answers "How do I protect this?"      | Answers "What should the user see?"     |

### Easy way to remember

```text
Encapsulation → Protect the data
Abstraction   → Hide the complexity
```

---

# 5. Abstraction Using Access Specifiers

C++ provides three major access specifiers:

```cpp
public
private
protected
```

---

## Public

Public members can be accessed from outside the class.

```cpp
class Car {
public:
    void start() {
        std::cout << "Car started";
    }
};
```

Usage:

```cpp
Car car;
car.start();
```

---

## Private

Private members cannot be directly accessed from outside the class.

```cpp
class BankAccount {
private:
    double balance;
};
```

This helps hide internal implementation.

---

## Protected

Protected members are accessible inside the class and derived classes.

```cpp
class Parent {
protected:
    int value;
};
```

A derived class can access `value`.

---

# 6. Basic Abstraction Example

```cpp
#include <iostream>

class Car {
private:
    void startEngine() {
        std::cout << "Engine started\n";
    }

    void checkFuel() {
        std::cout << "Fuel checked\n";
    }

public:
    void start() {
        checkFuel();
        startEngine();

        std::cout << "Car started\n";
    }
};

int main() {

    Car car;

    car.start();

    return 0;
}
```

Output:

```text
Fuel checked
Engine started
Car started
```

The user only calls:

```cpp
car.start();
```

The user does not directly interact with:

```cpp
startEngine();
checkFuel();
```

This is abstraction.

---

# 7. Real-World Example

Consider an ATM.

The user sees:

```text
ATM
 |
 +-- Insert Card
 +-- Enter PIN
 +-- Withdraw
 +-- Deposit
 +-- Check Balance
```

The user does not see:

```text
Bank Server
Database
Authentication
Encryption
Transaction Processing
Network Communication
```

The ATM provides a simple interface while hiding complex implementation.

In C++:

```cpp
class ATM {
public:
    void withdraw(double amount) {
        authenticate();
        checkBalance(amount);
        processTransaction(amount);
    }

private:
    void authenticate() {
        // Authentication logic
    }

    void checkBalance(double amount) {
        // Balance verification
    }

    void processTransaction(double amount) {
        // Transaction processing
    }
};
```

User:

```cpp
ATM atm;

atm.withdraw(5000);
```

The internal process remains hidden.

---

# 8. Abstract Classes

An **abstract class** is a class that cannot be instantiated directly.

It is generally used as a blueprint for derived classes.

An abstract class contains at least one **pure virtual function**.

Example:

```cpp
class Animal {
public:
    virtual void sound() = 0;
};
```

Because `sound()` is pure virtual, `Animal` is abstract.

This is invalid:

```cpp
Animal animal;
```

---

# 9. Pure Virtual Function

A pure virtual function is declared using:

```cpp
= 0;
```

Example:

```cpp
class Animal {
public:
    virtual void sound() = 0;
};
```

The syntax is:

```cpp
virtual returnType functionName() = 0;
```

Example:

```cpp
virtual void display() = 0;
```

---

# 10. Why Use Pure Virtual Functions?

A pure virtual function tells derived classes:

> "You must provide your own implementation of this function."

Example:

```cpp
class Animal {
public:
    virtual void sound() = 0;
};
```

Now every derived class must implement `sound()`.

```cpp
class Dog : public Animal {
public:
    void sound() override {
        std::cout << "Dog barks\n";
    }
};
```

```cpp
class Cat : public Animal {
public:
    void sound() override {
        std::cout << "Cat meows\n";
    }
};
```

---

# 11. Complete Abstract Class Example

```cpp
#include <iostream>

class Animal {
public:
    virtual void sound() = 0;

    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    void sound() override {
        std::cout << "Dog barks\n";
    }
};

class Cat : public Animal {
public:
    void sound() override {
        std::cout << "Cat meows\n";
    }
};

int main() {

    Dog dog;
    Cat cat;

    dog.sound();
    cat.sound();

    return 0;
}
```

Output:

```text
Dog barks
Cat meows
```

---

# 12. Abstract Class as a Blueprint

Think of an abstract class as a blueprint.

```text
              Animal
                |
       +--------+--------+
       |                 |
      Dog               Cat
       |                 |
   sound()            sound()
   = Bark              = Meow
```

The base class defines:

```cpp
sound()
```

But each derived class determines how it behaves.

---

# 13. Abstraction + Polymorphism

Abstraction and polymorphism often work together.

Example:

```cpp
class Shape {
public:
    virtual double area() = 0;

    virtual ~Shape() = default;
};
```

Derived classes:

```cpp
class Circle : public Shape {
public:
    double area() override {
        return 3.1416 * 5 * 5;
    }
};
```

```cpp
class Rectangle : public Shape {
public:
    double area() override {
        return 10 * 20;
    }
};
```

Now:

```cpp
Shape* shape1 = new Circle();
Shape* shape2 = new Rectangle();

std::cout << shape1->area() << '\n';
std::cout << shape2->area() << '\n';

delete shape1;
delete shape2;
```

The caller knows:

```cpp
Shape
    |
    +-- area()
```

but does not need to know the implementation details of each shape.

This is:

```text
Abstraction
     +
Polymorphism
     =
Flexible Architecture
```

---

# 14. Interface in C++

C++ does not have a separate `interface` keyword like some other languages.

Instead, interfaces are usually created using classes containing pure virtual functions.

Example:

```cpp
class PaymentGateway {
public:
    virtual void pay(double amount) = 0;
    virtual void refund(double amount) = 0;

    virtual ~PaymentGateway() = default;
};
```

Now different payment systems can implement the interface.

```cpp
class Bkash : public PaymentGateway {
public:
    void pay(double amount) override {
        std::cout << "Payment using bKash\n";
    }

    void refund(double amount) override {
        std::cout << "Refund using bKash\n";
    }
};
```

Another implementation:

```cpp
class CardPayment : public PaymentGateway {
public:
    void pay(double amount) override {
        std::cout << "Payment using Card\n";
    }

    void refund(double amount) override {
        std::cout << "Refund using Card\n";
    }
};
```

---

# 15. Real-World Payment Abstraction

A payment application may support:

```text
PaymentGateway
       |
       +----------+
       |          |
     bKash       Card
       |          |
     Pay()      Pay()
     Refund()   Refund()
```

The application can work with:

```cpp
PaymentGateway*
```

instead of knowing the specific payment provider.

Example:

```cpp
void processPayment(PaymentGateway& gateway) {
    gateway.pay(1000);
}
```

Then:

```cpp
Bkash bkash;

processPayment(bkash);
```

The function only cares about the abstraction:

```cpp
PaymentGateway
```

not the implementation.

---

# 16. Abstract Class with Constructor

An abstract class can have a constructor.

Example:

```cpp
class Animal {
protected:
    std::string name;

public:
    Animal(std::string n) : name(n) {}

    virtual void sound() = 0;
};
```

Derived class:

```cpp
class Dog : public Animal {
public:
    Dog(std::string n) : Animal(n) {}

    void sound() override {
        std::cout << name << " barks\n";
    }
};
```

Usage:

```cpp
Dog dog("Tommy");

dog.sound();
```

Output:

```text
Tommy barks
```

---

# 17. Abstract Class Can Have Normal Functions

An abstract class does not need to contain only pure virtual functions.

It can contain:

* Normal functions
* Pure virtual functions
* Member variables
* Constructors
* Destructors

Example:

```cpp
class Animal {
protected:
    std::string name;

public:

    Animal(std::string n) : name(n) {}

    void eat() {
        std::cout << name << " is eating\n";
    }

    virtual void sound() = 0;

    virtual ~Animal() = default;
};
```

Derived class:

```cpp
class Dog : public Animal {
public:

    Dog(std::string n) : Animal(n) {}

    void sound() override {
        std::cout << name << " barks\n";
    }
};
```

---

# 18. Pure Virtual Function Can Have an Implementation

A less commonly used C++ feature is that a pure virtual function can still have a definition.

Example:

```cpp
class Animal {
public:
    virtual void sound() = 0;
};

void Animal::sound() {
    std::cout << "Animal sound\n";
}
```

However, derived classes still need to provide their own override if they are to become concrete.

This feature is useful in certain advanced designs but should not be confused with making the function non-pure.

---

# 19. Virtual Destructor in Abstract Classes

When an abstract class is used polymorphically, its destructor should generally be virtual.

Example:

```cpp
class Animal {
public:
    virtual void sound() = 0;

    virtual ~Animal() = default;
};
```

This ensures proper destruction through a base pointer.

Example:

```cpp
Animal* animal = new Dog();

delete animal;
```

A virtual destructor allows the derived object's destructor to be called correctly.

---

# 20. Abstract Class with Smart Pointers

Modern C++ should generally prefer smart pointers over manual `new` and `delete`.

Example:

```cpp
#include <iostream>
#include <memory>
#include <vector>

class Animal {
public:
    virtual void sound() = 0;

    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    void sound() override {
        std::cout << "Dog\n";
    }
};

class Cat : public Animal {
public:
    void sound() override {
        std::cout << "Cat\n";
    }
};

int main() {

    std::vector<std::unique_ptr<Animal>> animals;

    animals.push_back(std::make_unique<Dog>());
    animals.push_back(std::make_unique<Cat>());

    for (const auto& animal : animals) {
        animal->sound();
    }

    return 0;
}
```

Output:

```text
Dog
Cat
```

This combines:

* Abstraction
* Inheritance
* Polymorphism
* Smart pointers

---

# 21. Abstraction Through Functions

Abstraction does not always require inheritance.

A function can also provide abstraction.

Example:

```cpp
double calculateTax(double salary) {
    // Complex tax calculation
    return salary * 0.10;
}
```

The caller only needs:

```cpp
double tax = calculateTax(50000);
```

The caller does not need to know the internal calculation.

---

# 22. Abstraction Through APIs

Software APIs are a practical example of abstraction.

Suppose you use:

```cpp
database.connect();
```

You don't need to understand:

```text
Socket creation
Authentication
Packet formation
Encryption
Network communication
Connection management
```

The API provides a simplified interface.

```text
Application
     |
     v
 database.connect()
     |
     v
+---------------------+
| Hidden Implementation|
|                     |
| Networking          |
| Authentication      |
| Protocol            |
| Connection          |
+---------------------+
```

---

# 23. Abstraction Through Libraries

When using a C++ library:

```cpp
std::vector<int> numbers;
numbers.push_back(10);
```

You don't need to know exactly how `std::vector` manages:

* Dynamic memory
* Capacity
* Reallocation
* Element construction
* Memory movement

The library exposes a simple interface.

```cpp
numbers.push_back(10);
```

This is abstraction.

---

# 24. Abstraction and Access Control

Consider:

```cpp
class BankAccount {

private:
    double balance;

public:

    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }

    double getBalance() {
        return balance;
    }
};
```

The user cannot directly do:

```cpp
account.balance = -100000;
```

Instead:

```cpp
account.deposit(5000);
```

The class controls how the operation happens.

---

# 25. Data Abstraction

Data abstraction means hiding unnecessary details about how data is stored or processed.

Example:

```cpp
class Student {
private:
    int marks;

public:

    void setMarks(int m) {
        if (m >= 0 && m <= 100) {
            marks = m;
        }
    }

    int getMarks() {
        return marks;
    }
};
```

The user does not need to know how `marks` is internally managed.

---

# 26. Procedural Code vs Abstraction

### Without abstraction

```cpp
float principal = 10000;
float rate = 5;
float time = 2;

float interest = principal * rate * time / 100;
```

The logic is exposed.

### With abstraction

```cpp
float calculateSimpleInterest(
    float principal,
    float rate,
    float time
) {
    return principal * rate * time / 100;
}
```

Usage:

```cpp
float interest = calculateSimpleInterest(10000, 5, 2);
```

The calculation is hidden behind a meaningful interface.

---

# 27. Abstraction in Large Software

Large applications often have layers.

```text
+-------------------------+
|       User Interface    |
+-------------------------+
            |
            v
+-------------------------+
|      Business Logic     |
+-------------------------+
            |
            v
+-------------------------+
|      Service Layer      |
+-------------------------+
            |
            v
+-------------------------+
|     Database Layer      |
+-------------------------+
            |
            v
+-------------------------+
|       Database          |
+-------------------------+
```

Each layer hides its internal implementation from other layers.

For example:

```cpp
userService.createUser();
```

The caller does not need to know:

```text
SQL query
Database connection
Validation
Transaction
Password hashing
```

---

# 28. Abstraction and Loose Coupling

Abstraction helps reduce dependency between components.

Bad design:

```cpp
class OrderService {
    MySQLDatabase database;
};
```

`OrderService` is tightly coupled to MySQL.

Better design:

```cpp
class Database {
public:
    virtual void save() = 0;

    virtual ~Database() = default;
};
```

Then:

```cpp
class MySQLDatabase : public Database {
public:
    void save() override {
        // MySQL implementation
    }
};
```

Now:

```cpp
class OrderService {
private:
    Database& database;

public:
    OrderService(Database& db)
        : database(db) {}

    void saveOrder() {
        database.save();
    }
};
```

This creates a more flexible architecture.

---

# 29. Dependency Inversion and Abstraction

Abstraction is strongly related to the **Dependency Inversion Principle (DIP)**.

Instead of:

```text
High-level module
       |
       v
Concrete implementation
```

we can design:

```text
High-level module
       |
       v
   Abstraction
       ^
       |
Concrete implementation
```

Example:

```cpp
class Notification {
public:
    virtual void send() = 0;

    virtual ~Notification() = default;
};
```

Implementations:

```cpp
class EmailNotification : public Notification {
public:
    void send() override {
        std::cout << "Email sent\n";
    }
};
```

```cpp
class SMSNotification : public Notification {
public:
    void send() override {
        std::cout << "SMS sent\n";
    }
};
```

Now the high-level application depends on:

```cpp
Notification
```

rather than directly depending on:

```cpp
EmailNotification
```

---

# 30. Abstraction with Dependency Injection

Example:

```cpp
class Logger {
public:
    virtual void log(const std::string& message) = 0;

    virtual ~Logger() = default;
};
```

Implementation:

```cpp
class ConsoleLogger : public Logger {
public:
    void log(const std::string& message) override {
        std::cout << message << '\n';
    }
};
```

Service:

```cpp
class UserService {
private:
    Logger& logger;

public:
    UserService(Logger& l)
        : logger(l) {}

    void createUser() {
        logger.log("User created");
    }
};
```

Usage:

```cpp
ConsoleLogger logger;

UserService service(logger);

service.createUser();
```

The service depends on the abstraction:

```cpp
Logger
```

not on a specific logger implementation.

---

# 31. Multiple Abstract Functions

An abstract class can contain multiple pure virtual functions.

```cpp
class Vehicle {
public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void accelerate() = 0;

    virtual ~Vehicle() = default;
};
```

Derived class:

```cpp
class Car : public Vehicle {
public:

    void start() override {
        std::cout << "Car started\n";
    }

    void stop() override {
        std::cout << "Car stopped\n";
    }

    void accelerate() override {
        std::cout << "Car accelerating\n";
    }
};
```

The abstract class defines the required behavior.

---

# 32. Abstract Class vs Concrete Class

## Abstract Class

```cpp
class Animal {
public:
    virtual void sound() = 0;
};
```

Cannot create:

```cpp
Animal a; // Error
```

---

## Concrete Class

```cpp
class Dog : public Animal {
public:
    void sound() override {
        std::cout << "Bark";
    }
};
```

Can create:

```cpp
Dog dog;
```

---

# 33. Abstract Class vs Interface

In C++, an interface is commonly represented by a class containing only pure virtual functions and a virtual destructor.

### Abstract class

```cpp
class Animal {
protected:
    std::string name;

public:
    void eat() {
        std::cout << "Eating\n";
    }

    virtual void sound() = 0;

    virtual ~Animal() = default;
};
```

It can contain:

* Variables
* Constructors
* Implemented methods
* Pure virtual methods

### Interface-like class

```cpp
class Printable {
public:
    virtual void print() = 0;

    virtual ~Printable() = default;
};
```

It primarily defines a contract.

---

# 34. Multiple Interfaces

A C++ class can implement multiple interface-like base classes.

```cpp
class Printable {
public:
    virtual void print() = 0;

    virtual ~Printable() = default;
};
```

```cpp
class Serializable {
public:
    virtual void serialize() = 0;

    virtual ~Serializable() = default;
};
```

Now:

```cpp
class User : public Printable, public Serializable {
public:

    void print() override {
        std::cout << "User information\n";
    }

    void serialize() override {
        std::cout << "User serialized\n";
    }
};
```

The class provides both behaviors.

---

# 35. Abstraction with Multiple Inheritance

```text
Printable       Serializable
     \             /
      \           /
           User
```

```cpp
class User
    : public Printable,
      public Serializable
{
    // implementation
};
```

This can be useful when a class needs to satisfy multiple independent contracts.

---

# 36. Abstraction and Polymorphism Difference

These concepts are related but different.

### Abstraction

Defines:

> **What should be available?**

Example:

```cpp
virtual void draw() = 0;
```

### Polymorphism

Determines:

> **Which implementation should execute?**

Example:

```cpp
Shape* shape = new Circle();

shape->draw();
```

Conceptually:

```text
Abstraction
    |
    v
What operation exists?
    |
    v
draw()

Polymorphism
    |
    v
Which draw() executes?
    |
    v
Circle::draw()
```

---

# 37. Abstraction + Inheritance + Polymorphism

These concepts frequently work together.

```cpp
class Shape {
public:
    virtual void draw() = 0;

    virtual ~Shape() = default;
};
```

Inheritance:

```cpp
class Circle : public Shape {
public:
    void draw() override {
        std::cout << "Drawing Circle\n";
    }
};
```

Polymorphism:

```cpp
Shape* shape = new Circle();

shape->draw();
```

Here:

* `Shape` provides abstraction.
* `Circle` inherits from `Shape`.
* `draw()` is overridden.
* The base pointer enables runtime polymorphism.

---

# 38. Abstraction in Game Development

A game engine might define:

```cpp
class GameObject {
public:
    virtual void update() = 0;
    virtual void render() = 0;

    virtual ~GameObject() = default;
};
```

Different objects implement their own behavior.

```cpp
class Player : public GameObject {
public:
    void update() override {
        // Player update
    }

    void render() override {
        // Player rendering
    }
};
```

```cpp
class Enemy : public GameObject {
public:
    void update() override {
        // Enemy update
    }

    void render() override {
        // Enemy rendering
    }
};
```

The engine can simply work with:

```cpp
GameObject*
```

without knowing whether it is a player or enemy.

---

# 39. Abstraction in Payment Systems

```text
             Payment
                |
      +---------+---------+
      |         |         |
    bKash     Nagad      Card
```

Abstract class:

```cpp
class Payment {
public:
    virtual void pay(double amount) = 0;

    virtual ~Payment() = default;
};
```

Implementations:

```cpp
class Bkash : public Payment {
public:
    void pay(double amount) override {
        std::cout << "bKash payment\n";
    }
};
```

```cpp
class Nagad : public Payment {
public:
    void pay(double amount) override {
        std::cout << "Nagad payment\n";
    }
};
```

```cpp
class Card : public Payment {
public:
    void pay(double amount) override {
        std::cout << "Card payment\n";
    }
};
```

The application depends on:

```cpp
Payment
```

rather than every specific payment system.

---

# 40. Abstraction in Database Systems

A database application may define:

```cpp
class Database {
public:
    virtual void connect() = 0;
    virtual void executeQuery() = 0;

    virtual ~Database() = default;
};
```

Different implementations:

```cpp
class MySQL : public Database {
public:
    void connect() override {
        std::cout << "Connected to MySQL\n";
    }

    void executeQuery() override {
        std::cout << "Executing MySQL query\n";
    }
};
```

Another:

```cpp
class PostgreSQL : public Database {
public:
    void connect() override {
        std::cout << "Connected to PostgreSQL\n";
    }

    void executeQuery() override {
        std::cout << "Executing PostgreSQL query\n";
    }
};
```

Application code can depend on:

```cpp
Database
```

instead of a specific database.

---

# 41. Abstraction in Software Architecture

A common architecture is:

```text
             UI
              |
              v
        Application Layer
              |
              v
        Business Logic
              |
              v
          Interfaces
              |
       +------+------+
       |             |
   Database       External API
```

Interfaces act as abstraction boundaries.

This makes it easier to:

* Replace implementations
* Test components
* Maintain code
* Scale applications
* Reduce coupling

---

# 42. Common Mistakes

## Mistake 1 — Trying to Instantiate an Abstract Class

```cpp
class Animal {
public:
    virtual void sound() = 0;
};

Animal animal; // ERROR
```

An abstract class cannot be instantiated.

---

## Mistake 2 — Forgetting to Override Pure Virtual Functions

```cpp
class Animal {
public:
    virtual void sound() = 0;
};
```

Incorrect:

```cpp
class Dog : public Animal {
};
```

`Dog` is still abstract because it does not implement `sound()`.

Correct:

```cpp
class Dog : public Animal {
public:
    void sound() override {
        std::cout << "Bark";
    }
};
```

---

# 43. Mistake 3 — Forgetting `override`

You can write:

```cpp
void sound() {
}
```

But it is safer to write:

```cpp
void sound() override {
}
```

`override` tells the compiler that the function must override a virtual function from the base class.

This catches mistakes.

---

# 44. Mistake 4 — Non-Virtual Destructor

Bad:

```cpp
class Animal {
public:
    virtual void sound() = 0;

    ~Animal() {}
};
```

Better:

```cpp
class Animal {
public:
    virtual void sound() = 0;

    virtual ~Animal() = default;
};
```

---

# 45. Mistake 5 — Making Everything Abstract

Not every class needs to be abstract.

Use abstraction when:

* Multiple implementations are expected
* A common contract is needed
* You want to hide implementation
* You need interchangeable components

Do not create abstraction simply for the sake of abstraction.

---

# 46. Mistake 6 — Too Many Interfaces

Overusing interfaces can make a project unnecessarily complicated.

Bad architecture:

```text
Interface A
   |
Interface B
   |
Interface C
   |
Interface D
   |
Concrete Class
```

Good abstraction should make the system **simpler**, not harder to understand.

---

# 47. Best Practices

### 1. Program against abstractions

Prefer:

```cpp
Database&
```

over:

```cpp
MySQLDatabase&
```

when the code does not need MySQL-specific behavior.

---

### 2. Use pure virtual functions for contracts

```cpp
virtual void save() = 0;
```

Clearly communicates required behavior.

---

### 3. Use `override`

```cpp
void save() override;
```

It improves safety and readability.

---

### 4. Use virtual destructors

For polymorphic base classes:

```cpp
virtual ~Base() = default;
```

---

### 5. Prefer composition when inheritance is unnecessary

Not every relationship needs inheritance.

Use:

```text
"is-a"
```

for inheritance.

Use:

```text
"has-a"
```

for composition.

---

### 6. Keep interfaces focused

Instead of one huge interface:

```cpp
class Everything {
    // 50 functions
};
```

prefer smaller interfaces based on responsibilities.

---

# 48. Abstraction and SOLID

Abstraction plays an important role in SOLID design.

Especially:

### Single Responsibility Principle

Classes should have focused responsibilities.

### Open/Closed Principle

Code should be open for extension but closed for unnecessary modification.

### Liskov Substitution Principle

Derived classes should be usable through their base abstraction without breaking expected behavior.

### Interface Segregation Principle

Clients should not be forced to depend on methods they do not need.

### Dependency Inversion Principle

High-level modules should depend on abstractions rather than concrete implementations.

---

# 49. Complete Advanced Example

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>

class Payment {
public:
    virtual void pay(double amount) = 0;

    virtual void refund(double amount) = 0;

    virtual std::string getName() const = 0;

    virtual ~Payment() = default;
};

class Bkash : public Payment {
public:

    void pay(double amount) override {
        std::cout
            << "Paid " << amount
            << " using bKash\n";
    }

    void refund(double amount) override {
        std::cout
            << "Refunded " << amount
            << " through bKash\n";
    }

    std::string getName() const override {
        return "bKash";
    }
};

class CardPayment : public Payment {
public:

    void pay(double amount) override {
        std::cout
            << "Paid " << amount
            << " using Card\n";
    }

    void refund(double amount) override {
        std::cout
            << "Refunded " << amount
            << " through Card\n";
    }

    std::string getName() const override {
        return "Card";
    }
};

class PaymentService {
public:

    void process(
        Payment& payment,
        double amount
    ) {
        std::cout
            << "Processing with "
            << payment.getName()
            << '\n';

        payment.pay(amount);
    }
};

int main() {

    Bkash bkash;
    CardPayment card;

    PaymentService service;

    service.process(bkash, 1000);
    service.process(card, 2000);

    return 0;
}
```

The important architecture is:

```text
                 Payment
                    |
          +---------+---------+
          |                   |
        Bkash              CardPayment
          |                   |
          +---------+---------+
                    |
                    v
             PaymentService
```

`PaymentService` does not need to know how bKash or Card payments work.

It only knows the abstraction:

```cpp
Payment
```

This is one of the most important practical uses of abstraction.

---

# 50. Abstraction vs Concrete Implementation

Think about a coffee machine.

### Abstraction

```cpp
makeCoffee();
```

### Implementation

```cpp
grindBeans();
heatWater();
pumpWater();
mixCoffee();
cleanMachine();
```

The user sees:

```text
makeCoffee()
```

The machine handles:

```text
grindBeans()
heatWater()
pumpWater()
mixCoffee()
cleanMachine()
```

Therefore:

```text
Abstraction
    =
Expose essential behavior
+
Hide unnecessary implementation
```

---

# 51. Abstraction Mental Model

Remember this:

```text
                 ABSTRACTION
                      |
                      v
          +-----------------------+
          | What can I do?        |
          +-----------------------+
                      |
                      v
              Public Interface
                      |
                      v
          +-----------------------+
          | How does it work?    |
          |                       |
          | Hidden Implementation |
          +-----------------------+
```

The user should interact with:

```text
WHAT
```

rather than needing to understand:

```text
HOW
```

---

# 52. Abstraction vs Encapsulation vs Inheritance vs Polymorphism

| Concept       | Main Purpose                                         |
| ------------- | ---------------------------------------------------- |
| Encapsulation | Protect and control data                             |
| Abstraction   | Hide unnecessary complexity                          |
| Inheritance   | Reuse/extend existing behavior                       |
| Polymorphism  | Allow one interface to have multiple implementations |

### Simple memory trick

```text
Encapsulation → Protect
Abstraction   → Hide
Inheritance   → Reuse
Polymorphism  → Many forms
```

---

# 53. Four Pillars of OOP

```text
                 OOP
                  |
       +----------+----------+
       |          |          |
       v          v          v
 Encapsulation Abstraction Inheritance
       |          |          |
       +----------+----------+
                  |
                  v
            Polymorphism
```

They often work together.

For example:

```cpp
class Shape {
private:
    std::string name;

public:
    virtual void draw() = 0;

    virtual ~Shape() = default;
};
```

Here:

* `private` → Encapsulation
* Pure virtual `draw()` → Abstraction
* `Circle : public Shape` → Inheritance
* Calling `draw()` through `Shape*` → Polymorphism

---

# 54. When Should You Use Abstraction?

Use abstraction when:

### 1. Multiple implementations exist

```text
Payment
 ├── bKash
 ├── Nagad
 └── Card
```

### 2. You want to hide complex implementation

```text
API
 |
Hidden internal logic
```

### 3. You want interchangeable components

```text
Database
 ├── MySQL
 └── PostgreSQL
```

### 4. You want loose coupling

```text
Service → Interface ← Implementation
```

### 5. You want easier testing

You can replace a real implementation with a test implementation.

---

# 55. When Not to Use Abstraction

Do not create an abstraction simply because OOP allows it.

For a simple class:

```cpp
class Student {
public:
    void study();
};
```

There may be no need for:

```cpp
class IStudent;
class AbstractStudent;
class StudentBase;
```

if there is only one implementation and no meaningful abstraction boundary.

Good design is about **appropriate abstraction**, not maximum abstraction.

---

# 56. Interview Questions

### Q1. What is abstraction?

Abstraction is the process of hiding unnecessary implementation details and exposing only essential functionality.

---

### Q2. How is abstraction implemented in C++?

Common techniques include:

* Access specifiers
* Classes
* Abstract classes
* Pure virtual functions
* Interface-like classes
* Polymorphism

---

### Q3. What is an abstract class?

A class that cannot be instantiated directly and typically contains at least one pure virtual function.

---

### Q4. What is a pure virtual function?

A virtual function declared with:

```cpp
= 0;
```

Example:

```cpp
virtual void display() = 0;
```

---

### Q5. Can an abstract class have a constructor?

Yes.

---

### Q6. Can an abstract class have normal functions?

Yes.

---

### Q7. Can an abstract class have data members?

Yes.

---

### Q8. Can we create an object of an abstract class?

No.

```cpp
Animal a; // Error
```

---

### Q9. Why use a virtual destructor?

To ensure proper destruction when a derived object is deleted through a base pointer.

---

### Q10. Does C++ have an `interface` keyword?

No. Interface-like behavior is commonly implemented using classes with pure virtual functions.

---

# 57. Final Summary

Abstraction is one of the most important concepts in C++ OOP.

The core idea is:

> **Expose what the object can do and hide how it does it.**

For example:

```cpp
class Payment {
public:
    virtual void pay(double amount) = 0;

    virtual ~Payment() = default;
};
```

The user knows:

```cpp
payment.pay(1000);
```

But does not need to know:

```text
Network communication
Authentication
Database operations
Encryption
Transaction processing
```

The implementation remains hidden.

The complete concept can be remembered as:

```text
                ABSTRACTION
                     |
                     v
              Hide Complexity
                     |
                     v
             Expose Interface
                     |
                     v
            Pure Virtual Functions
                     |
                     v
              Abstract Classes
                     |
                     v
                Interfaces
                     |
                     v
             Polymorphic Design
                     |
                     v
          Flexible Software Architecture
```

---

# 58. Abstraction Checklist

Before moving to the next OOP concept, make sure you understand:

* [ ] What abstraction means
* [ ] Why abstraction is needed
* [ ] Encapsulation vs abstraction
* [ ] Access specifiers
* [ ] Abstract classes
* [ ] Pure virtual functions
* [ ] `virtual`
* [ ] `override`
* [ ] Virtual destructors
* [ ] Interfaces in C++
* [ ] Abstract class constructors
* [ ] Abstract class data members
* [ ] Normal functions inside abstract classes
* [ ] Abstraction + inheritance
* [ ] Abstraction + polymorphism
* [ ] Dependency inversion
* [ ] Dependency injection
* [ ] Loose coupling
* [ ] SOLID relationship
* [ ] Real-world abstraction
* [ ] Smart pointers with abstract classes
* [ ] When to use abstraction
* [ ] When not to overuse abstraction

---

# 59. One-Line Definition

> **Abstraction in C++ is the process of exposing essential behavior through a simple interface while hiding unnecessary implementation details.**

### Mental Shortcut

```text
ENCAPSULATION → Protect data
ABSTRACTION   → Hide complexity
INHERITANCE   → Reuse/extend
POLYMORPHISM  → One interface, many behaviors
```
