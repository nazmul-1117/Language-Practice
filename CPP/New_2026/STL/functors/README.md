# C++ STL — Functors

## 1. What is a Functor?

A **Functor**, also called a **Function Object**, is an object that behaves like a function.

The key idea is:

> A class or struct becomes a functor when it provides `operator()`.

Your STL material defines it exactly this way: a functor is an object that behaves like a function and is created by defining `operator()`.

Basic example:

```cpp
struct IsEven
{
    bool operator()(int value) const
    {
        return value % 2 == 0;
    }
};
```

Now create an object:

```cpp
IsEven check;
```

and call it like a function:

```cpp
cout << check(10);
```

Output:

```text
1
```

The important part is:

```cpp
check(10)
```

Although `check` is an **object**, it behaves like a function because it has:

```cpp
operator()
```

---

# 2. Why Is It Called a Function Object?

Normally we call a function like:

```cpp
isEven(10);
```

With a functor:

```cpp
IsEven check;

check(10);
```

`check` is not a normal function.

It is an **object**.

But:

```cpp
check(10)
```

works because C++ interprets it approximately as:

```cpp
check.operator()(10);
```

This is the fundamental concept.

---

# 3. `operator()` — The Heart of a Functor

Consider:

```cpp
struct Add
{
    int operator()(int a, int b) const
    {
        return a + b;
    }
};
```

Create an object:

```cpp
Add add;
```

Call:

```cpp
cout << add(10, 20);
```

Output:

```text
30
```

Internally, this:

```cpp
add(10, 20);
```

is essentially:

```cpp
add.operator()(10, 20);
```

So:

```text
Object
   ↓
operator()
   ↓
Behaves like function
```

---

# 4. Basic Functor Structure

General form:

```cpp
struct FunctorName
{
    ReturnType operator()(parameters) const
    {
        // logic
    }
};
```

Example:

```cpp
struct Multiply
{
    int operator()(int a, int b) const
    {
        return a * b;
    }
};
```

Usage:

```cpp
Multiply multiply;

cout << multiply(5, 4);
```

Output:

```text
20
```

---

# 5. Functor with One Parameter

```cpp
struct Square
{
    int operator()(int x) const
    {
        return x * x;
    }
};
```

Usage:

```cpp
Square square;

cout << square(5);
```

Output:

```text
25
```

---

# 6. Functor with Multiple Parameters

A functor can accept multiple parameters.

```cpp
struct Add
{
    int operator()(int a, int b, int c) const
    {
        return a + b + c;
    }
};
```

Usage:

```cpp
Add add;

cout << add(10, 20, 30);
```

Output:

```text
60
```

---

# 7. Why Do We Need Functors?

The main reason functors are important in STL is that **algorithms often need a custom operation**.

For example:

```cpp
sort(v.begin(), v.end());
```

uses the default sorting rule.

But what if we want descending order?

```cpp
sort(
    v.begin(),
    v.end(),
    greater<int>()
);
```

Here:

```cpp
greater<int>()
```

is a function object.

Similarly, algorithms such as:

```text
sort()
count_if()
find_if()
for_each()
transform()
accumulate()
```

can receive callable objects.

---

# 8. Functor + STL Algorithm

Your source explicitly shows functors being passed to STL algorithms.

Example:

```cpp
struct IsEven
{
    bool operator()(int value) const
    {
        return value % 2 == 0;
    }
};
```

Use it:

```cpp
vector<int> numbers = {
    1, 2, 3, 4, 5, 6
};

int count = count_if(
    numbers.begin(),
    numbers.end(),
    IsEven{}
);
```

Result:

```text
3
```

because:

```text
2
4
6
```

are even.

---

# 9. What Does `IsEven{}` Mean?

This:

```cpp
IsEven{}
```

creates a temporary object of type `IsEven`.

You can also write:

```cpp
IsEven check;

count_if(
    numbers.begin(),
    numbers.end(),
    check
);
```

Both provide the algorithm with a callable object.

---

# 10. Functor vs Normal Function

Normal function:

```cpp
bool isEven(int x)
{
    return x % 2 == 0;
}
```

Usage:

```cpp
count_if(
    v.begin(),
    v.end(),
    isEven
);
```

Functor:

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

Usage:

```cpp
count_if(
    v.begin(),
    v.end(),
    IsEven{}
);
```

Both can solve the same simple problem.

But functors have additional capabilities.

---

# 11. Main Advantage of Functors — State

A normal function does not naturally carry object-specific state.

A functor can store data inside itself.

Example:

```cpp
struct GreaterThan
{
    int limit;

    GreaterThan(int value)
        : limit(value)
    {
    }

    bool operator()(int x) const
    {
        return x > limit;
    }
};
```

Now:

```cpp
GreaterThan check(50);
```

means:

```text
limit = 50
```

Then:

```cpp
check(70);
```

returns:

```text
true
```

while:

```cpp
check(30);
```

returns:

```text
false
```

---

# 12. Stateful Functor

This is one of the most important reasons functors are useful.

```cpp
struct GreaterThan
{
    int limit;

    GreaterThan(int limit)
        : limit(limit)
    {
    }

    bool operator()(int value) const
    {
        return value > limit;
    }
};
```

Usage:

```cpp
vector<int> numbers = {
    10, 20, 60, 80, 40
};

int count = count_if(
    numbers.begin(),
    numbers.end(),
    GreaterThan(50)
);
```

Output:

```text
2
```

Because:

```text
60 > 50
80 > 50
```

---

# 13. Why State Is Powerful

We can create different functors with different states.

```cpp
GreaterThan(10)
GreaterThan(50)
GreaterThan(100)
```

All use the same class.

For example:

```cpp
count_if(
    v.begin(),
    v.end(),
    GreaterThan(10)
);
```

and:

```cpp
count_if(
    v.begin(),
    v.end(),
    GreaterThan(100)
);
```

The algorithm remains unchanged.

Only the functor's state changes.

---

# 14. Functor with Internal Counter

A functor can maintain state between calls.

```cpp
struct Counter
{
    int count = 0;

    void operator()(int value)
    {
        count++;
    }
};
```

Usage:

```cpp
Counter c;

for_each(
    v.begin(),
    v.end(),
    c
);

cout << c.count;
```

However, there is an important issue with some algorithms: algorithms may copy callable objects. Therefore, relying on internal mutation requires understanding how the particular algorithm invokes the callable.

For simple counting, `count()` or `count_if()` is usually preferable.

---

# 15. `const` After `operator()`

You will often see:

```cpp
bool operator()(int x) const
```

What does the final `const` mean?

It means the functor's `operator()` promises not to modify the object's non-mutable state.

Example:

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

This is a common STL style.

---

# 16. Functor as Comparator

One of the most important applications of functors is **custom comparison**.

Example:

```cpp
struct Descending
{
    bool operator()(int a, int b) const
    {
        return a > b;
    }
};
```

Use:

```cpp
sort(
    v.begin(),
    v.end(),
    Descending{}
);
```

Result:

```text
9 7 5 3 1
```

instead of:

```text
1 3 5 7 9
```

---

# 17. How Comparator Functor Works

Suppose:

```text
a = 5
b = 3
```

The comparator:

```cpp
a > b
```

returns:

```text
true
```

meaning:

```text
5 should come before 3
```

Therefore:

```cpp
struct Descending
{
    bool operator()(int a, int b) const
    {
        return a > b;
    }
};
```

means:

> Put larger values before smaller values.

---

# 18. `std::less`

C++ provides built-in function objects.

For example:

```cpp
less<int>()
```

means roughly:

```cpp
a < b
```

Example:

```cpp
sort(
    v.begin(),
    v.end(),
    less<int>()
);
```

This gives ascending order.

---

# 19. `std::greater`

```cpp
greater<int>()
```

means roughly:

```cpp
a > b
```

Example:

```cpp
sort(
    v.begin(),
    v.end(),
    greater<int>()
);
```

This gives descending order.

---

# 20. Standard Comparison Functors

Important STL function objects include:

```text
less
greater
less_equal
greater_equal
equal_to
not_equal_to
```

Conceptually:

| Functor            | Operation |
| ------------------ | --------- |
| `less<T>`          | `a < b`   |
| `greater<T>`       | `a > b`   |
| `less_equal<T>`    | `a <= b`  |
| `greater_equal<T>` | `a >= b`  |
| `equal_to<T>`      | `a == b`  |
| `not_equal_to<T>`  | `a != b`  |

Header:

```cpp
#include <functional>
```

---

# 21. Arithmetic Functors

STL also provides arithmetic function objects.

Important ones:

```text
plus
minus
multiplies
divides
modulus
negate
```

Conceptually:

| Functor         | Operation |
| --------------- | --------- |
| `plus<T>`       | `a + b`   |
| `minus<T>`      | `a - b`   |
| `multiplies<T>` | `a * b`   |
| `divides<T>`    | `a / b`   |
| `modulus<T>`    | `a % b`   |
| `negate<T>`     | `-a`      |

Example:

```cpp
plus<int> add;

cout << add(10, 20);
```

Output:

```text
30
```

---

# 22. Logical Functors

STL also provides:

```text
logical_and
logical_or
logical_not
```

Conceptually:

```text
logical_and
→ a && b

logical_or
→ a || b

logical_not
→ !a
```

Example:

```cpp
logical_and<bool> operation;

cout << operation(true, false);
```

Output:

```text
0
```

---

# 23. Standard Functor Families

A useful classification:

```text
std::functional
      │
      ├── Comparison
      │     ├── less
      │     ├── greater
      │     ├── equal_to
      │     └── not_equal_to
      │
      ├── Arithmetic
      │     ├── plus
      │     ├── minus
      │     ├── multiplies
      │     ├── divides
      │     └── modulus
      │
      └── Logical
            ├── logical_and
            ├── logical_or
            └── logical_not
```

---

# 24. Functor with `find_if()`

```cpp
struct IsPositive
{
    bool operator()(int x) const
    {
        return x > 0;
    }
};
```

Use:

```cpp
auto it = find_if(
    v.begin(),
    v.end(),
    IsPositive{}
);
```

Then:

```cpp
if(it != v.end())
{
    cout << *it;
}
```

---

# 25. Functor with `remove_if()`

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

Use:

```cpp
v.erase(
    remove_if(
        v.begin(),
        v.end(),
        IsEven{}
    ),
    v.end()
);
```

This removes even values.

---

# 26. Functor with `for_each()`

```cpp
struct Print
{
    void operator()(int x) const
    {
        cout << x << " ";
    }
};
```

Use:

```cpp
for_each(
    v.begin(),
    v.end(),
    Print{}
);
```

Output:

```text
1 2 3 4 5
```

---

# 27. Functor with `transform()`

Suppose we want to double every value.

```cpp
struct Double
{
    int operator()(int x) const
    {
        return x * 2;
    }
};
```

Use:

```cpp
transform(
    v.begin(),
    v.end(),
    v.begin(),
    Double{}
);
```

If:

```text
1 2 3 4
```

then:

```text
2 4 6 8
```

---

# 28. Functor with `accumulate()`

`accumulate()` can receive a custom operation.

```cpp
struct Multiply
{
    int operator()(int a, int b) const
    {
        return a * b;
    }
};
```

Then:

```cpp
int result = accumulate(
    v.begin(),
    v.end(),
    1,
    Multiply{}
);
```

For:

```text
1 2 3 4
```

result:

```text
24
```

because:

```text
1 × 1 × 2 × 3 × 4
= 24
```

---

# 29. Functor with User-Defined Objects

Functors become particularly useful when working with classes or structs.

Suppose:

```cpp
struct Student
{
    string name;
    int marks;
};
```

We want to sort students by marks.

Create:

```cpp
struct CompareMarks
{
    bool operator()(
        const Student& a,
        const Student& b
    ) const
    {
        return a.marks < b.marks;
    }
};
```

Then:

```cpp
sort(
    students.begin(),
    students.end(),
    CompareMarks{}
);
```

Now students are sorted by marks.

---

# 30. Sort Students in Descending Order

```cpp
struct CompareMarks
{
    bool operator()(
        const Student& a,
        const Student& b
    ) const
    {
        return a.marks > b.marks;
    }
};
```

Now:

```text
90
85
75
60
```

instead of:

```text
60
75
85
90
```

---

# 31. Multiple Conditions

Functors can implement complicated comparison rules.

Suppose students should be sorted:

1. Higher marks first
2. If marks are equal, smaller age first

```cpp
struct Student
{
    string name;
    int marks;
    int age;
};
```

Functor:

```cpp
struct CompareStudent
{
    bool operator()(
        const Student& a,
        const Student& b
    ) const
    {
        if(a.marks != b.marks)
            return a.marks > b.marks;

        return a.age < b.age;
    }
};
```

Use:

```cpp
sort(
    students.begin(),
    students.end(),
    CompareStudent{}
);
```

This is extremely useful in competitive programming.

---

# 32. Functor Can Store Configuration

Example:

```cpp
struct CompareBy
{
    bool ascending;

    CompareBy(bool ascending)
        : ascending(ascending)
    {
    }

    bool operator()(int a, int b) const
    {
        if(ascending)
            return a < b;

        return a > b;
    }
};
```

Usage:

```cpp
sort(
    v.begin(),
    v.end(),
    CompareBy(true)
);
```

Ascending.

Or:

```cpp
sort(
    v.begin(),
    v.end(),
    CompareBy(false)
);
```

Descending.

This demonstrates the major advantage of stateful functors.

---

# 33. Functor vs Lambda

A lambda:

```cpp
auto isEven = [](int x)
{
    return x % 2 == 0;
};
```

A functor:

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

Both can be used:

```cpp
count_if(
    v.begin(),
    v.end(),
    isEven
);
```

or:

```cpp
count_if(
    v.begin(),
    v.end(),
    IsEven{}
);
```

Your source notes that modern C++ commonly uses lambdas instead of manually creating simple functors.

---

# 34. Functor vs Lambda — When to Use Which?

### Lambda

Best when:

```text
small
simple
one-time operation
```

Example:

```cpp
sort(
    v.begin(),
    v.end(),
    [](int a, int b)
    {
        return a > b;
    }
);
```

### Functor

Useful when:

```text
complex logic
reusable operation
state/configuration
named behavior
```

Example:

```cpp
sort(
    students.begin(),
    students.end(),
    CompareStudent{}
);
```

---

# 35. Functor vs Function

### Function

```cpp
bool isEven(int x)
{
    return x % 2 == 0;
}
```

### Functor

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

### Lambda

```cpp
[](int x)
{
    return x % 2 == 0;
}
```

All three can be **callable**.

---

# 36. Callable Objects

A broader C++ concept is **Callable**.

Your source groups several callable types together:

```text
Callable
│
├── Function
├── Function Pointer
├── Lambda
├── Functor
└── std::function
```

This distinction is important:

> A functor is a type of callable, but not every callable is a functor.

---

# 37. Function Pointer

Example:

```cpp
int add(int a, int b)
{
    return a + b;
}
```

Function pointer:

```cpp
int (*operation)(int, int) = add;
```

Call:

```cpp
cout << operation(10, 20);
```

Output:

```text
30
```

---

# 38. Lambda

```cpp
auto operation = [](int a, int b)
{
    return a + b;
};
```

Call:

```cpp
cout << operation(10, 20);
```

Output:

```text
30
```

---

# 39. Functor

```cpp
struct Add
{
    int operator()(int a, int b)
    {
        return a + b;
    }
};
```

Call:

```cpp
Add operation;

cout << operation(10, 20);
```

Output:

```text
30
```

So all three can be used as callables.

---

# 40. `std::function`

`std::function` is a general-purpose polymorphic function wrapper.

Your source introduces it after functors and lambdas.

Example:

```cpp
#include <functional>

function<int(int, int)> operation;

operation = [](int a, int b)
{
    return a + b;
};

cout << operation(10, 20);
```

Output:

```text
30
```

It can hold:

```text
Function
Lambda
Functor
Function pointer
```

---

# 41. Functor + `std::function`

A functor can be stored inside `std::function`.

```cpp
struct Add
{
    int operator()(int a, int b) const
    {
        return a + b;
    }
};

function<int(int, int)> operation = Add{};

cout << operation(10, 20);
```

Output:

```text
30
```

---

# 42. Functor and Templates

Functors work especially well with templates.

Example:

```cpp
template<typename Func>
void execute(Func operation)
{
    cout << operation(10, 20);
}
```

Now:

```cpp
struct Add
{
    int operator()(int a, int b) const
    {
        return a + b;
    }
};
```

Call:

```cpp
execute(Add{});
```

Output:

```text
30
```

The template does not need to know the exact type of the functor.

---

# 43. Generic Functor

A functor can itself be templated.

Modern C++:

```cpp
struct Add
{
    template<typename T>
    T operator()(T a, T b) const
    {
        return a + b;
    }
};
```

Now:

```cpp
Add add;

cout << add(10, 20);
cout << add(2.5, 3.5);
```

Both can work with appropriate types.

---

# 44. Generic Lambda vs Functor

Generic lambda:

```cpp
auto add = [](auto a, auto b)
{
    return a + b;
};
```

Functor:

```cpp
struct Add
{
    template<typename T>
    T operator()(T a, T b) const
    {
        return a + b;
    }
};
```

They express a similar idea.

For modern C++, generic lambdas are often simpler for short operations.

---

# 45. Predicate

A **predicate** is a callable that returns a Boolean result.

Example:

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

It answers a question:

```text
Is x even?
```

Result:

```text
true / false
```

Predicates are heavily used by STL algorithms.

---

# 46. Unary Predicate

A unary predicate accepts one argument.

```cpp
bool operator()(int x) const
```

Example:

```cpp
find_if(
    v.begin(),
    v.end(),
    IsEven{}
);
```

One element is passed at a time.

---

# 47. Binary Predicate

A binary predicate accepts two arguments.

```cpp
bool operator()(int a, int b) const
```

Example:

```cpp
sort(
    v.begin(),
    v.end(),
    Compare{}
);
```

The comparator receives two elements.

---

# 48. Unary vs Binary Functor

```text
Unary Functor
     ↓
one argument

operator()(x)
```

```text
Binary Functor
     ↓
two arguments

operator()(a, b)
```

Examples:

```text
count_if()
→ unary predicate

sort()
→ binary comparison
```

---

# 49. Function Objects for Algorithms

Think about STL algorithms like this:

```text
Algorithm
    +
Iterator Range
    +
Callable
    ↓
Result
```

For example:

```cpp
sort(
    v.begin(),
    v.end(),
    greater<int>()
);
```

Here:

```text
sort
 ↓
Algorithm

begin/end
 ↓
Iterators

greater<int>()
 ↓
Functor
```

This is the same fundamental STL architecture described in your source.

---

# 50. Functor + Iterator + Algorithm

A complete STL example:

```cpp
vector<int> numbers = {
    1, 2, 3, 4, 5, 6
};

struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};

int result = count_if(
    numbers.begin(),
    numbers.end(),
    IsEven{}
);
```

Breakdown:

```text
vector
 ↓
Container

begin/end
 ↓
Iterators

count_if
 ↓
Algorithm

IsEven
 ↓
Functor
```

This is one of the most important STL patterns to understand.

---

# 51. Common Built-in Functors

For practical STL programming, remember these.

### Comparison

```cpp
less<int>()
greater<int>()
less_equal<int>()
greater_equal<int>()
equal_to<int>()
not_equal_to<int>()
```

### Arithmetic

```cpp
plus<int>()
minus<int>()
multiplies<int>()
divides<int>()
modulus<int>()
negate<int>()
```

### Logical

```cpp
logical_and<bool>()
logical_or<bool>()
logical_not<bool>()
```

Header:

```cpp
#include <functional>
```

---

# 52. Most Common Competitive Programming Usage

### Descending sort

```cpp
sort(
    v.begin(),
    v.end(),
    greater<int>()
);
```

### Ascending sort

```cpp
sort(
    v.begin(),
    v.end(),
    less<int>()
);
```

### Custom object sorting

```cpp
sort(
    students.begin(),
    students.end(),
    CompareStudent{}
);
```

### Conditional searching

```cpp
find_if(
    v.begin(),
    v.end(),
    IsEven{}
);
```

### Conditional counting

```cpp
count_if(
    v.begin(),
    v.end(),
    IsEven{}
);
```

---

# 53. Common Mistake — Forgetting `operator()`

Wrong:

```cpp
struct IsEven
{
    bool check(int x)
    {
        return x % 2 == 0;
    }
};
```

This is just a class with a member function.

It is not a functor in the usual sense.

Correct:

```cpp
struct IsEven
{
    bool operator()(int x) const
    {
        return x % 2 == 0;
    }
};
```

Now:

```cpp
IsEven{}(10);
```

works.

---

# 54. Common Mistake — Wrong Comparator

Suppose:

```cpp
sort(
    v.begin(),
    v.end(),
    [](int a, int b)
    {
        return a < b;
    }
);
```

This produces ascending order.

For descending:

```cpp
return a > b;
```

Remember:

```text
a < b
→ smaller before larger

a > b
→ larger before smaller
```

---

# 55. Common Mistake — Returning the Wrong Type

A comparator normally returns something usable as a Boolean condition:

```cpp
bool operator()(int a, int b) const
{
    return a < b;
}
```

Do not write:

```cpp
int operator()(int a, int b)
```

unless you have a specific reason and the callable is not being used as a standard comparator.

---

# 56. Common Mistake — Modifying Elements in a Comparator

A comparison function should normally compare its arguments rather than modify them.

Good:

```cpp
bool operator()(const Student& a,
                const Student& b) const
{
    return a.marks > b.marks;
}
```

This is why you commonly see:

```cpp
const Student&
```

and:

```cpp
const
```

after `operator()`.

---

# 57. Functor vs Lambda — Quick Table

| Feature                         | Functor     | Lambda                |
| ------------------------------- | ----------- | --------------------- |
| Callable                        | ✓           | ✓                     |
| Uses `operator()`               | ✓           | Hidden/generated      |
| Can store state                 | ✓           | ✓                     |
| Reusable named type             | ✓           | Usually less explicit |
| Good for one-time operation     | Sometimes   | ✓                     |
| Good for complex reusable logic | ✓           | ✓                     |
| Common modern syntax            | Less common | Very common           |

---

# 58. Functor vs Function vs Lambda

```text
                 CALLABLE
                    │
        ┌───────────┼───────────┐
        │           │           │
     Function     Lambda     Functor
        │           │           │
    function()    [](...)    operator()
```

All can be passed to suitable STL algorithms.

---

# 59. Important Mental Model

Do not think:

> "Functor = some complicated STL syntax."

Think:

> **Functor = object + `operator()`**

That's it.

Example:

```cpp
struct Add
{
    int operator()(int a, int b) const
    {
        return a + b;
    }
};
```

Then:

```cpp
Add add;

add(10, 20);
```

The object behaves like a function.

---

# 60. Functors and Lambdas in STL

Your source's overall STL mental model is:

```text
STL
 │
 ├── Containers
 │
 ├── Algorithms
 │
 ├── Iterators
 │
 └── Callables
       │
       ├── Functors
       │
       └── Lambdas
```

A typical example is:

```cpp
vector<int> numbers = {
    5, 2, 8, 1, 3
};

sort(
    numbers.begin(),
    numbers.end(),
    [](int a, int b)
    {
        return a < b;
    }
);
```

Here:

```text
vector
→ Container

begin/end
→ Iterators

sort
→ Algorithm

lambda
→ Callable
```

A functor can replace the lambda:

```cpp
struct Compare
{
    bool operator()(int a, int b) const
    {
        return a < b;
    }
};

sort(
    numbers.begin(),
    numbers.end(),
    Compare{}
);
```

---

# 61. Complete Functor Example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct GreaterThan
{
    int limit;

    GreaterThan(int value)
        : limit(value)
    {
    }

    bool operator()(int x) const
    {
        return x > limit;
    }
};

int main()
{
    vector<int> numbers = {
        10, 20, 60, 80, 40
    };

    int result = count_if(
        numbers.begin(),
        numbers.end(),
        GreaterThan(50)
    );

    cout << result;
}
```

Output:

```text
2
```

This example demonstrates:

```text
Functor
+
operator()
+
State
+
Constructor
+
STL algorithm
+
Iterators
```

---

# 62. Final Functor Cheat Sheet

### Definition

```text
Functor
=
Function Object
=
Object that behaves like a function
```

### Main requirement

```cpp
operator()
```

### Basic form

```cpp
struct MyFunctor
{
    ReturnType operator()(parameters) const
    {
        // logic
    }
};
```

### Create object

```cpp
MyFunctor f;
```

### Call

```cpp
f(arguments);
```

### STL usage

```cpp
algorithm(
    begin,
    end,
    MyFunctor{}
);
```

---

# 63. What You Should Memorize

### Must know

```cpp
operator()
```

### Must understand

```cpp
Functor{}
```

means:

> Create a temporary functor object.

### Must know applications

```cpp
sort()
find_if()
count_if()
for_each()
remove_if()
transform()
accumulate()
```

### Must know standard functors

```cpp
less
greater
equal_to
not_equal_to

plus
minus
multiplies
divides
modulus
negate

logical_and
logical_or
logical_not
```

### Must understand

```text
Functor
    ↓
Callable
    ↓
Passed to STL Algorithm
```

---

# 64. Final Mental Map

```text
                         CALLABLES
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
       Function            Lambda            Functor
                                                │
                                           operator()
                                                │
                                    ┌───────────┴───────────┐
                                    │                       │
                               Stateless                 Stateful
                                    │                       │
                                IsEven{}              GreaterThan(50)
                                    │                       │
                                    └───────────┬───────────┘
                                                │
                                         STL Algorithms
                                                │
                ┌───────────────┬──────────────┼──────────────┐
                │               │              │              │
              sort()         find_if()      count_if()    transform()
```

## The one sentence to remember

> **A functor is an object that behaves like a function because it overloads `operator()`, and STL uses these callable objects to customize algorithms.**

Your source places **Functors → Lambda Expressions → Function Objects/Callables → `std::function`** in this learning sequence, so the natural next chapter after this is **Lambda Expressions in depth**, including capture lists, `[=]`, `[&]`, `[this]`, mutable lambdas, generic lambdas, and lambda use with STL algorithms.
