# C++ STL — Standard Template Library

The **Standard Template Library (STL)** is one of the most important parts of modern C++.

STL provides reusable, generic components for working with:

* Data structures
* Collections
* Iteration
* Searching
* Sorting
* Algorithms
* Function objects
* Utilities
* Memory management

Instead of implementing common data structures and algorithms from scratch, C++ provides well-tested generic implementations through the STL.

The goal of learning STL is **not to memorize every function**.

The real goal is to understand:

> **Which container, algorithm, iterator, or utility should I use, and why?**

---

# Table of Contents

1. [What is STL?](#1-what-is-stl)
2. [STL Architecture](#2-stl-architecture)
3. [Containers](#3-containers)
4. [Sequence Containers](#4-sequence-containers)
5. [Associative Containers](#5-associative-containers)
6. [Unordered Containers](#6-unordered-containers)
7. [Container Adaptors](#7-container-adaptors)
8. [Container Comparison](#8-container-comparison)
9. [Iterators](#9-iterators)
10. [Iterator Categories](#10-iterator-categories)
11. [Algorithms](#11-algorithms)
12. [Sorting Algorithms](#12-sorting-algorithms)
13. [Searching Algorithms](#13-searching-algorithms)
14. [Modification Algorithms](#14-modification-algorithms)
15. [Numeric Algorithms](#15-numeric-algorithms)
16. [Functors](#16-functors)
17. [Lambda Expressions](#17-lambda-expressions)
18. [Function Objects and Callables](#18-function-objects-and-callables)
19. [std::function](#19-stdfunction)
20. [Pairs and Tuples](#20-pairs-and-tuples)
21. [Optional](#21-optional)
22. [Structured Bindings](#22-structured-bindings)
23. [Algorithms + Lambdas + Containers](#23-algorithms--lambdas--containers)
24. [Range-Based for Loops](#24-range-based-for-loops)
25. [Common STL Patterns](#25-common-stl-patterns)
26. [Time Complexity](#26-time-complexity)
27. [Choosing the Right Container](#27-choosing-the-right-container)
28. [Common Mistakes](#28-common-mistakes)
29. [Recommended Learning Order](#29-recommended-learning-order)
30. [Practice Projects](#30-practice-projects)

---

# 1. What is STL?

STL stands for:

> **Standard Template Library**

It is a collection of generic C++ components built primarily around templates.

The major STL concepts are:

```text
STL
│
├── Containers
│
├── Iterators
│
├── Algorithms
│
├── Function Objects / Functors
│
├── Lambdas
│
└── Utilities
```

For example, suppose we want to store numbers.

Without STL, we might manually implement:

```cpp
int numbers[5];
```

But STL provides:

```cpp
std::vector<int> numbers;
```

Then we can use existing algorithms:

```cpp
std::sort(numbers.begin(), numbers.end());
```

The important idea is:

```text
Container
    +
Iterator
    +
Algorithm
    +
Callable
    =
Powerful Generic Code
```

---

# 2. STL Architecture

STL is designed around cooperation between different components.

For example:

```cpp
std::vector<int> numbers = {5, 2, 8, 1, 3};

std::sort(numbers.begin(), numbers.end());
```

Here:

```text
vector
  ↓
stores data

begin()
  ↓
returns iterator

end()
  ↓
marks end of range

sort()
  ↓
algorithm

result
  ↓
sorted vector
```

This separation is extremely important.

The algorithm does not need to know that it is working specifically with a `vector`.

It works with iterators.

This is one of the central design ideas of STL.

---

# 3. Containers

Containers are objects that store collections of data.

The major STL containers are:

```text
Containers
│
├── Sequence Containers
│   ├── vector
│   ├── array
│   ├── deque
│   ├── list
│   └── forward_list
│
├── Associative Containers
│   ├── map
│   ├── set
│   ├── multimap
│   └── multiset
│
├── Unordered Containers
│   ├── unordered_map
│   ├── unordered_set
│   ├── unordered_multimap
│   └── unordered_multiset
│
└── Container Adaptors
    ├── stack
    ├── queue
    └── priority_queue
```

The most important containers to master first are:

```cpp
std::vector
std::array
std::string
std::map
std::unordered_map
std::set
std::unordered_set
```

---

# 4. Sequence Containers

Sequence containers store elements in a particular sequence.

## 4.1 std::vector

`vector` is a dynamic array.

```cpp
#include <vector>

std::vector<int> numbers;
```

Add elements:

```cpp
numbers.push_back(10);
numbers.push_back(20);
numbers.push_back(30);
```

Result:

```text
10 20 30
```

Access:

```cpp
numbers[0];
numbers.at(1);
numbers.front();
numbers.back();
```

Example:

```cpp
#include <iostream>
#include <vector>

int main()
{
    std::vector<int> numbers = {10, 20, 30};

    for (int number : numbers)
    {
        std::cout << number << '\n';
    }
}
```

### Important properties

```text
Contiguous memory
Dynamic size
Fast random access
Efficient at the end
```

### Complexity

| Operation     |     Complexity |
| ------------- | -------------: |
| Access        |           O(1) |
| `push_back()` | Amortized O(1) |
| `pop_back()`  |           O(1) |
| Insert middle |           O(n) |
| Delete middle |           O(n) |
| Search        |           O(n) |

### When should you use vector?

Use `vector` when:

* You need dynamic storage.
* You need random access.
* You frequently iterate through elements.
* Most insertions happen at the end.

In general:

> **`vector` should usually be your default sequence container.**

---

# 4.2 std::array

`std::array` is a fixed-size container.

```cpp
#include <array>

std::array<int, 5> numbers = {1, 2, 3, 4, 5};
```

Unlike a built-in array, `std::array` provides STL functionality.

```cpp
numbers.size();
numbers.front();
numbers.back();
```

Random access:

```cpp
numbers[2];
```

### Difference

```cpp
int arr[5];
```

versus:

```cpp
std::array<int, 5> arr;
```

`std::array` works naturally with STL algorithms.

```cpp
std::sort(arr.begin(), arr.end());
```

---

# 4.3 std::deque

`deque` means:

> Double-ended queue

It allows efficient insertion/removal from both ends.

```cpp
#include <deque>

std::deque<int> numbers;

numbers.push_back(10);
numbers.push_front(5);

numbers.pop_back();
numbers.pop_front();
```

Conceptually:

```text
push_front()
     ↓
[ 5 ][10][20][30]
                 ↑
             push_back()
```

Use `deque` when you need efficient operations at both ends.

---

# 4.4 std::list

`std::list` is a doubly linked list.

```cpp
#include <list>

std::list<int> numbers = {10, 20, 30};
```

Insert:

```cpp
numbers.push_front(5);
numbers.push_back(40);
```

A list does not provide efficient random access.

This is invalid:

```cpp
numbers[2]; // ❌
```

You must use iterators.

```cpp
auto it = numbers.begin();

std::advance(it, 2);

std::cout << *it;
```

### Complexity

| Operation                | Complexity |
| ------------------------ | ---------: |
| Insert at known position |       O(1) |
| Delete at known position |       O(1) |
| Random access            |       O(n) |
| Search                   |       O(n) |

### vector vs list

Do not automatically assume:

> "Linked list = faster."

The correct question is:

> **What operation do I need frequently?**

For many real-world applications, `vector` is preferable because of its contiguous memory and cache efficiency.

---

# 4.5 std::forward_list

`forward_list` is a singly linked list.

```cpp
std::forward_list<int> numbers = {1, 2, 3};
```

It uses less memory than `list`, but only supports forward traversal.

Use it when you specifically need singly-linked-list behavior.

---

# 5. Associative Containers

Associative containers store elements according to keys.

The primary containers are:

```cpp
std::map
std::set
std::multimap
std::multiset
```

They are generally implemented using balanced tree structures.

---

# 5.1 std::map

A `map` stores:

```text
key → value
```

Example:

```cpp
#include <map>

std::map<std::string, int> ages;

ages["Alice"] = 25;
ages["Bob"] = 30;
ages["Charlie"] = 22;
```

Conceptually:

```text
Alice   → 25
Bob     → 30
Charlie → 22
```

Access:

```cpp
std::cout << ages["Alice"];
```

Search:

```cpp
auto it = ages.find("Bob");

if (it != ages.end())
{
    std::cout << it->second;
}
```

### Complexity

| Operation | Complexity |
| --------- | ---------: |
| Insert    |   O(log n) |
| Search    |   O(log n) |
| Delete    |   O(log n) |

Keys are maintained in sorted order.

---

# 5.2 std::set

A `set` stores unique values.

```cpp
std::set<int> numbers = {
    5, 2, 8, 2, 5
};
```

Result:

```text
2 5 8
```

Duplicates are automatically removed.

Search:

```cpp
if (numbers.find(5) != numbers.end())
{
    std::cout << "Found";
}
```

Use `set` when you need:

* Unique elements
* Ordered elements
* Efficient searching

---

# 5.3 std::multimap

Unlike `map`, multiple elements can have the same key.

```cpp
std::multimap<std::string, int> students;

students.insert({"Computer Science", 101});
students.insert({"Computer Science", 102});
students.insert({"Electrical", 103});
```

---

# 5.4 std::multiset

`multiset` allows duplicate values.

```cpp
std::multiset<int> numbers;

numbers.insert(10);
numbers.insert(10);
numbers.insert(20);
```

Result:

```text
10 10 20
```

---

# 6. Unordered Containers

Unordered containers use hashing.

Important examples:

```cpp
std::unordered_map
std::unordered_set
```

---

# 6.1 std::unordered_map

Example:

```cpp
std::unordered_map<std::string, int> ages;

ages["Alice"] = 25;
ages["Bob"] = 30;
```

Average complexity:

| Operation | Average |
| --------- | ------: |
| Insert    |    O(1) |
| Search    |    O(1) |
| Delete    |    O(1) |

Worst case can be:

```text
O(n)
```

Unlike `map`, an `unordered_map` does not maintain sorted key order.

### map vs unordered_map

```text
map
│
├── Ordered
├── Tree-based
└── O(log n)

unordered_map
│
├── Unordered
├── Hash-based
└── Average O(1)
```

Choose based on requirements.

Use `map` when ordering matters.

Use `unordered_map` when fast average lookup is more important than ordering.

---

# 6.2 std::unordered_set

Stores unique values using hashing.

```cpp
std::unordered_set<int> numbers;

numbers.insert(10);
numbers.insert(20);
numbers.insert(10);
```

The duplicate `10` is ignored.

---

# 7. Container Adaptors

Container adaptors provide specialized interfaces.

Main adaptors:

```text
stack
queue
priority_queue
```

---

# 7.1 std::stack

Stack follows:

> LIFO — Last In, First Out

```cpp
std::stack<int> s;

s.push(10);
s.push(20);
s.push(30);
```

Conceptually:

```text
   30 ← top
   20
   10
```

```cpp
s.top();
s.pop();
```

Applications:

* Undo systems
* Function call stacks
* Expression evaluation
* DFS

---

# 7.2 std::queue

Queue follows:

> FIFO — First In, First Out

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);
```

Conceptually:

```text
front                 back
  ↓                     ↓
[10] [20] [30]
```

Applications:

* Task scheduling
* BFS
* Request processing

---

# 7.3 std::priority_queue

The highest-priority element is accessible first.

```cpp
std::priority_queue<int> pq;

pq.push(10);
pq.push(50);
pq.push(20);

std::cout << pq.top();
```

Output:

```text
50
```

Useful for:

* Priority scheduling
* Dijkstra's algorithm
* Top-K problems
* Event systems

---

# 8. Container Comparison

| Container     | Ordered         | Duplicate   | Random Access | Typical Search |
| ------------- | --------------- | ----------- | ------------- | -------------- |
| vector        | Insertion order | Yes         | Yes           | O(n)           |
| array         | Insertion order | Yes         | Yes           | O(n)           |
| deque         | Insertion order | Yes         | Yes           | O(n)           |
| list          | Insertion order | Yes         | No            | O(n)           |
| map           | Sorted key      | Keys unique | No            | O(log n)       |
| multimap      | Sorted key      | Yes         | No            | O(log n)       |
| set           | Sorted          | No          | No            | O(log n)       |
| multiset      | Sorted          | Yes         | No            | O(log n)       |
| unordered_map | No              | Keys unique | No            | Avg O(1)       |
| unordered_set | No              | No          | No            | Avg O(1)       |

---

# 9. Iterators

An iterator is an object that allows us to traverse elements of a container.

Think of an iterator as a generalized pointer.

Example:

```cpp
std::vector<int> numbers = {10, 20, 30};

auto it = numbers.begin();

std::cout << *it;
```

Output:

```text
10
```

Move iterator:

```cpp
++it;

std::cout << *it;
```

Output:

```text
20
```

End iterator:

```cpp
numbers.end();
```

Important:

> `end()` does not point to the last element.

It points **one position past the last element**.

Therefore:

```cpp
*numbers.end(); // ❌ invalid
```

Correct:

```cpp
auto it = numbers.begin();

while (it != numbers.end())
{
    std::cout << *it << '\n';
    ++it;
}
```

---

# 10. Iterator Categories

STL iterators have different capabilities.

```text
Iterator
│
├── Input Iterator
├── Output Iterator
├── Forward Iterator
├── Bidirectional Iterator
└── Random Access Iterator
```

Modern C++ also introduces the concept of **contiguous iterators**.

---

## Input Iterator

Can read elements while moving forward.

---

## Output Iterator

Can write elements while moving forward.

---

## Forward Iterator

Can move forward repeatedly.

Example:

```cpp
std::forward_list
```

---

## Bidirectional Iterator

Can move:

```cpp
++
--
```

Example:

```cpp
std::list
std::set
std::map
```

---

## Random Access Iterator

Supports operations such as:

```cpp
it + 5
it - 2
it[3]
it1 - it2
```

Examples:

```cpp
vector
deque
array
```

---

# 11. Algorithms

STL provides many algorithms through:

```cpp
#include <algorithm>
```

Examples:

```text
sort
find
count
reverse
copy
remove
replace
min
max
binary_search
lower_bound
upper_bound
```

Algorithms generally work on iterator ranges:

```cpp
algorithm(begin, end);
```

Example:

```cpp
std::sort(numbers.begin(), numbers.end());
```

This separation allows algorithms to work with many different containers.

---

# 12. Sorting Algorithms

## std::sort

```cpp
std::vector<int> numbers = {
    5, 2, 8, 1, 3
};

std::sort(
    numbers.begin(),
    numbers.end()
);
```

Result:

```text
1 2 3 5 8
```

Average complexity:

```text
O(n log n)
```

---

## Descending order

```cpp
std::sort(
    numbers.begin(),
    numbers.end(),
    std::greater<int>()
);
```

---

# 13. Searching Algorithms

## std::find

```cpp
auto it = std::find(
    numbers.begin(),
    numbers.end(),
    8
);
```

Check:

```cpp
if (it != numbers.end())
{
    std::cout << "Found";
}
```

Complexity:

```text
O(n)
```

---

## std::binary_search

Requires sorted data.

```cpp
std::sort(numbers.begin(), numbers.end());

bool found = std::binary_search(
    numbers.begin(),
    numbers.end(),
    8
);
```

Complexity:

```text
O(log n)
```

---

## lower_bound

Returns the first position where a value can be inserted without breaking sorted order.

```cpp
auto it = std::lower_bound(
    numbers.begin(),
    numbers.end(),
    5
);
```

---

## upper_bound

Returns the first position after the range of elements equivalent to the given value.

---

# 14. Modification Algorithms

## reverse

```cpp
std::reverse(
    numbers.begin(),
    numbers.end()
);
```

---

## count

```cpp
int result = std::count(
    numbers.begin(),
    numbers.end(),
    5
);
```

---

## count_if

```cpp
int evenCount = std::count_if(
    numbers.begin(),
    numbers.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

---

## remove

A very important concept:

```cpp
numbers.erase(
    std::remove(
        numbers.begin(),
        numbers.end(),
        5
    ),
    numbers.end()
);
```

This is commonly called the:

> **Erase-remove idiom**

---

# 15. Numeric Algorithms

Include:

```cpp
#include <numeric>
```

## accumulate

```cpp
int sum = std::accumulate(
    numbers.begin(),
    numbers.end(),
    0
);
```

Example:

```text
10 + 20 + 30 = 60
```

---

## iota

```cpp
std::vector<int> numbers(5);

std::iota(
    numbers.begin(),
    numbers.end(),
    1
);
```

Result:

```text
1 2 3 4 5
```

---

# 16. Functors

A **functor**, or function object, is an object that behaves like a function.

It is created by defining:

```cpp
operator()
```

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

Usage:

```cpp
IsEven check;

std::cout << check(10);
```

Output:

```text
1
```

Functors can be passed to STL algorithms.

```cpp
std::count_if(
    numbers.begin(),
    numbers.end(),
    IsEven{}
);
```

---

# 17. Lambda Expressions

Modern C++ commonly uses lambdas instead of manually creating simple functors.

Example:

```cpp
auto isEven = [](int x)
{
    return x % 2 == 0;
};
```

Then:

```cpp
std::count_if(
    numbers.begin(),
    numbers.end(),
    isEven
);
```

Or directly:

```cpp
int count = std::count_if(
    numbers.begin(),
    numbers.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

Lambda syntax:

```cpp
[capture](parameters) -> return_type
{
    // body
};
```

Example:

```cpp
[](int x)
{
    return x * 2;
}
```

---

# 18. Function Objects and Callables

C++ has several types of callable objects.

```text
Callable
│
├── Function
├── Function Pointer
├── Lambda
├── Functor
└── std::function
```

Example function:

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

Lambda:

```cpp
auto operation = [](int a, int b)
{
    return a + b;
};
```

Functor:

```cpp
struct Add
{
    int operator()(int a, int b)
    {
        return a + b;
    }
};
```

---

# 19. std::function

`std::function` is a general-purpose polymorphic function wrapper.

```cpp
#include <functional>

std::function<int(int, int)> operation;

operation = [](int a, int b)
{
    return a + b;
};
```

Now:

```cpp
std::cout << operation(10, 20);
```

Output:

```text
30
```

It can store:

```text
Function
Lambda
Functor
Function pointer
```

Example:

```cpp
void execute(
    std::function<void()> callback
)
{
    callback();
}
```

Usage:

```cpp
execute([]()
{
    std::cout << "Hello";
});
```

---

# 20. Pairs and Tuples

## std::pair

Stores two values.

```cpp
std::pair<std::string, int> student;

student.first = "Alice";
student.second = 25;
```

Or:

```cpp
auto student =
    std::make_pair("Alice", 25);
```

---

## std::tuple

Stores multiple values.

```cpp
std::tuple<std::string, int, double> data;

data = {"Alice", 25, 3.75};
```

Access:

```cpp
std::get<0>(data);
std::get<1>(data);
std::get<2>(data);
```

---

# 21. std::optional

`std::optional` represents a value that may or may not exist.

```cpp
#include <optional>

std::optional<int> findValue(bool found)
{
    if (found)
        return 42;

    return std::nullopt;
}
```

Usage:

```cpp
auto result = findValue(true);

if (result.has_value())
{
    std::cout << result.value();
}
```

Modern C++ code often uses `optional` instead of special "magic" values such as:

```cpp
-1
0
nullptr
```

when the absence of a value needs to be represented explicitly.

---

# 22. Structured Bindings

Structured bindings make it easier to unpack pairs, tuples, and other suitable objects.

Example:

```cpp
std::pair<std::string, int> student{
    "Alice", 25
};

auto [name, age] = student;
```

Now:

```cpp
std::cout << name;
std::cout << age;
```

Very useful with maps:

```cpp
std::map<std::string, int> scores;

for (const auto& [name, score] : scores)
{
    std::cout << name
              << ": "
              << score
              << '\n';
}
```

---

# 23. Algorithms + Lambdas + Containers

This combination is one of the most powerful STL patterns.

Example:

```cpp
std::vector<int> numbers = {
    1, 2, 3, 4, 5, 6
};
```

Find even numbers:

```cpp
auto count = std::count_if(
    numbers.begin(),
    numbers.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

This combines:

```text
vector
 +
iterators
 +
algorithm
 +
lambda
```

The result is concise and expressive code.

---

# 24. Range-Based for Loops

Instead of manually using iterators:

```cpp
for (
    auto it = numbers.begin();
    it != numbers.end();
    ++it
)
{
    std::cout << *it;
}
```

Modern C++ allows:

```cpp
for (const auto& number : numbers)
{
    std::cout << number;
}
```

This is easier to read and should generally be preferred when you simply need to visit every element.

---

# 25. Common STL Patterns

## Pattern 1 — Find an element

```cpp
auto it = std::find(
    numbers.begin(),
    numbers.end(),
    value
);

if (it != numbers.end())
{
    // Found
}
```

---

## Pattern 2 — Sort

```cpp
std::sort(
    numbers.begin(),
    numbers.end()
);
```

---

## Pattern 3 — Sort with custom condition

```cpp
std::sort(
    numbers.begin(),
    numbers.end(),
    [](int a, int b)
    {
        return a > b;
    }
);
```

---

## Pattern 4 — Count condition

```cpp
auto result = std::count_if(
    numbers.begin(),
    numbers.end(),
    [](int x)
    {
        return x > 10;
    }
);
```

---

## Pattern 5 — Iterate through map

```cpp
for (const auto& [key, value] : data)
{
    std::cout << key
              << " "
              << value
              << '\n';
}
```

---

## Pattern 6 — Remove elements

```cpp
numbers.erase(
    std::remove_if(
        numbers.begin(),
        numbers.end(),
        [](int x)
        {
            return x < 0;
        }
    ),
    numbers.end()
);
```

---

# 26. Time Complexity

Understanding complexity is more important than memorizing container names.

Common complexities:

```text
O(1)
O(log n)
O(n)
O(n log n)
O(n²)
```

Examples:

```text
vector access              → O(1)

map search                 → O(log n)

unordered_map average      → O(1)

vector search              → O(n)

sort                       → O(n log n)
```

Always ask:

> What is the complexity of this operation?

---

# 27. Choosing the Right Container

A practical decision process:

```text
Need a general dynamic array?
        ↓
      vector

Need fixed-size storage?
        ↓
      array

Need insertion/removal at both ends?
        ↓
      deque

Need linked-list behavior?
        ↓
      list / forward_list

Need key → value?
        ↓
      map / unordered_map

Need unique values?
        ↓
      set / unordered_set

Need sorted keys?
        ↓
      map

Need average O(1) lookup?
        ↓
      unordered_map

Need LIFO?
        ↓
      stack

Need FIFO?
        ↓
      queue

Need highest-priority element?
        ↓
      priority_queue
```

The most important habit is:

> **Choose a container based on the operations your program performs most often.**

---

# 28. Common Mistakes

## Mistake 1 — Using list everywhere

Do not assume linked lists are always faster.

For many workloads:

```cpp
std::vector
```

is the better choice.

---

## Mistake 2 — Using unordered_map when ordering matters

`unordered_map` does not provide sorted key order.

If ordering is required:

```cpp
std::map
```

may be more appropriate.

---

## Mistake 3 — Dereferencing end()

Never do:

```cpp
*container.end();
```

`end()` represents one position past the last element.

---

## Mistake 4 — Invalidating iterators

Some container operations can invalidate iterators.

For example, inserting into a `vector` can cause reallocation, which may invalidate existing iterators and references.

Always understand iterator invalidation rules when modifying containers.

---

## Mistake 5 — Reimplementing STL algorithms unnecessarily

Instead of manually writing:

```cpp
for (...)
{
    if (...)
    {
        ...
    }
}
```

check whether an STL algorithm already solves the problem:

```cpp
std::find
std::count
std::count_if
std::sort
std::reverse
std::binary_search
```

---

# 29. Recommended Learning Order

For a beginner/intermediate C++ programmer, study STL in this order:

```text
1. vector
      ↓
2. array
      ↓
3. string
      ↓
4. iterators
      ↓
5. algorithms
      ↓
6. lambda expressions
      ↓
7. pair / tuple
      ↓
8. map
      ↓
9. unordered_map
      ↓
10. set
      ↓
11. unordered_set
      ↓
12. deque
      ↓
13. list
      ↓
14. stack
      ↓
15. queue
      ↓
16. priority_queue
      ↓
17. functors
      ↓
18. std::function
      ↓
19. numeric algorithms
      ↓
20. optional
```

Do not try to memorize everything at once.

Focus first on:

```text
vector
map
unordered_map
set
iterators
algorithms
lambdas
```

These will cover a huge amount of practical C++ programming.

---

# 30. Practice Projects

STL becomes much easier when you build projects.

## Project 1 — Contact Management System

Use:

```cpp
vector
string
sort
find
```

Features:

```text
Add contact
Delete contact
Search contact
Sort contacts
Display contacts
```

---

## Project 2 — Student Management System

Use:

```cpp
vector
struct/class
sort
find_if
count_if
```

Features:

```text
Add student
Delete student
Search student
Calculate average
Sort by marks
Find highest scorer
```

---

## Project 3 — Word Frequency Counter

Use:

```cpp
unordered_map
string
```

Input:

```text
hello world hello cpp world
```

Output:

```text
hello → 2
world → 2
cpp   → 1
```

---

## Project 4 — Priority Task Manager

Use:

```cpp
priority_queue
```

Features:

```text
Add task
Set priority
Process highest-priority task
Display pending tasks
```

---

## Project 5 — Inventory System

Use:

```cpp
unordered_map
vector
algorithm
```

Features:

```text
Add product
Remove product
Search product
Update stock
Sort products
Calculate total inventory
```

---

# STL Mental Model

The most important thing to remember is:

```text
                 STL
                  │
       ┌──────────┼──────────┐
       │          │          │
   Containers  Algorithms  Iterators
       │          │          │
       └──────────┼──────────┘
                  │
              Callables
                  │
          ┌───────┴───────┐
          │               │
       Functors         Lambdas
```

A typical STL program looks like:

```cpp
std::vector<int> numbers = {
    5, 2, 8, 1, 3
};

std::sort(
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
vector      → Container
begin/end   → Iterators
sort        → Algorithm
lambda      → Callable
```

That is the fundamental STL model.

---

# Final Goal

Do not aim for:

> "I memorized all STL functions."

Aim for:

> **"I understand how STL components work together, and I can choose the right container and algorithm for a problem."**

You should eventually be able to look at a problem and think:

```text
What data am I storing?
        ↓
Which container fits?
        ↓
What operations are frequent?
        ↓
What is the complexity?
        ↓
Which STL algorithm can solve it?
        ↓
Do I need an iterator?
        ↓
Do I need a lambda/functor?
```

Once this becomes natural, STL becomes a powerful foundation for larger C++ systems, including networking and Unreal Engine development.
