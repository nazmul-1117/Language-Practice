# `std::list` — Complete Guide

`std::list` is a **doubly linked list** container provided by the C++ Standard Library.

```cpp
#include <list>
```

Unlike `std::vector`, which stores elements in contiguous memory, `std::list` stores each element in a separate node connected through pointers.

Conceptually:

```text
┌────────────┐      ┌────────────┐      ┌────────────┐
│ Node       │      │ Node       │      │ Node       │
│            │      │            │      │            │
│ prev ──────┼─────→│ prev       │─────→│ prev       │
│ value: 10  │      │ value: 20  │      │ value: 30  │
│ next ──────┼─────→│ next       │─────→│ next       │
└────────────┘      └────────────┘      └────────────┘
```

The two important characteristics are:

* **Fast insertion and removal at known positions**
* **No random access**

---

# 1. Basic Syntax

```cpp
std::list<T> name;
```

Example:

```cpp
std::list<int> numbers;
```

You can use many different types:

```cpp
std::list<int> numbers;
std::list<double> prices;
std::list<std::string> names;
```

---

# 2. Creating a List

## Empty list

```cpp
std::list<int> numbers;
```

---

## List with initial values

```cpp
std::list<int> numbers = {10, 20, 30, 40};
```

Or:

```cpp
std::list<int> numbers{10, 20, 30, 40};
```

---

## List with a specific size

```cpp
std::list<int> numbers(5);
```

Result:

```text
0 0 0 0 0
```

---

## List with size and value

```cpp
std::list<int> numbers(5, 100);
```

Result:

```text
100 100 100 100 100
```

---

## Copy a list

```cpp
std::list<int> a = {1, 2, 3};

std::list<int> b = a;
```

---

## Move a list

```cpp
std::list<int> a = {1, 2, 3};

std::list<int> b = std::move(a);
```

Include:

```cpp
#include <utility>
```

---

# 3. Adding Elements

## `push_back()`

Adds an element at the end.

```cpp
std::list<int> numbers;

numbers.push_back(10);
numbers.push_back(20);
numbers.push_back(30);
```

Result:

```text
10 → 20 → 30
```

Complexity:

```text
O(1)
```

---

# 4. `push_front()`

Adds an element at the beginning.

```cpp
numbers.push_front(5);
```

Result:

```text
5 → 10 → 20 → 30
```

Complexity:

```text
O(1)
```

This is one of the major differences from `std::vector`.

---

# 5. `emplace_back()`

Constructs an object directly at the end.

```cpp
numbers.emplace_back(100);
```

For custom objects:

```cpp
std::list<Player> players;

players.emplace_back("Alice", 100);
```

---

# 6. `emplace_front()`

Constructs an object directly at the beginning.

```cpp
numbers.emplace_front(100);
```

---

# 7. `pop_back()`

Removes the last element.

```cpp
numbers.pop_back();
```

Complexity:

```text
O(1)
```

Do not call it on an empty list.

---

# 8. `pop_front()`

Removes the first element.

```cpp
numbers.pop_front();
```

Complexity:

```text
O(1)
```

---

# 9. Accessing Elements

Unlike `vector`, `std::list` does **not** support:

```cpp
numbers[0];      // ERROR
numbers.at(0);   // ERROR
```

There is no random access.

Instead, use:

```cpp
numbers.front();
numbers.back();
```

---

# 10. `front()`

Returns the first element.

```cpp
std::list<int> numbers = {10, 20, 30};

std::cout << numbers.front();
```

Output:

```text
10
```

---

# 11. `back()`

Returns the last element.

```cpp
std::cout << numbers.back();
```

Output:

```text
30
```

Calling `front()` or `back()` on an empty list is invalid.

---

# 12. `size()`

Returns the number of elements.

```cpp
std::cout << numbers.size();
```

Example:

```text
10 → 20 → 30
```

Then:

```text
size = 3
```

---

# 13. `empty()`

Checks whether the list contains no elements.

```cpp
if (numbers.empty()) {
    std::cout << "Empty";
}
```

Equivalent conceptually to:

```cpp
numbers.size() == 0
```

---

# 14. `max_size()`

Returns the theoretical maximum number of elements that the list can contain.

```cpp
std::cout << numbers.max_size();
```

The practical limit is normally determined by available memory.

---

# 15. Iterators

Lists provide **bidirectional iterators**.

## `begin()`

Points to the first element.

```cpp
auto it = numbers.begin();
```

---

## `end()`

Points one position after the last element.

```cpp
auto it = numbers.end();
```

Never dereference `end()`:

```cpp
*numbers.end(); // Invalid
```

---

# 16. Traversing a List

```cpp
for (auto it = numbers.begin(); it != numbers.end(); ++it) {
    std::cout << *it << ' ';
}
```

Example:

```text
10 20 30 40
```

---

# 17. Reverse Iterators

## `rbegin()`

Starts from the last element.

```cpp
for (auto it = numbers.rbegin();
     it != numbers.rend();
     ++it) {

    std::cout << *it << ' ';
}
```

Output:

```text
40 30 20 10
```

---

## `rend()`

Represents the position before the first element during reverse traversal.

---

# 18. Const Iterators

```cpp
numbers.cbegin();
numbers.cend();

numbers.crbegin();
numbers.crend();
```

These provide read-only iterators.

Example:

```cpp
for (auto it = numbers.cbegin();
     it != numbers.cend();
     ++it) {

    std::cout << *it;
}
```

---

# 19. Iterator Summary

| Function    | Purpose                 |
| ----------- | ----------------------- |
| `begin()`   | First element           |
| `end()`     | One past last           |
| `rbegin()`  | Last element            |
| `rend()`    | Reverse end             |
| `cbegin()`  | Const beginning         |
| `cend()`    | Const end               |
| `crbegin()` | Const reverse beginning |
| `crend()`   | Const reverse end       |

---

# 20. Range-Based For Loop

The easiest way to traverse a list:

```cpp
for (int x : numbers) {
    std::cout << x << ' ';
}
```

---

# 21. Modify Elements with References

```cpp
for (int& x : numbers) {
    x *= 2;
}
```

Every element is modified.

---

# 22. Read-Only Traversal

```cpp
for (const int& x : numbers) {
    std::cout << x << ' ';
}
```

For large objects, this avoids unnecessary copies.

---

# 23. `insert()`

Inserts elements before a specified iterator position.

```cpp
std::list<int> numbers = {10, 30};

auto it = numbers.begin();
++it;

numbers.insert(it, 20);
```

Result:

```text
10 → 20 → 30
```

---

# 24. Insert Multiple Elements

```cpp
numbers.insert(numbers.begin(), 3, 100);
```

Result:

```text
100 → 100 → 100 → ...
```

---

# 25. Insert Another Range

```cpp
std::list<int> a = {1, 2};
std::list<int> b = {3, 4};

a.insert(a.end(), b.begin(), b.end());
```

Result:

```text
1 → 2 → 3 → 4
```

---

# 26. `emplace()`

Constructs an object directly before a specified position.

```cpp
auto it = numbers.begin();

numbers.emplace(it, 100);
```

Useful for complex objects.

---

# 27. `erase()`

Removes an element at a specified iterator.

```cpp
auto it = numbers.begin();

++it;

numbers.erase(it);
```

If the list is:

```text
10 → 20 → 30
```

Result:

```text
10 → 30
```

---

# 28. Erase a Range

```cpp
auto first = numbers.begin();
auto last = numbers.end();

numbers.erase(first, last);
```

This removes the entire range.

---

# 29. Return Value of `erase()`

`erase()` returns an iterator pointing to the element after the erased element.

This makes safe iteration possible:

```cpp
for (auto it = numbers.begin();
     it != numbers.end();) {

    if (*it == 10) {
        it = numbers.erase(it);
    }
    else {
        ++it;
    }
}
```

This is an important pattern when modifying a list during traversal.

---

# 30. `clear()`

Removes every element.

```cpp
numbers.clear();
```

After:

```cpp
numbers.empty() == true
```

---

# 31. `remove()`

Removes all elements equal to a value.

```cpp
std::list<int> numbers = {
    10, 20, 10, 30, 10
};

numbers.remove(10);
```

Result:

```text
20 → 30
```

This is an important difference from `vector`.

With a list, `remove()` directly performs the removal.

---

# 32. `remove_if()`

Removes elements satisfying a condition.

```cpp
numbers.remove_if(
    [](int x) {
        return x % 2 == 0;
    }
);
```

This removes all even numbers.

---

# 33. `unique()`

Removes **consecutive duplicate elements**.

```cpp
std::list<int> numbers = {
    1, 1, 2, 2, 3, 3
};

numbers.unique();
```

Result:

```text
1 → 2 → 3
```

Important:

`unique()` only removes **consecutive** duplicates.

For:

```text
1 → 2 → 1 → 2
```

calling:

```cpp
numbers.unique();
```

does not remove anything.

---

# 34. Remove All Duplicates

If you want all duplicate values removed:

```cpp
numbers.sort();
numbers.unique();
```

Example:

```text
Before:
3 → 1 → 3 → 2 → 1

After sort:
1 → 1 → 2 → 3 → 3

After unique:
1 → 2 → 3
```

---

# 35. `sort()`

One of the most important special features of `std::list`.

```cpp
numbers.sort();
```

Example:

```text
Before:
40 → 10 → 30 → 20

After:
10 → 20 → 30 → 40
```

You should normally use the list's own:

```cpp
list.sort();
```

rather than:

```cpp
std::sort(list.begin(), list.end());
```

because `std::sort()` requires random-access iterators, while `std::list` only provides bidirectional iterators.

---

# 36. Descending Sort

```cpp
numbers.sort(std::greater<int>());
```

Result:

```text
40 → 30 → 20 → 10
```

---

# 37. Custom Sorting

```cpp
numbers.sort(
    [](int a, int b) {
        return a > b;
    }
);
```

---

# 38. Sorting Objects

```cpp
struct Player {
    std::string name;
    int score;
};
```

Then:

```cpp
std::list<Player> players;

players.sort(
    [](const Player& a, const Player& b) {
        return a.score > b.score;
    }
);
```

Players are sorted by score.

---

# 39. `reverse()`

Reverses the list.

```cpp
numbers.reverse();
```

Example:

```text
Before:
1 → 2 → 3 → 4

After:
4 → 3 → 2 → 1
```

---

# 40. `merge()`

Merges two **sorted lists**.

```cpp
std::list<int> a = {1, 3, 5};
std::list<int> b = {2, 4, 6};

a.merge(b);
```

Result:

```text
a:
1 → 2 → 3 → 4 → 5 → 6
```

After merging, `b` becomes empty.

```cpp
b.empty() == true
```

Both lists must already be sorted according to the same ordering.

---

# 41. Merge with Custom Comparator

```cpp
a.merge(
    b,
    std::greater<int>()
);
```

Both lists must be sorted in descending order.

---

# 42. Why `merge()` Is Special

`std::list::merge()` can merge nodes directly without creating a new set of nodes for the merged elements.

This is one of the operations where linked-list structure can be very useful.

---

# 43. `splice()`

`splice()` transfers elements from one list to another.

This is one of the most important unique features of `std::list`.

Example:

```cpp
std::list<int> a = {1, 2, 3};
std::list<int> b = {4, 5, 6};

a.splice(a.end(), b);
```

Result:

```text
a:
1 → 2 → 3 → 4 → 5 → 6

b:
empty
```

The nodes are transferred rather than copied.

---

# 44. Splice an Entire List at a Position

```cpp
a.splice(a.begin(), b);
```

All elements of `b` are transferred before `a.begin()`.

---

# 45. Splice One Element

```cpp
std::list<int> a = {1, 2, 3};
std::list<int> b = {4, 5, 6};

auto it = b.begin();

a.splice(a.end(), b, it);
```

Result:

```text
a:
1 → 2 → 3 → 4

b:
5 → 6
```

---

# 46. Splice a Range

```cpp
auto first = b.begin();
auto last = b.end();

a.splice(a.end(), b, first, last);
```

Moves the specified range from `b` into `a`.

---

# 47. `swap()`

Swaps two lists.

```cpp
std::list<int> a = {1, 2, 3};
std::list<int> b = {4, 5, 6};

a.swap(b);
```

Now:

```text
a:
4 → 5 → 6

b:
1 → 2 → 3
```

---

# 48. `assign()`

Replaces the contents of the list.

```cpp
std::list<int> numbers;

numbers.assign(5, 100);
```

Result:

```text
100 → 100 → 100 → 100 → 100
```

---

## Assign from a Range

```cpp
std::list<int> a = {1, 2, 3};

numbers.assign(a.begin(), a.end());
```

---

# 49. Resizing

## `resize()`

```cpp
std::list<int> numbers = {1, 2, 3};

numbers.resize(5);
```

Result:

```text
1 → 2 → 3 → 0 → 0
```

---

## Resize with a value

```cpp
numbers.resize(5, 100);
```

New elements become:

```text
100
```

---

## Shrink

```cpp
numbers.resize(2);
```

---

# 50. `get_allocator()`

Returns the allocator associated with the list.

```cpp
auto allocator = numbers.get_allocator();
```

This is an advanced feature and usually not needed for everyday C++ programming.

---

# 51. List and Algorithms

Many STL algorithms work with lists:

```cpp
std::find(
    numbers.begin(),
    numbers.end(),
    20
);
```

However, not every algorithm works with list.

For example:

```cpp
std::sort(
    numbers.begin(),
    numbers.end()
);
```

is invalid because `std::sort()` requires random-access iterators.

Use:

```cpp
numbers.sort();
```

instead.

---

# 52. `std::find()`

```cpp
auto it = std::find(
    numbers.begin(),
    numbers.end(),
    20
);
```

Check:

```cpp
if (it != numbers.end()) {
    std::cout << "Found";
}
```

Complexity:

```text
O(n)
```

---

# 53. `std::count()`

```cpp
int count = std::count(
    numbers.begin(),
    numbers.end(),
    10
);
```

Counts occurrences of `10`.

---

# 54. `std::for_each()`

```cpp
std::for_each(
    numbers.begin(),
    numbers.end(),
    [](int x) {
        std::cout << x << ' ';
    }
);
```

---

# 55. List Does Not Support Random Access

This is one of the most important concepts.

With vector:

```cpp
v[100];
```

is:

```text
O(1)
```

With list:

```cpp
list[100];
```

doesn't exist.

To reach the 100th element, the list must traverse nodes.

Conceptually:

```text
head
 ↓
Node 1 → Node 2 → Node 3 → ... → Node 100
```

Therefore:

```text
Access by position = O(n)
```

---

# 56. Finding an Element and Then Inserting

Suppose:

```cpp
std::list<int> numbers = {
    10, 20, 30, 40
};
```

You want to insert `25` before `30`.

First find `30`:

```cpp
auto it = std::find(
    numbers.begin(),
    numbers.end(),
    30
);
```

Then:

```cpp
numbers.insert(it, 25);
```

The search costs:

```text
O(n)
```

But once the iterator is known, insertion itself is:

```text
O(1)
```

This distinction is extremely important.

---

# 57. List Complexity

| Operation                |                                                                                                   Complexity |
| ------------------------ | -----------------------------------------------------------------------------------------------------------: |
| `front()`                |                                                                                                         O(1) |
| `back()`                 |                                                                                                         O(1) |
| `push_front()`           |                                                                                                         O(1) |
| `push_back()`            |                                                                                                         O(1) |
| `pop_front()`            |                                                                                                         O(1) |
| `pop_back()`             |                                                                                                         O(1) |
| `insert()` with iterator |                                                                                                         O(1) |
| `erase()` with iterator  |                                                                                                         O(1) |
| `splice()`               | O(1) for the transferred constant-size operation / range transfer depending on overload and standard version |
| `remove()`               |                                                                                                         O(n) |
| `remove_if()`            |                                                                                                         O(n) |
| `find()`                 |                                                                                                         O(n) |
| `size()`                 |                                                                                                         O(1) |
| `empty()`                |                                                                                                         O(1) |
| `sort()`                 |                                                                                                   O(n log n) |
| `reverse()`              |                                                                                                         O(n) |
| `unique()`               |                                                                                                         O(n) |
| `merge()`                |                                                                                                     O(n + m) |
| Random access            |                                                                                                Not supported |

The key idea:

> `std::list` gives fast modification **when you already have an iterator to the position**.

It does **not** make finding a position fast.

---

# 58. Iterator Invalidation

One major advantage of `std::list` is iterator stability.

Inserting or removing elements generally does not invalidate iterators to other elements.

For example:

```cpp
auto it = numbers.begin();

numbers.push_back(100);
```

`it` generally remains valid.

Similarly, erasing one element invalidates the iterator referring to that erased element, but iterators to other elements remain valid.

This makes list useful when stable references/iterators are important.

---

# 59. Memory Layout

A vector:

```text
┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │
└────┴────┴────┴────┘
```

A list:

```text
┌──────────┐
│ prev     │
│ value 10 │
│ next ────┼──────┐
└──────────┘      ↓
             ┌──────────┐
             │ prev     │
             │ value 20 │
             │ next ────┼──────┐
             └──────────┘      ↓
                          ┌──────────┐
                          │ value 30 │
                          └──────────┘
```

Every node requires additional pointer storage.

Therefore, `std::list` generally uses significantly more memory per element than `std::vector`.

---

# 60. Cache Locality

Vector:

```text
Memory:
[10][20][30][40][50]
```

Elements are next to each other.

This provides excellent cache locality.

List:

```text
[10] → [20] → [30] → [40]
```

Nodes may exist at unrelated memory addresses.

This results in poorer cache locality.

This is an important reason why `vector` can outperform `list` even when the list has theoretically cheaper insertion/erasure.

---

# 61. Vector vs List

| Feature            | `vector`       | `list`             |
| ------------------ | -------------- | ------------------ |
| Data structure     | Dynamic array  | Doubly linked list |
| Memory             | Contiguous     | Separate nodes     |
| Random access      | O(1)           | Not supported      |
| `push_back()`      | O(1) amortized | O(1)               |
| `push_front()`     | O(n)           | O(1)               |
| Middle insertion   | O(n)           | O(1) with iterator |
| Middle erase       | O(n)           | O(1) with iterator |
| Cache locality     | Excellent      | Poor               |
| Memory overhead    | Low            | Higher             |
| Iterator stability | Lower          | High               |
| Typical default    | Yes            | No                 |

---

# 62. List vs Deque

| Feature          | `list`             | `deque`        |
| ---------------- | ------------------ | -------------- |
| Random access    | No                 | O(1)           |
| `push_front()`   | O(1)               | O(1)           |
| `push_back()`    | O(1)               | O(1)           |
| Middle insertion | O(1) with iterator | O(n)           |
| Contiguous       | No                 | No             |
| Cache locality   | Poorer             | Usually better |
| Memory overhead  | High               | Moderate       |

If you need:

* Fast access by index → `deque`
* Fast modification at known middle positions → `list`

---

# 63. When Should You Use `std::list`?

Use `std::list` when you genuinely need characteristics such as:

### 1. Frequent insertion/removal in the middle

And you already have iterators pointing to the positions.

### 2. Fast insertion/removal at both ends

```cpp
push_front()
push_back()
pop_front()
pop_back()
```

### 3. Stable iterators/references

You need iterators to remain valid when other elements are inserted or removed.

### 4. Efficient node transfer

Operations such as:

```cpp
splice()
merge()
```

are useful.

---

# 64. When Should You NOT Use `std::list`?

Do not choose `list` simply because:

> "Insertion is O(1)."

If you constantly need:

```cpp
v[i]
```

then list is a poor choice.

If you frequently search for positions:

```cpp
std::find(...)
```

then the search is still:

```text
O(n)
```

If you care about memory efficiency and cache locality, vector is often better.

---

# 65. Common Mistake

### Mistake:

```cpp
std::list<int> numbers;

numbers[5];
```

This does not compile.

### Correct:

```cpp
auto it = numbers.begin();

std::advance(it, 5);

std::cout << *it;
```

But remember:

```text
std::advance() → O(n) for list
```

So this does not provide random access.

---

# 66. `std::next()` and `std::prev()`

You can move an iterator:

```cpp
auto it = std::next(numbers.begin(), 3);
```

or:

```cpp
auto it = std::prev(numbers.end(), 2);
```

Include:

```cpp
#include <iterator>
```

For `std::list`, moving an iterator multiple positions takes linear time.

---

# 67. Safe Removal While Iterating

Use the iterator returned by `erase()`:

```cpp
for (auto it = numbers.begin();
     it != numbers.end();) {

    if (*it % 2 == 0) {
        it = numbers.erase(it);
    }
    else {
        ++it;
    }
}
```

This is a very important pattern.

---

# 68. List of Strings

```cpp
std::list<std::string> names = {
    "Alice",
    "Bob",
    "Charlie"
};
```

Traversal:

```cpp
for (const auto& name : names) {
    std::cout << name << '\n';
}
```

---

# 69. List of Pairs

```cpp
std::list<std::pair<int, int>> points;

points.push_back({10, 20});
points.push_back({30, 40});
```

Structured bindings:

```cpp
for (auto [x, y] : points) {
    std::cout << x << ' ' << y << '\n';
}
```

---

# 70. List of Objects

```cpp
struct Player {
    std::string name;
    int score;
};
```

Then:

```cpp
std::list<Player> players;

players.emplace_back("Alice", 100);
players.emplace_back("Bob", 200);
```

Sort:

```cpp
players.sort(
    [](const Player& a, const Player& b) {
        return a.score > b.score;
    }
);
```

---

# 71. List of Smart Pointers

You can store smart pointers:

```cpp
std::list<std::unique_ptr<Player>> players;
```

Example:

```cpp
players.push_back(
    std::make_unique<Player>("Alice", 100)
);
```

This can be useful when objects need dynamic lifetime management.

---

# 72. List vs `forward_list`

C++ also provides:

```cpp
std::forward_list<T>
```

`forward_list` is a **singly linked list**.

Conceptually:

```text
forward_list:

10 → 20 → 30 → 40
```

Whereas:

```text
list:

10 ⇄ 20 ⇄ 30 ⇄ 40
```

`std::list` has both forward and backward links.

---

# 73. `list` vs `forward_list`

| Feature            | `list`        | `forward_list`                   |
| ------------------ | ------------- | -------------------------------- |
| Links              | Doubly linked | Singly linked                    |
| Forward traversal  | Yes           | Yes                              |
| Backward traversal | Yes           | No                               |
| `push_front()`     | Yes           | Yes                              |
| `push_back()`      | Yes           | No direct `push_back()`          |
| `size()`           | Yes           | No constant-time `size()` member |
| Memory overhead    | Higher        | Lower                            |

Use `forward_list` when you specifically need a lightweight singly linked list.

---

# 74. Complete Member Function Reference

## Constructors

```cpp
list()
list(size)
list(size, value)
list(first, last)
list(initializer_list)
list(other)
list(std::move(other))
```

---

## Element Access

```cpp
front()
back()
```

---

## Iterators

```cpp
begin()
end()
rbegin()
rend()
cbegin()
cend()
crbegin()
crend()
```

---

## Capacity

```cpp
empty()
size()
max_size()
```

---

## Modifiers

```cpp
clear()

insert()
emplace()

erase()

push_back()
emplace_back()
pop_back()

push_front()
emplace_front()
pop_front()

resize()

swap()

assign()
```

---

## List-Specific Operations

```cpp
remove()
remove_if()

sort()

merge()

splice()

unique()

reverse()
```

These list-specific operations are a major reason to understand `std::list` separately from other containers.

---

# 75. Practical Example

```cpp
#include <iostream>
#include <list>

int main() {

    std::list<int> numbers = {
        40, 10, 30, 20, 10, 50
    };

    // Add elements
    numbers.push_front(5);
    numbers.push_back(100);

    // Remove all 10s
    numbers.remove(10);

    // Sort
    numbers.sort();

    // Reverse
    numbers.reverse();

    // Print
    for (int x : numbers) {
        std::cout << x << ' ';
    }

    return 0;
}
```

---

# 76. Important Concepts Checklist

Before moving on from `std::list`, understand:

* [ ] What a doubly linked list is
* [ ] Nodes
* [ ] `prev` and `next`
* [ ] `push_back()`
* [ ] `push_front()`
* [ ] `pop_back()`
* [ ] `pop_front()`
* [ ] `emplace_back()`
* [ ] `emplace_front()`
* [ ] `insert()`
* [ ] `emplace()`
* [ ] `erase()`
* [ ] `clear()`
* [ ] `remove()`
* [ ] `remove_if()`
* [ ] `unique()`
* [ ] `sort()`
* [ ] `merge()`
* [ ] `splice()`
* [ ] `reverse()`
* [ ] `assign()`
* [ ] `resize()`
* [ ] `swap()`
* [ ] Iterators
* [ ] Bidirectional iterators
* [ ] Iterator stability
* [ ] No random access
* [ ] `std::advance()`
* [ ] `std::next()`
* [ ] `std::prev()`
* [ ] Memory overhead
* [ ] Cache locality
* [ ] List vs vector
* [ ] List vs deque
* [ ] List vs forward_list
* [ ] Complexity analysis

---

# 77. Recommended Learning Order

```text
std::list
   │
   ├── Doubly linked list
   │
   ├── Nodes
   │
   ├── Iterators
   │
   ├── push_front / push_back
   │
   ├── pop_front / pop_back
   │
   ├── insert / erase
   │
   ├── remove / remove_if
   │
   ├── sort
   │
   ├── unique
   │
   ├── reverse
   │
   ├── merge
   │
   ├── splice
   │
   ├── Iterator stability
   │
   ├── Complexity
   │
   └── vector vs list
```

---

# 78. Key Takeaway

The most important thing to remember about `std::list` is:

> **Fast insertion/erasure does not mean fast searching or random access.**

Think of the container as:

```text
                    std::list
                       │
              Doubly Linked List
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   Fast modification          No random access
          │                         │
     O(1) with iterator             │
          │                         ↓
          │                    O(n) traversal
          │
          ↓
   Iterator stability
          │
          ↓
   splice / merge / sort
```

The most important comparison is:

```text
                 VECTOR              LIST
                 ──────              ────
Access            O(1)               O(n)
Push back         O(1)*              O(1)
Push front        O(n)               O(1)
Middle insert     O(n)               O(1)*
Middle erase      O(n)               O(1)*
Memory locality   Excellent           Poor
Memory overhead   Low                 High
Iterator stability Lower              High
```

`*` means the operation is O(1) when the relevant position is already known through an iterator.

For most general-purpose workloads, **start with `std::vector`**. Choose `std::list` when its specific properties—especially stable iterators, node-based insertion/erasure, `splice()`, or `merge()`—actually solve a problem you have.
