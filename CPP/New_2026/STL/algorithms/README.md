# C++ STL Algorithms

STL algorithms are reusable functions provided by the C++ Standard Library for common operations such as:

* Sorting
* Searching
* Counting
* Reversing
* Copying
* Removing
* Replacing
* Finding minimum/maximum values

Most STL algorithms work with **iterator ranges**:

```cpp
algorithm(begin, end);
```

For example:

```cpp
std::sort(numbers.begin(), numbers.end());
```

This design allows the same algorithm to work with different STL containers because algorithms operate on iterators rather than directly depending on a specific container.

---

## 1. Required Headers

Most common STL algorithms are available through:

```cpp
#include <algorithm>
```

Numeric algorithms such as `accumulate()` and `iota()` are available through:

```cpp
#include <numeric>
```

---

# 2. Sorting Algorithms

## `std::sort()`

`std::sort()` sorts elements in ascending order by default.

### Syntax

```cpp
std::sort(begin, end);
```

### Example

```cpp
#include <algorithm>
#include <vector>

std::vector<int> numbers = {
    5, 2, 8, 1, 3
};

std::sort(numbers.begin(), numbers.end());
```

Result:

```text
1 2 3 5 8
```

Average complexity:

```text
O(n log n)
```

### Descending Order

Use `std::greater<>`:

```cpp
std::sort(
    numbers.begin(),
    numbers.end(),
    std::greater<int>()
);
```

Result:

```text
8 5 3 2 1
```

---

# 3. Searching Algorithms

## `std::find()`

`std::find()` searches for a specific value.

### Example

```cpp
auto it = std::find(
    numbers.begin(),
    numbers.end(),
    8
);
```

Check whether the value was found:

```cpp
if (it != numbers.end())
{
    std::cout << "Found";
}
```

If the iterator equals `numbers.end()`, the value was not found.

Complexity:

```text
O(n)
```

---

## `std::binary_search()`

`std::binary_search()` checks whether a value exists in a **sorted range**.

### Example

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

Important:

> The data must be sorted before using `binary_search()`.

---

## `std::lower_bound()`

Returns an iterator pointing to the first position where a value can be inserted without breaking sorted order.

```cpp
auto it = std::lower_bound(
    numbers.begin(),
    numbers.end(),
    5
);
```

Think:

```text
first position >= value
```

---

## `std::upper_bound()`

Returns the first position **after the range of elements equivalent to the given value**.

Think:

```text
first position > value
```

---

# 4. Modification and Counting Algorithms

## `std::reverse()`

Reverses the elements in a range.

```cpp
std::reverse(
    numbers.begin(),
    numbers.end()
);
```

Example:

```text
Before:
1 2 3 4 5

After:
5 4 3 2 1
```

---

## `std::count()`

Counts how many times a particular value appears.

```cpp
int result = std::count(
    numbers.begin(),
    numbers.end(),
    5
);
```

If `5` appears three times:

```text
result = 3
```

---

## `std::count_if()`

Counts elements that satisfy a condition.

Example: count even numbers.

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

The lambda provides the condition.

---

# 5. Remove Algorithm

## `std::remove()`

`std::remove()` moves elements matching a value toward the end of the range.

For a `vector`, it is commonly used with `erase()`.

### Erase-Remove Idiom

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

This is known as the:

**Erase-remove idiom**

---

# 6. Numeric Algorithms

Numeric algorithms are provided through:

```cpp
#include <numeric>
```

---

## `std::accumulate()`

Calculates an accumulated result, commonly a sum.

```cpp
int sum = std::accumulate(
    numbers.begin(),
    numbers.end(),
    0
);
```

For:

```text
10 20 30
```

Result:

```text
60
```

---

## `std::iota()`

Fills a range with sequentially increasing values.

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

# 7. Algorithms + Lambdas

One of the most useful STL patterns is combining:

```text
Container
    +
Iterator
    +
Algorithm
    +
Lambda
```

Example:

```cpp
std::vector<int> numbers = {
    1, 2, 3, 4, 5, 6
};

auto evenCount = std::count_if(
    numbers.begin(),
    numbers.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

Here:

* `vector` → stores the data
* `begin()` / `end()` → provide the range
* `count_if()` → performs the algorithm
* Lambda → defines the condition

This is a fundamental STL programming pattern.

---

# 8. Common Algorithm Patterns

## Find an element

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

## Sort

```cpp
std::sort(
    numbers.begin(),
    numbers.end()
);
```

## Custom sorting

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

## Count using a condition

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

## Remove elements

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

# 9. Algorithm Quick Reference

| Algorithm         | Purpose                     | Typical Complexity |
| ----------------- | --------------------------- | -----------------: |
| `sort()`          | Sort elements               |         O(n log n) |
| `find()`          | Find a value                |               O(n) |
| `binary_search()` | Search sorted data          |           O(log n) |
| `lower_bound()`   | First position ≥ value      |           O(log n) |
| `upper_bound()`   | First position > value      |           O(log n) |
| `reverse()`       | Reverse range               |                  — |
| `count()`         | Count a value               |               O(n) |
| `count_if()`      | Count by condition          |               O(n) |
| `remove()`        | Move matching values        |                  — |
| `accumulate()`    | Calculate accumulated value |               O(n) |
| `iota()`          | Generate sequential values  |               O(n) |

---

# 10. Important Rules

### Rule 1 — Algorithms usually work with ranges

```cpp
algorithm(container.begin(), container.end());
```

### Rule 2 — `binary_search()` requires sorted data

```cpp
std::sort(numbers.begin(), numbers.end());

std::binary_search(
    numbers.begin(),
    numbers.end(),
    value
);
```

### Rule 3 — Check iterators returned by searching algorithms

```cpp
if (it != numbers.end())
{
    // Found
}
```

### Rule 4 — Prefer STL algorithms instead of unnecessarily writing manual loops

Instead of manually implementing common operations, check whether an STL algorithm already solves the problem:

```cpp
std::find
std::count
std::count_if
std::sort
std::reverse
std::binary_search
```

---

# 11. Mental Model

When solving a problem, think:

```text
What data do I have?
        ↓
Which container stores it?
        ↓
What range do I need?
        ↓
Which algorithm solves the operation?
        ↓
Do I need a lambda/custom condition?
```

The goal is not to memorize every STL algorithm.

The goal is to understand **which algorithm to choose for a given problem**.
