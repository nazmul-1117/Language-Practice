# `std::vector` — Complete Guide

`std::vector` is a dynamic array provided by the C++ Standard Library.

It stores elements in **contiguous memory**, supports **random access**, and can automatically grow or shrink when elements are added or removed.

```cpp
#include <vector>
```

```cpp
std::vector<int> numbers;
```

---

# 1. Why `std::vector`?

A normal C-style array has a fixed size:

```cpp
int arr[5];
```

A vector can grow dynamically:

```cpp
std::vector<int> numbers;

numbers.push_back(10);
numbers.push_back(20);
numbers.push_back(30);
```

Now:

```text
numbers
┌────┬────┬────┐
│ 10 │ 20 │ 30 │
└────┴────┴────┘
  0    1    2
```

Vectors provide:

* Dynamic size
* Fast random access
* Contiguous memory
* Automatic memory management
* Compatibility with STL algorithms
* Efficient insertion/removal at the end
* Easy interaction with iterators and ranges

---

# 2. Basic Syntax

```cpp
std::vector<T> name;
```

Examples:

```cpp
std::vector<int> numbers;
std::vector<double> prices;
std::vector<char> letters;
std::vector<std::string> names;
```

---

# 3. Creating a Vector

## Empty vector

```cpp
std::vector<int> v;
```

---

## Vector with initial values

```cpp
std::vector<int> v = {10, 20, 30, 40};
```

Equivalent:

```cpp
std::vector<int> v{10, 20, 30, 40};
```

---

## Vector with a specific size

```cpp
std::vector<int> v(5);
```

Creates:

```text
0 0 0 0 0
```

---

## Vector with size and initial value

```cpp
std::vector<int> v(5, 100);
```

Result:

```text
100 100 100 100 100
```

---

## Copy another vector

```cpp
std::vector<int> a = {1, 2, 3};

std::vector<int> b = a;
```

Now:

```text
a → 1 2 3
b → 1 2 3
```

They are independent vectors.

---

## Move a vector

```cpp
std::vector<int> a = {1, 2, 3};

std::vector<int> b = std::move(a);
```

This can transfer ownership of the vector's allocated memory instead of copying every element.

Include:

```cpp
#include <utility>
```

---

# 4. Accessing Elements

## `operator[]`

```cpp
std::vector<int> v = {10, 20, 30};

std::cout << v[0];
```

Output:

```text
10
```

No bounds checking is performed.

---

## `at()`

```cpp
std::cout << v.at(0);
```

Unlike `[]`, `at()` performs bounds checking.

```cpp
v.at(10);
```

If the index is invalid, it throws:

```text
std::out_of_range
```

### `[]` vs `at()`

| Method | Bounds checking | Speed           |
| ------ | --------------- | --------------- |
| `[]`   | No              | Fast            |
| `at()` | Yes             | Slight overhead |

Use `[]` when the index is known to be valid.

Use `at()` when runtime safety is important.

---

# 5. First and Last Elements

## `front()`

Returns the first element.

```cpp
std::vector<int> v = {10, 20, 30};

std::cout << v.front();
```

Output:

```text
10
```

---

## `back()`

Returns the last element.

```cpp
std::cout << v.back();
```

Output:

```text
30
```

---

## Important

Calling `front()` or `back()` on an empty vector is invalid.

```cpp
std::vector<int> v;

v.front(); // invalid
v.back();  // invalid
```

Always make sure the vector is not empty.

---

# 6. Size

## `size()`

Returns the number of elements.

```cpp
std::vector<int> v = {10, 20, 30};

std::cout << v.size();
```

Output:

```text
3
```

Return type:

```cpp
std::size_t
```

Example:

```cpp
for (std::size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i] << '\n';
}
```

---

# 7. Checking Empty Vector

## `empty()`

```cpp
if (v.empty()) {
    std::cout << "Vector is empty";
}
```

Equivalent conceptually to:

```cpp
v.size() == 0
```

But `empty()` communicates the intention more clearly.

---

# 8. `max_size()`

Returns the theoretical maximum number of elements the vector can contain.

```cpp
std::cout << v.max_size();
```

This is generally much larger than the practical amount of memory available.

---

# 9. Adding Elements

## `push_back()`

Adds an element to the end.

```cpp
std::vector<int> v;

v.push_back(10);
v.push_back(20);
v.push_back(30);
```

Result:

```text
10 20 30
```

Complexity:

```text
O(1) amortized
```

---

# 10. `emplace_back()`

Constructs an element directly at the end of the vector.

```cpp
std::vector<std::string> names;

names.emplace_back("Alice");
```

For custom objects:

```cpp
std::vector<Player> players;

players.emplace_back("Alice", 100);
```

This can avoid an unnecessary temporary object.

### `push_back()` vs `emplace_back()`

```cpp
v.push_back(Player("Alice", 100));
```

vs.

```cpp
v.emplace_back("Alice", 100);
```

`emplace_back()` constructs the object directly inside the vector.

---

# 11. Removing Elements

## `pop_back()`

Removes the last element.

```cpp
std::vector<int> v = {10, 20, 30};

v.pop_back();
```

Result:

```text
10 20
```

Complexity:

```text
O(1)
```

Calling `pop_back()` on an empty vector is invalid.

---

# 12. `clear()`

Removes all elements.

```cpp
v.clear();
```

After:

```cpp
v.size() == 0
```

Important:

`clear()` changes the size, but normally does **not require the vector to release its allocated capacity**.

Example:

```cpp
v.clear();

std::cout << v.size();
std::cout << v.capacity();
```

The size becomes zero, while capacity may remain unchanged.

---

# 13. `erase()`

Removes one or more elements.

## Erase one element

```cpp
std::vector<int> v = {10, 20, 30, 40};

v.erase(v.begin() + 1);
```

Result:

```text
10 30 40
```

---

## Erase a range

```cpp
v.erase(v.begin() + 1, v.begin() + 3);
```

Removes:

```text
20 30
```

---

# 14. `insert()`

Inserts elements at a specific position.

```cpp
std::vector<int> v = {10, 30};

v.insert(v.begin() + 1, 20);
```

Result:

```text
10 20 30
```

---

## Insert multiple copies

```cpp
v.insert(v.begin(), 3, 100);
```

Adds three `100`s at the beginning.

---

## Insert another vector

```cpp
std::vector<int> a = {1, 2};
std::vector<int> b = {3, 4};

a.insert(a.end(), b.begin(), b.end());
```

Result:

```text
1 2 3 4
```

---

# 15. `emplace()`

Constructs an element at a specific position.

```cpp
v.emplace(v.begin(), 100);
```

Useful for complex objects.

---

# 16. Resizing

## `resize()`

Changes the number of elements.

```cpp
std::vector<int> v = {1, 2, 3};

v.resize(5);
```

Result:

```text
1 2 3 0 0
```

---

## Resize with a value

```cpp
v.resize(5, 100);
```

New elements become `100`.

---

## Shrinking

```cpp
v.resize(2);
```

If:

```text
1 2 3 4 5
```

becomes:

```text
1 2
```

---

# 17. Capacity vs Size

This is one of the most important vector concepts.

## `size()`

Number of elements currently stored.

## `capacity()`

Number of elements that can be stored in currently allocated memory without requiring another allocation.

Example:

```cpp
std::vector<int> v;

v.push_back(10);
v.push_back(20);
v.push_back(30);

std::cout << v.size();
std::cout << v.capacity();
```

You might see:

```text
size = 3
capacity = 4
```

The exact capacity growth is implementation-dependent.

---

# 18. `capacity()`

```cpp
std::cout << v.capacity();
```

Capacity is usually greater than or equal to size.

Always:

```text
size <= capacity
```

---

# 19. `reserve()`

Pre-allocates memory.

```cpp
std::vector<int> v;

v.reserve(1000);
```

Now the vector can store up to at least 1000 elements without needing to reallocate.

Important:

```cpp
reserve()
```

does **not** change the vector's size.

```cpp
v.reserve(100);

std::cout << v.size();     // 0
std::cout << v.capacity(); // at least 100
```

---

# 20. Why `reserve()` Matters

Suppose:

```cpp
std::vector<int> v;

for (int i = 0; i < 1000000; ++i) {
    v.push_back(i);
}
```

The vector may repeatedly allocate larger memory blocks and move/copy elements.

Better when the approximate size is known:

```cpp
std::vector<int> v;

v.reserve(1000000);

for (int i = 0; i < 1000000; ++i) {
    v.push_back(i);
}
```

This can significantly reduce reallocations.

---

# 21. `shrink_to_fit()`

Requests that the vector reduce capacity to fit its size.

```cpp
v.shrink_to_fit();
```

Important:

It is a **non-binding request**. The implementation may or may not reduce the capacity.

---

# 22. `data()`

Returns a pointer to the underlying contiguous memory.

```cpp
std::vector<int> v = {10, 20, 30};

int* ptr = v.data();

std::cout << ptr[0];
```

Output:

```text
10
```

Because vector elements are contiguous:

```text
v.data()
   ↓
┌────┬────┬────┐
│ 10 │ 20 │ 30 │
└────┴────┴────┘
```

This is useful when interacting with APIs that require raw pointers.

---

# 23. Iterators

A vector supports random-access iterators.

## `begin()`

Points to the first element.

```cpp
auto it = v.begin();
```

---

## `end()`

Points **one position after the last element**.

```cpp
auto it = v.end();
```

Do not dereference `end()`.

```cpp
*v.end(); // invalid
```

---

# 24. Reverse Iterators

## `rbegin()`

Points to the last element.

```cpp
for (auto it = v.rbegin(); it != v.rend(); ++it) {
    std::cout << *it << ' ';
}
```

---

## `rend()`

Points one position before the first element in reverse traversal.

---

# 25. Const Iterators

## `cbegin()`

Returns a const iterator to the beginning.

```cpp
auto it = v.cbegin();
```

---

## `cend()`

Returns a const iterator to the end.

```cpp
auto it = v.cend();
```

---

## `crbegin()` and `crend()`

Const reverse iterators.

---

# 26. Iterator Summary

| Function    | Purpose                 |
| ----------- | ----------------------- |
| `begin()`   | First element           |
| `end()`     | One past last           |
| `rbegin()`  | Last element            |
| `rend()`    | One before first        |
| `cbegin()`  | Const beginning         |
| `cend()`    | Const end               |
| `crbegin()` | Const reverse beginning |
| `crend()`   | Const reverse end       |

---

# 27. Range-Based For Loop

The easiest way to iterate:

```cpp
std::vector<int> v = {10, 20, 30};

for (int x : v) {
    std::cout << x << ' ';
}
```

---

# 28. Using References

If you want to modify elements:

```cpp
for (int& x : v) {
    x *= 2;
}
```

---

# 29. Using `const` References

For efficient read-only access:

```cpp
for (const int& x : v) {
    std::cout << x;
}
```

For small types such as `int`, copying is usually fine:

```cpp
for (int x : v)
```

For large objects:

```cpp
for (const auto& object : v)
```

is usually preferable.

---

# 30. `auto` with Vectors

Instead of:

```cpp
for (std::vector<int>::iterator it = v.begin();
     it != v.end();
     ++it)
```

Use:

```cpp
for (auto it = v.begin(); it != v.end(); ++it)
```

Or simply:

```cpp
for (auto x : v)
```

---

# 31. Searching with `std::find`

```cpp
#include <algorithm>

auto it = std::find(v.begin(), v.end(), 30);
```

Check whether found:

```cpp
if (it != v.end()) {
    std::cout << "Found";
}
```

Complexity:

```text
O(n)
```

---

# 32. Counting Elements

```cpp
int count = std::count(v.begin(), v.end(), 10);
```

Counts how many times `10` occurs.

---

# 33. `count_if()`

Count elements satisfying a condition.

```cpp
int count = std::count_if(
    v.begin(),
    v.end(),
    [](int x) {
        return x > 10;
    }
);
```

---

# 34. Sorting a Vector

```cpp
#include <algorithm>

std::sort(v.begin(), v.end());
```

Ascending order:

```text
1 2 3 4 5
```

Complexity:

```text
O(n log n)
```

---

# 35. Descending Sort

```cpp
std::sort(v.rbegin(), v.rend());
```

Or:

```cpp
std::sort(v.begin(), v.end(), std::greater<int>());
```

Requires:

```cpp
#include <functional>
```

---

# 36. Custom Sorting

```cpp
std::sort(v.begin(), v.end(),
    [](int a, int b) {
        return a > b;
    }
);
```

---

# 37. `reverse()`

```cpp
std::reverse(v.begin(), v.end());
```

Example:

```text
Before:
1 2 3 4 5

After:
5 4 3 2 1
```

---

# 38. `min_element()`

```cpp
auto it = std::min_element(v.begin(), v.end());

std::cout << *it;
```

---

# 39. `max_element()`

```cpp
auto it = std::max_element(v.begin(), v.end());

std::cout << *it;
```

---

# 40. `minmax_element()`

Find both minimum and maximum.

```cpp
auto [minIt, maxIt] =
    std::minmax_element(v.begin(), v.end());
```

---

# 41. Binary Search

For a sorted vector:

```cpp
std::sort(v.begin(), v.end());

bool found = std::binary_search(
    v.begin(),
    v.end(),
    30
);
```

Complexity:

```text
O(log n)
```

The vector must be sorted according to the same ordering.

---

# 42. `lower_bound()`

For a sorted vector:

```cpp
auto it = std::lower_bound(
    v.begin(),
    v.end(),
    30
);
```

Returns the first position where `30` could be inserted without violating sorted order.

Conceptually:

```text
1 2 4 4 7 9
      ↑
lower_bound(4)
```

---

# 43. `upper_bound()`

```cpp
auto it = std::upper_bound(
    v.begin(),
    v.end(),
    4
);
```

Returns the first element greater than `4`.

---

# 44. Finding Frequency in a Sorted Vector

```cpp
auto first = std::lower_bound(v.begin(), v.end(), 4);
auto last  = std::upper_bound(v.begin(), v.end(), 4);

int frequency = last - first;
```

---

# 45. Removing Elements

A common pattern:

```cpp
v.erase(
    std::remove(v.begin(), v.end(), 10),
    v.end()
);
```

This removes all `10`s.

This is known as the:

**Erase-Remove Idiom**

---

# 46. `remove_if()`

Remove elements according to a condition.

```cpp
v.erase(
    std::remove_if(
        v.begin(),
        v.end(),
        [](int x) {
            return x % 2 == 0;
        }
    ),
    v.end()
);
```

This removes all even numbers.

---

# 47. C++20 `std::erase`

Modern C++ provides:

```cpp
std::erase(v, 10);
```

This removes all occurrences of `10`.

---

# 48. C++20 `std::erase_if`

```cpp
std::erase_if(
    v,
    [](int x) {
        return x % 2 == 0;
    }
);
```

This is often cleaner than the traditional erase-remove pattern.

---

# 49. Duplicate Removal

One common approach:

```cpp
std::sort(v.begin(), v.end());

v.erase(
    std::unique(v.begin(), v.end()),
    v.end()
);
```

Example:

```text
Before:
1 2 2 3 3 3 4

After:
1 2 3 4
```

Important:

`std::unique()` alone does **not** change the vector's size.

---

# 50. `swap()`

Swap two vectors:

```cpp
std::vector<int> a = {1, 2};
std::vector<int> b = {3, 4};

a.swap(b);
```

Now:

```text
a → 3 4
b → 1 2
```

You can also use:

```cpp
std::swap(a, b);
```

---

# 51. `assign()`

Replace the vector's contents.

```cpp
std::vector<int> v;

v.assign(5, 100);
```

Result:

```text
100 100 100 100 100
```

---

## Assign from another range

```cpp
std::vector<int> a = {1, 2, 3};

v.assign(a.begin(), a.end());
```

---

# 52. Vector Comparison

Vectors support relational comparison.

```cpp
std::vector<int> a = {1, 2, 3};
std::vector<int> b = {1, 2, 4};

if (a < b) {
    // true
}
```

Comparison is lexicographical.

Similar to dictionary ordering.

---

# 53. 1D Vector

The most common form:

```cpp
std::vector<int> v;
```

Example:

```cpp
std::vector<int> numbers = {
    10, 20, 30, 40, 50
};
```

Access:

```cpp
numbers[0];
numbers[1];
numbers[2];
```

---

# 54. Dynamic 1D Vector

A vector can grow dynamically:

```cpp
std::vector<int> v;

int n;
std::cin >> n;

for (int i = 0; i < n; ++i) {
    int x;
    std::cin >> x;

    v.push_back(x);
}
```

---

# 55. 2D Vector

A 2D vector is a vector containing vectors.

```cpp
std::vector<std::vector<int>> matrix;
```

Example:

```cpp
std::vector<std::vector<int>> matrix = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};
```

Conceptually:

```text
1 2 3
4 5 6
7 8 9
```

Access:

```cpp
matrix[0][0]; // 1
matrix[1][2]; // 6
matrix[2][1]; // 8
```

---

# 56. Creating a Fixed-Size 2D Vector

```cpp
int rows = 3;
int cols = 4;

std::vector<std::vector<int>> matrix(
    rows,
    std::vector<int>(cols)
);
```

Creates:

```text
0 0 0 0
0 0 0 0
0 0 0 0
```

---

# 57. Initialize 2D Vector with a Value

```cpp
std::vector<std::vector<int>> matrix(
    3,
    std::vector<int>(4, 100)
);
```

Result:

```text
100 100 100 100
100 100 100 100
100 100 100 100
```

---

# 58. Traverse a 2D Vector

Traditional loops:

```cpp
for (int i = 0; i < matrix.size(); ++i) {
    for (int j = 0; j < matrix[i].size(); ++j) {
        std::cout << matrix[i][j] << ' ';
    }

    std::cout << '\n';
}
```

---

# 59. Range-Based 2D Traversal

Cleaner:

```cpp
for (const auto& row : matrix) {
    for (int value : row) {
        std::cout << value << ' ';
    }

    std::cout << '\n';
}
```

---

# 60. Jagged 2D Vector

Unlike a traditional 2D array, rows can have different sizes.

```cpp
std::vector<std::vector<int>> matrix = {
    {1, 2},
    {3, 4, 5},
    {6},
    {7, 8, 9, 10}
};
```

This is called a:

**Jagged Vector / Ragged Array**

The rows do not need to have the same number of elements.

---

# 61. Adding Rows to a 2D Vector

```cpp
std::vector<std::vector<int>> matrix;

matrix.push_back({1, 2, 3});
matrix.push_back({4, 5, 6});
```

Result:

```text
1 2 3
4 5 6
```

---

# 62. Adding Elements to a Specific Row

```cpp
matrix[0].push_back(100);
```

---

# 63. 3D Vector

A 3D vector can be created as:

```cpp
std::vector<
    std::vector<
        std::vector<int>
    >
> cube;
```

Example:

```cpp
std::vector<std::vector<std::vector<int>>> cube(
    3,
    std::vector<std::vector<int>>(
        4,
        std::vector<int>(5)
    )
);
```

Access:

```cpp
cube[x][y][z];
```

Think:

```text
Layer
 ├── Row
 │    └── Column
```

---

# 64. N-Dimensional Vectors

The same concept can be extended:

```cpp
std::vector<std::vector<int>>
```

2D

```cpp
std::vector<std::vector<std::vector<int>>>
```

3D

Higher dimensions are possible but can become difficult to read and inefficient in some situations.

For high-dimensional numerical data, specialized data structures may sometimes be more appropriate.

---

# 65. Vector of Strings

```cpp
std::vector<std::string> names = {
    "Alice",
    "Bob",
    "Charlie"
};
```

Access:

```cpp
std::cout << names[0];
```

---

# 66. Vector of Pairs

```cpp
std::vector<std::pair<int, int>> points;

points.push_back({10, 20});
points.push_back({30, 40});
```

Access:

```cpp
std::cout << points[0].first;
std::cout << points[0].second;
```

With structured bindings:

```cpp
for (auto [x, y] : points) {
    std::cout << x << ' ' << y << '\n';
}
```

---

# 67. Vector of Structs

```cpp
struct Player {
    std::string name;
    int score;
};

std::vector<Player> players;

players.push_back({"Alice", 100});
players.push_back({"Bob", 200});
```

Access:

```cpp
std::cout << players[0].name;
std::cout << players[0].score;
```

---

# 68. Vector of Objects

```cpp
std::vector<Player> players;

players.emplace_back("Alice", 100);
players.emplace_back("Bob", 200);
```

This is especially useful in game development and Unreal Engine-style data structures.

---

# 69. Vector of Pointers

Possible:

```cpp
std::vector<Player*> players;
```

But raw pointers require careful lifetime management.

Modern C++ often prefers smart pointers when ownership is involved:

```cpp
std::vector<std::unique_ptr<Player>> players;
```

or:

```cpp
std::vector<std::shared_ptr<Player>> players;
```

Use smart pointers when the vector owns or shares ownership of dynamically allocated objects.

---

# 70. Vector of `bool`

There is a special case:

```cpp
std::vector<bool>
```

`std::vector<bool>` is a specialized implementation that stores boolean values in a packed representation rather than behaving exactly like a normal `std::vector<T>`.

Example:

```cpp
std::vector<bool> flags(10);

flags[0] = true;
```

Be aware that it has special behavior compared with ordinary vectors.

---

# 71. Vector with `const`

You can create a read-only vector:

```cpp
const std::vector<int> v = {1, 2, 3};
```

You cannot modify it:

```cpp
// v.push_back(4); // Error
```

---

# 72. Passing Vector to Functions

## Pass by value

```cpp
void print(std::vector<int> v) {
}
```

This copies the entire vector.

Usually avoid this when you only need to read it.

---

## Pass by const reference

```cpp
void print(const std::vector<int>& v) {
}
```

No copy.

This is the common choice for read-only access.

---

## Pass by reference

```cpp
void modify(std::vector<int>& v) {
    v.push_back(100);
}
```

Allows modification.

---

# 73. Returning a Vector from a Function

```cpp
std::vector<int> createNumbers() {
    return {1, 2, 3, 4, 5};
}
```

Modern C++ can efficiently return vectors using move semantics and return-value optimization.

---

# 74. Vector and Memory

Vector elements are stored contiguously.

Example:

```cpp
std::vector<int> v = {10, 20, 30, 40};
```

Conceptually:

```text
Address
 ↓
┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │
└────┴────┴────┴────┘
```

This provides:

```cpp
v[i]
```

in constant time.

---

# 75. Vector Reallocation

Suppose:

```cpp
std::vector<int> v;

v.push_back(10);
v.push_back(20);
```

Eventually capacity may become insufficient.

The vector then:

1. Allocates a larger memory block.
2. Moves/copies existing elements.
3. Destroys old elements if necessary.
4. Releases old memory.
5. Continues operation.

This process is called:

**Reallocation**

---

# 76. Iterator Invalidation

Reallocation is important because references, pointers, and iterators to vector elements may become invalid.

Example:

```cpp
auto it = v.begin();

v.push_back(100);
```

If `push_back()` causes reallocation, `it` may no longer be valid.

Therefore, be careful when storing:

* Iterators
* References
* Pointers

to vector elements while modifying the vector.

---

# 77. Insert and Erase Invalidation

Insertion and erasure can also invalidate iterators/references depending on where they occur and whether reallocation happens.

General rule:

> Modifying a vector can invalidate references, pointers, and iterators to its elements.

Always check the specific operation's invalidation rules when writing performance-critical or complex iterator code.

---

# 78. Vector Complexity

| Operation         |           Complexity |
| ----------------- | -------------------: |
| Access `v[i]`     |                 O(1) |
| `at()`            |                 O(1) |
| `front()`         |                 O(1) |
| `back()`          |                 O(1) |
| `push_back()`     |       O(1) amortized |
| `emplace_back()`  |       O(1) amortized |
| `pop_back()`      |                 O(1) |
| `insert()` at end |       O(1) amortized |
| `insert()` middle |                 O(n) |
| `erase()` end     |                 O(1) |
| `erase()` middle  |                 O(n) |
| `find`            |                 O(n) |
| `sort`            |           O(n log n) |
| `binary_search`   |             O(log n) |
| `lower_bound`     |             O(log n) |
| `clear()`         |                 O(n) |
| `resize()`        | Depends on operation |
| `swap()`          |                 O(1) |

---

# 79. Why Vector Is Usually the Default Container

For many problems, start with:

```cpp
std::vector<T>
```

because it provides:

* Excellent cache locality
* Random access
* Simple API
* Dynamic size
* Good performance
* Compatibility with STL algorithms
* Contiguous storage

Unless you have a specific reason to choose another container, `vector` is often a strong default.

---

# 80. Vector vs Array

### `std::array`

```cpp
std::array<int, 5> arr;
```

Fixed size.

### `std::vector`

```cpp
std::vector<int> v;
```

Dynamic size.

| Feature       | `array` | `vector`           |
| ------------- | ------- | ------------------ |
| Size          | Fixed   | Dynamic            |
| Memory        | Inline  | Dynamic allocation |
| Random access | Yes     | Yes                |
| Resize        | No      | Yes                |
| `push_back()` | No      | Yes                |
| Contiguous    | Yes     | Yes                |

---

# 81. Vector vs List

| Feature          | `vector`       | `list`               |
| ---------------- | -------------- | -------------------- |
| Memory           | Contiguous     | Non-contiguous nodes |
| Random access    | O(1)           | O(n)                 |
| `push_back()`    | O(1) amortized | O(1)                 |
| Middle insertion | O(n)           | O(1) with iterator   |
| Cache locality   | Excellent      | Poorer               |
| Typical default  | Yes            | No                   |

Do not automatically choose `list` just because you see many insertions.

In many real-world workloads, `vector` performs better due to memory locality.

---

# 82. Vector vs Deque

| Feature           | `vector`       | `deque` |
| ----------------- | -------------- | ------- |
| Random access     | O(1)           | O(1)    |
| `push_back()`     | O(1) amortized | O(1)    |
| `push_front()`    | O(n)           | O(1)    |
| Contiguous memory | Yes            | No      |
| Cache locality    | Excellent      | Good    |

If you frequently need insertion/removal at both ends, consider:

```cpp
std::deque
```

---

# 83. Common Vector Patterns

## Read input

```cpp
int n;
std::cin >> n;

std::vector<int> v(n);

for (int& x : v) {
    std::cin >> x;
}
```

---

## Find maximum

```cpp
int mx = *std::max_element(v.begin(), v.end());
```

---

## Find minimum

```cpp
int mn = *std::min_element(v.begin(), v.end());
```

---

## Sort

```cpp
std::sort(v.begin(), v.end());
```

---

## Reverse

```cpp
std::reverse(v.begin(), v.end());
```

---

## Remove duplicates

```cpp
std::sort(v.begin(), v.end());

v.erase(
    std::unique(v.begin(), v.end()),
    v.end()
);
```

---

## Check if value exists

```cpp
if (std::find(v.begin(), v.end(), x) != v.end()) {
    // Found
}
```

---

# 84. Vector with Lambda

```cpp
std::vector<int> v = {1, 2, 3, 4, 5};

std::for_each(
    v.begin(),
    v.end(),
    [](int x) {
        std::cout << x << '\n';
    }
);
```

---

# 85. Transform a Vector

```cpp
std::vector<int> v = {1, 2, 3, 4};

std::transform(
    v.begin(),
    v.end(),
    v.begin(),
    [](int x) {
        return x * 2;
    }
);
```

Result:

```text
2 4 6 8
```

---

# 86. Copy a Vector

```cpp
std::vector<int> a = {1, 2, 3};
std::vector<int> b;

b = a;
```

Or:

```cpp
std::vector<int> b(a);
```

---

# 87. Move a Vector

```cpp
std::vector<int> a = {1, 2, 3};

std::vector<int> b = std::move(a);
```

Useful when transferring a large vector instead of copying it.

After moving from `a`, do not assume its old contents remain.

The moved-from vector remains valid, but its exact contents are not something you should rely on.

---

# 88. Empty Vector Safety

Avoid:

```cpp
v.front();
v.back();
v[0];
```

without knowing whether:

```cpp
v.empty()
```

is false.

Safer:

```cpp
if (!v.empty()) {
    std::cout << v.front();
}
```

---

# 89. Common Mistake: `reserve()` vs `resize()`

These are very different.

```cpp
v.reserve(10);
```

Means:

```text
Capacity >= 10
Size = unchanged
```

While:

```cpp
v.resize(10);
```

Means:

```text
Size = 10
```

Example:

```cpp
std::vector<int> v;

v.reserve(10);

v[0] = 100; // WRONG
```

Why?

Because the size is still zero.

Correct:

```cpp
v.resize(10);

v[0] = 100;
```

Or:

```cpp
v.push_back(100);
```

---

# 90. Common Mistake: Using `size()` as `int`

This is common:

```cpp
for (int i = 0; i < v.size(); ++i)
```

It often works, but `size()` returns `std::size_t`.

Better:

```cpp
for (std::size_t i = 0; i < v.size(); ++i)
```

Or simply:

```cpp
for (std::size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i];
}
```

Or use range-based loops when an index is unnecessary.

---

# 91. Common Mistake: Modifying While Iterating

Be careful:

```cpp
for (auto it = v.begin(); it != v.end(); ++it) {
    if (*it == 10) {
        v.erase(it);
    }
}
```

After erasing, the iterator may become invalid.

A safer pattern:

```cpp
for (auto it = v.begin(); it != v.end();) {
    if (*it == 10) {
        it = v.erase(it);
    } else {
        ++it;
    }
}
```

Or use:

```cpp
std::erase(v, 10);
```

in C++20.

---

# 92. Common Mistake: Using `vector` for Everything

Vector is an excellent default, but not always the best choice.

Consider other containers when you specifically need:

* Fast insertion at front → `deque`
* Constant-time insertion/erase with a valid iterator → `list`
* Key-value lookup → `map` / `unordered_map`
* Unique ordered values → `set`
* Stack behavior → `stack`
* Queue behavior → `queue`
* Priority-based access → `priority_queue`

The goal is not:

> "Always use vector."

The goal is:

> "Know why vector is or isn't the right container."

---

# 93. Complete Vector Function Reference

## Constructors

```cpp
vector()
vector(size)
vector(size, value)
vector(first, last)
vector(initializer_list)
vector(other)
vector(std::move(other))
```

---

## Element Access

```cpp
operator[]
at()
front()
back()
data()
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
reserve()
capacity()
shrink_to_fit()
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
resize()
swap()
assign()
```

---

## C++20 Non-Member Erasure

```cpp
std::erase()
std::erase_if()
```

---

# 94. Important Vector Concepts Checklist

Before moving on from `vector`, you should understand:

* [ ] What `std::vector` is
* [ ] Dynamic arrays
* [ ] Contiguous memory
* [ ] Random access
* [ ] `size()`
* [ ] `capacity()`
* [ ] `reserve()`
* [ ] `resize()`
* [ ] `shrink_to_fit()`
* [ ] `push_back()`
* [ ] `emplace_back()`
* [ ] `pop_back()`
* [ ] `insert()`
* [ ] `erase()`
* [ ] `clear()`
* [ ] `assign()`
* [ ] `swap()`
* [ ] `front()`
* [ ] `back()`
* [ ] `at()`
* [ ] `data()`
* [ ] Iterators
* [ ] Reverse iterators
* [ ] Iterator invalidation
* [ ] Reallocation
* [ ] 1D vectors
* [ ] 2D vectors
* [ ] Jagged vectors
* [ ] 3D vectors
* [ ] Vector of strings
* [ ] Vector of pairs
* [ ] Vector of structs/classes
* [ ] Vector of smart pointers
* [ ] `vector<bool>`
* [ ] STL algorithms with vector
* [ ] Sorting
* [ ] Searching
* [ ] Binary search
* [ ] `lower_bound()`
* [ ] `upper_bound()`
* [ ] Remove-erase idiom
* [ ] `std::erase()`
* [ ] `std::erase_if()`
* [ ] `unique()`
* [ ] Passing vectors to functions
* [ ] Returning vectors
* [ ] Move semantics
* [ ] Vector memory behavior
* [ ] Vector complexity
* [ ] Vector vs array
* [ ] Vector vs list
* [ ] Vector vs deque

---

# 95. Recommended Learning Order

Learn vector in this order:

```text
1. Creating vectors
       ↓
2. Accessing elements
       ↓
3. push_back / pop_back
       ↓
4. size / empty
       ↓
5. insert / erase
       ↓
6. Iterators
       ↓
7. capacity / reserve / resize
       ↓
8. Reallocation
       ↓
9. Iterator invalidation
       ↓
10. STL algorithms
       ↓
11. 1D vectors
       ↓
12. 2D vectors
       ↓
13. Jagged vectors
       ↓
14. 3D vectors
       ↓
15. Vector of objects
       ↓
16. Performance & complexity
       ↓
17. Vector vs other containers
```

---

# 96. Practical Exercises

## Beginner

1. Create a vector of 10 integers.
2. Take input from the user.
3. Print all elements.
4. Find the sum.
5. Find the maximum.
6. Find the minimum.
7. Count even numbers.
8. Reverse the vector.
9. Sort the vector.
10. Search for a specific value.

---

## Intermediate

1. Remove duplicates.
2. Find the second-largest element.
3. Rotate the vector.
4. Merge two vectors.
5. Find common elements.
6. Use `lower_bound()`.
7. Use `upper_bound()`.
8. Implement frequency counting.
9. Sort using a custom comparator.
10. Use `erase()` safely while iterating.

---

## 2D Vector Practice

1. Create a matrix.
2. Input a matrix.
3. Print a matrix.
4. Find row sums.
5. Find column sums.
6. Find the maximum element.
7. Transpose a matrix.
8. Search for an element.
9. Rotate a matrix.
10. Implement matrix multiplication.

---

# 97. Vector Mental Model

Think of `std::vector` as:

```text
                std::vector
                     │
        ┌────────────┴────────────┐
        │                         │
      SIZE                    CAPACITY
        │                         │
  elements used          allocated storage
        │                         │
        └────────────┬────────────┘
                     │
             contiguous memory
                     │
        ┌────┬────┬────┬────┐
        │  A │  B │  C │  D │
        └────┴────┴────┴────┘
          0    1    2    3
```

The most important concepts are:

```text
vector
  │
  ├── dynamic size
  │
  ├── contiguous memory
  │
  ├── O(1) random access
  │
  ├── push_back()
  │
  ├── capacity()
  │
  ├── reserve()
  │
  ├── reallocation
  │
  ├── iterators
  │
  ├── iterator invalidation
  │
  ├── STL algorithms
  │
  ├── 1D
  │
  ├── 2D
  │
  ├── 3D
  │
  └── vector of objects
```

---

# 98. Key Takeaway

`std::vector` should be one of the first containers you master in modern C++.

Do not just memorize its functions.

Understand **why it works the way it does**:

```text
Contiguous Memory
       ↓
Fast Random Access
       ↓
Dynamic Growth
       ↓
Capacity Management
       ↓
Reallocation
       ↓
Iterator Invalidation
       ↓
STL Algorithms
       ↓
Performance Decisions
```

Once these concepts are clear, `std::vector` becomes more than just a dynamic array—it becomes the foundation for understanding much of the C++ Standard Library and efficient data handling.
