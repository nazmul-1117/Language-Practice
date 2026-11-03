# C++ STL — Iterators

## 1. What is an Iterator?

An **iterator** is an object used to **traverse elements of a container**.

The easiest way to understand an iterator is:

> **An iterator is like a generalized pointer.**

Your STL material uses exactly this mental model.

For example:

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {

    vector<int> numbers = {10, 20, 30};

    vector<int>::iterator it = numbers.begin();

    cout << *it;
}
```

Output:

```text
10
```

Here:

```text
numbers
   ↓
[10] [20] [30]
 ↑
it
```

`it` points to `10`.

---

# 2. Why Do We Need Iterators?

Different containers have different internal structures.

For example:

```text
vector
list
set
map
deque
```

Their memory organization is different.

Instead of creating different algorithms for every container, STL uses iterators as a common interface.

The basic STL idea is:

```text
Container
    +
Iterator
    +
Algorithm
    +
Callable
    =
Generic C++ Code
```

This separation is one of the central ideas of STL.

For example:

```cpp
sort(v.begin(), v.end());
```

`sort()` does not need to know that `v` is specifically a `vector`.

It works with the iterators provided by the container.

---

# 3. `begin()`

`begin()` returns an iterator pointing to the **first element**.

```cpp
vector<int> v = {10, 20, 30};

auto it = v.begin();
```

Conceptually:

```text
[10] [20] [30]
 ↑
begin()
```

Therefore:

```cpp
cout << *it;
```

Output:

```text
10
```

---

# 4. `end()`

`end()` returns an iterator pointing **one position after the last element**.

```cpp
vector<int> v = {10, 20, 30};

auto it = v.end();
```

Conceptually:

```text
[10] [20] [30] [END]
                ↑
               end()
```

Very important:

```cpp
*v.end();   // ❌ invalid
```

`end()` does **not** point to the last element. It points one position past it.

Correct:

```cpp
cout << *(v.end() - 1);
```

for a random-access container such as `vector`.

Output:

```text
30
```

---

# 5. Basic Iterator Traversal

```cpp
vector<int> v = {10, 20, 30, 40};

auto it = v.begin();

while(it != v.end()) {

    cout << *it << " ";

    ++it;
}
```

Output:

```text
10 20 30 40
```

The pattern is:

```text
initialize
    ↓
begin()
    ↓
check != end()
    ↓
use *it
    ↓
++it
    ↓
repeat
```

---

# 6. Dereferencing an Iterator

The `*` operator is used to access the element pointed to by an iterator.

```cpp
vector<int> v = {10, 20, 30};

auto it = v.begin();

cout << *it;
```

Output:

```text
10
```

Move:

```cpp
++it;
```

Now:

```text
[10] [20] [30]
      ↑
      it
```

Therefore:

```cpp
cout << *it;
```

Output:

```text
20
```

---

# 7. Changing an Element Through an Iterator

Iterators can also be used to modify elements when the iterator is not const.

```cpp
vector<int> v = {10, 20, 30};

auto it = v.begin();

*it = 100;
```

Now:

```text
100 20 30
```

So:

```cpp
*it
```

can behave similarly to a normal pointer:

```cpp
int x = 10;

int* p = &x;

*p = 20;
```

---

# 8. `++it`

Moves the iterator to the next element.

```cpp
vector<int> v = {10, 20, 30};

auto it = v.begin();

cout << *it << endl;

++it;

cout << *it << endl;
```

Output:

```text
10
20
```

---

# 9. `--it`

Not every iterator supports `--`.

Bidirectional and stronger iterators do.

```cpp
auto it = v.end();

--it;

cout << *it;
```

Output:

```text
30
```

---

# 10. Iterator Operators

Depending on iterator category, different operations are available.

Common operations:

```cpp
*it
++it
--it
it + n
it - n
it += n
it -= n
it[n]
it1 - it2
it1 == it2
it1 != it2
it1 < it2
it1 > it2
```

But **not every iterator supports every operation**.

This is why iterator categories are important.

---

# 11. Iterator Categories

STL iterator categories are traditionally:

```text
Iterator
   │
   ├── Input Iterator
   │
   ├── Output Iterator
   │
   ├── Forward Iterator
   │
   ├── Bidirectional Iterator
   │
   └── Random Access Iterator
```

Modern C++ also introduces:

```text
Contiguous Iterator
```

Your uploaded STL material follows this same classification.

---

# 12. Iterator Capability Hierarchy

A useful way to remember the hierarchy:

```text
Input
  ↑
Forward
  ↑
Bidirectional
  ↑
Random Access
  ↑
Contiguous
```

But be careful:

**Output iterator is a separate write-oriented category**, not simply a stronger version of input iterator.

Think:

```text
                 Iterator
                /        \
             Input      Output
               |
           Forward
               |
        Bidirectional
               |
        Random Access
               |
          Contiguous
```

---

# 13. Input Iterator

An **Input Iterator** can:

* Read elements
* Move forward
* Be incremented
* Be compared for equality/inequality
* Be dereferenced for reading

Basic operations:

```cpp
*it
++it
it == other
it != other
```

Conceptually:

```text
→ → → →
```

It is mainly used for **single-pass reading**.

Example use:

```cpp
istream_iterator<int>
```

---

# 14. Output Iterator

An **Output Iterator** is used mainly to **write elements**.

Example:

```cpp
*it = value;
```

It supports forward movement.

A common STL example is:

```cpp
back_inserter()
```

which we will discuss later.

Output iterators are particularly useful when algorithms generate output.

---

# 15. Forward Iterator

A Forward Iterator can:

* Read
* Write when permitted
* Move forward
* Move forward repeatedly
* Be used in multi-pass algorithms

Basic movement:

```cpp
++it;
```

but not:

```cpp
--it;    // ❌
```

and generally not:

```cpp
it + 5;  // ❌
```

Example container:

```cpp
forward_list<int>
```

Your STL material specifically gives `std::forward_list` as an example.

---

# 16. Bidirectional Iterator

A Bidirectional Iterator supports movement in **both directions**.

```cpp
++it;
--it;
```

It can move:

```text
← ← ←
→ → →
```

Examples:

```cpp
list<int>
set<int>
map<int, int>
```

Your source specifically lists `list`, `set`, and `map`.

But it does not support:

```cpp
it + 5
```

because it does not provide random-access movement.

---

# 17. Random Access Iterator

Random Access Iterators support direct movement to arbitrary positions.

For example:

```cpp
it + 5
it - 2
it += 3
it -= 2
it[3]
it1 - it2
```

Your source explicitly lists these operations.

Examples:

```cpp
vector
deque
array
```

---

# 18. Random Access Example

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};

auto it = v.begin();

cout << *(it + 3);
```

Output:

```text
40
```

Because:

```text
begin()
 ↓
10 20 30 40 50
          ↑
        +3
```

---

# 19. Iterator Difference

Random-access iterators can be subtracted.

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};

auto first = v.begin();
auto last = v.end();

cout << last - first;
```

Output:

```text
5
```

This gives the number of elements in the range.

---

# 20. `it[n]`

Random-access iterators support:

```cpp
it[n]
```

Example:

```cpp
vector<int> v = {10, 20, 30, 40};

auto it = v.begin();

cout << it[2];
```

Output:

```text
30
```

This is conceptually similar to:

```cpp
v[2]
```

---

# 21. Contiguous Iterator

Modern C++ introduces the **contiguous iterator** category.

A contiguous iterator guarantees that successive elements are stored contiguously in memory.

Examples include iterators for:

```cpp
vector
array
```

and ordinary pointers into contiguous arrays.

Conceptually:

```text
Memory:

1000 → 10
1004 → 20
1008 → 30
1012 → 40
```

The elements occupy consecutive memory locations.

This is stronger than simply being random-access.

---

# 22. Iterator Category Comparison

| Feature           | Input        | Output       | Forward      | Bidirectional | Random Access | Contiguous |
| ----------------- | ------------ | ------------ | ------------ | ------------- | ------------- | ---------- |
| Read              | ✓            | Usually no   | ✓            | ✓             | ✓             | ✓          |
| Write             | Limited      | ✓            | ✓            | ✓             | ✓             | ✓          |
| `++`              | ✓            | ✓            | ✓            | ✓             | ✓             | ✓          |
| `--`              | ❌            | ❌            | ❌            | ✓             | ✓             | ✓          |
| `+ n`             | ❌            | ❌            | ❌            | ❌             | ✓             | ✓          |
| `- n`             | ❌            | ❌            | ❌            | ❌             | ✓             | ✓          |
| `it[n]`           | ❌            | ❌            | ❌            | ❌             | ✓             | ✓          |
| `it1-it2`         | ❌            | ❌            | ❌            | ❌             | ✓             | ✓          |
| Contiguous memory | Not required | Not required | Not required | Not required  | Not required  | ✓          |

---

# 23. Containers and Iterator Categories

A useful practical table:

| Container      | Typical Iterator Category  |
| -------------- | -------------------------- |
| `vector`       | Random Access / Contiguous |
| `array`        | Random Access / Contiguous |
| `deque`        | Random Access              |
| `list`         | Bidirectional              |
| `forward_list` | Forward                    |
| `set`          | Bidirectional              |
| `multiset`     | Bidirectional              |
| `map`          | Bidirectional              |
| `multimap`     | Bidirectional              |

This explains why some algorithms work on some containers but not others.

---

# 24. Why `sort()` Doesn't Work on `list`

Consider:

```cpp
list<int> l = {5, 2, 4, 1};
```

This does not work:

```cpp
sort(l.begin(), l.end()); // ❌
```

because `std::sort` requires random-access iterators.

But `list` provides bidirectional iterators.

Instead:

```cpp
l.sort();
```

Use the container's own sorting member function.

---

# 25. Iterator Requirements of Algorithms

Different algorithms require different iterator capabilities.

For example:

```cpp
find()
```

can work with weaker iterators.

But:

```cpp
sort()
```

requires random-access iterators.

Conceptually:

```text
find()
 ↓
Input/Forward compatible

sort()
 ↓
Random Access required
```

This is why understanding iterator categories is important.

---

# 26. `std::advance()`

`advance()` moves an iterator forward or backward by a specified number of positions.

Header:

```cpp
#include <iterator>
```

Syntax:

```cpp
advance(it, n);
```

Example:

```cpp
vector<int> v = {10, 20, 30, 40, 50};

auto it = v.begin();

advance(it, 3);

cout << *it;
```

Output:

```text
40
```

---

# 27. Why Use `advance()`?

Suppose you have a `list`.

You cannot do:

```cpp
it + 3; // ❌
```

because list iterators are bidirectional.

Instead:

```cpp
advance(it, 3);
```

works.

Example:

```cpp
list<int> l = {10, 20, 30, 40, 50};

auto it = l.begin();

advance(it, 3);

cout << *it;
```

Output:

```text
40
```

---

# 28. `advance()` Complexity

This is important.

For random-access iterators:

```text
O(1)
```

For bidirectional/forward iterators:

```text
O(n)
```

Why?

For a vector:

```text
it + 100
```

can jump directly.

For a list, it must move:

```text
→ → → → → → → ...
```

one node at a time.

---

# 29. `std::next()`

`next()` returns a new iterator advanced by `n`.

Syntax:

```cpp
next(it, n)
```

Example:

```cpp
vector<int> v = {10, 20, 30, 40, 50};

auto it = next(v.begin(), 2);

cout << *it;
```

Output:

```text
30
```

Important difference:

```cpp
advance(it, 2);
```

changes `it`.

But:

```cpp
next(it, 2);
```

returns another iterator.

---

# 30. `advance()` vs `next()`

| Function    | Changes original iterator? | Returns iterator |
| ----------- | -------------------------- | ---------------- |
| `advance()` | Yes                        | No               |
| `next()`    | No                         | Yes              |
| `prev()`    | No                         | Yes              |

Example:

```cpp
auto it2 = next(it, 3);
```

---

# 31. `std::prev()`

`prev()` returns an iterator moved backward.

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};

auto it = prev(v.end());

cout << *it;
```

Output:

```text
50
```

Another example:

```cpp
auto it = prev(v.end(), 2);
```

Output:

```text
40
```

Requires an iterator that supports backward movement.

---

# 32. `std::distance()`

`distance()` calculates the number of positions between two iterators.

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};

auto first = v.begin();
auto last = v.end();

cout << distance(first, last);
```

Output:

```text
5
```

---

# 33. `distance()` Complexity

For random-access iterators:

```text
O(1)
```

For forward/bidirectional iterators:

```text
O(n)
```

Example:

```cpp
distance(l.begin(), l.end());
```

for a `list` requires traversal.

---

# 34. `advance`, `next`, `prev`, `distance`

Remember:

```text
advance()
→ Move iterator

next()
→ Get next iterator

prev()
→ Get previous iterator

distance()
→ Count distance
```

---

# 35. Reverse Iterators

STL also provides **reverse iterators**.

They allow traversal from the end toward the beginning.

Functions:

```cpp
rbegin()
rend()
```

Example:

```cpp
vector<int> v = {
    10, 20, 30, 40
};

for(auto it = v.rbegin();
    it != v.rend();
    ++it)
{
    cout << *it << " ";
}
```

Output:

```text
40 30 20 10
```

---

# 36. `rbegin()` and `rend()`

Normal:

```text
begin()                    end()
  ↓                          ↓
[10] [20] [30] [40] [END]
```

Reverse:

```text
rend()                    rbegin()
  ↓                          ↓
[END] [10] [20] [30] [40]
```

More intuitively:

```text
rbegin()
   ↓
40 → 30 → 20 → 10
                  ↑
                rend()
```

---

# 37. `reverse_iterator`

You can explicitly declare a reverse iterator:

```cpp
vector<int>::reverse_iterator it;
```

But usually:

```cpp
auto it = v.rbegin();
```

is cleaner.

---

# 38. `const_iterator`

A `const_iterator` allows you to read elements but not modify them through the iterator.

```cpp
vector<int> v = {10, 20, 30};

vector<int>::const_iterator it = v.begin();

cout << *it;
```

But:

```cpp
*it = 100; // ❌
```

This prevents modification through the iterator.

---

# 39. `cbegin()` and `cend()`

Modern C++ provides:

```cpp
cbegin()
cend()
```

Example:

```cpp
vector<int> v = {10, 20, 30};

auto it = v.cbegin();

cout << *it;
```

You cannot modify:

```cpp
*it = 100; // ❌
```

Useful when you explicitly want read-only traversal.

---

# 40. `crbegin()` and `crend()`

For const reverse traversal:

```cpp
crbegin()
crend()
```

Example:

```cpp
vector<int> v = {10, 20, 30};

for(auto it = v.crbegin();
    it != v.crend();
    ++it)
{
    cout << *it << " ";
}
```

Output:

```text
30 20 10
```

---

# 41. Iterator Types Summary

```text
begin()
→ normal iterator

end()
→ past-the-end iterator

cbegin()
→ const iterator

cend()
→ const past-the-end iterator

rbegin()
→ reverse iterator

rend()
→ reverse past-the-end iterator

crbegin()
→ const reverse iterator

crend()
→ const reverse past-the-end iterator
```

---

# 42. `back_inserter()`

`back_inserter()` creates an output iterator that inserts elements at the end of a container.

Example:

```cpp
vector<int> result;

auto it = back_inserter(result);

*it = 10;
*it = 20;
*it = 30;
```

Result:

```text
10 20 30
```

It essentially performs:

```cpp
result.push_back(value);
```

---

# 43. Why `back_inserter()` Is Useful

Consider:

```cpp
vector<int> a = {1, 2, 3};
vector<int> result;
```

You can write:

```cpp
copy(
    a.begin(),
    a.end(),
    back_inserter(result)
);
```

Now:

```text
result = {1, 2, 3}
```

This is commonly used with STL algorithms.

---

# 44. `front_inserter()`

For containers supporting `push_front()`:

```cpp
front_inserter(container)
```

Example:

```cpp
list<int> l;

auto it = front_inserter(l);

*it = 10;
*it = 20;
*it = 30;
```

Because insertion happens at the front, the resulting order can be:

```text
30 20 10
```

---

# 45. `inserter()`

`inserter()` inserts at a specified position.

Syntax:

```cpp
inserter(container, position)
```

Example:

```cpp
vector<int> v = {1, 4};

auto it = inserter(v, v.begin() + 1);

*it = 2;
*it = 3;
```

Result:

```text
1 2 3 4
```

---

# 46. Output Iterator Helpers

The important ones are:

```text
back_inserter()
front_inserter()
inserter()
```

Mental map:

```text
back_inserter
→ push_back()

front_inserter
→ push_front()

inserter
→ insert()
```

---

# 47. Iterator with `find()`

One of the most important STL patterns:

```cpp
auto it = find(
    v.begin(),
    v.end(),
    30
);
```

Then:

```cpp
if(it != v.end()) {
    cout << *it;
}
```

Your STL source uses this exact pattern: algorithms return an iterator, and you compare it with `end()` to determine whether the element was found.

---

# 48. Iterator with `lower_bound()`

```cpp
auto it = lower_bound(
    v.begin(),
    v.end(),
    x
);
```

Then:

```cpp
if(it != v.end()) {
    cout << *it;
}
```

`lower_bound()` returns an iterator to the first suitable position.

Your source describes it as the first position where the value can be inserted without breaking sorted order.

---

# 49. Iterator with `sort()`

```cpp
sort(
    v.begin(),
    v.end()
);
```

Here:

```text
v.begin()
→ starting iterator

v.end()
→ ending iterator

sort()
→ algorithm
```

This is the fundamental STL pattern:

```text
Container
    ↓
begin/end
    ↓
Iterators
    ↓
Algorithm
```

Your uploaded material emphasizes this exact separation.

---

# 50. Iterator with `for_each()`

```cpp
for_each(
    v.begin(),
    v.end(),
    [](int x) {
        cout << x << " ";
    }
);
```

Here:

```text
v
↓
container

begin/end
↓
iterators

for_each
↓
algorithm

lambda
↓
callable
```

---

# 51. Iterator with `count_if()`

```cpp
int result = count_if(
    v.begin(),
    v.end(),
    [](int x) {
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

Your STL material explicitly presents this combination as a core STL pattern.

---

# 52. Iterator Returned by Algorithms

Many STL algorithms return iterators.

Examples:

```text
find()
find_if()
find_if_not()

lower_bound()
upper_bound()
equal_range()

min_element()
max_element()

partition()
stable_partition()

remove()
remove_if()

unique()

rotate()
```

So a very important skill is:

```cpp
auto it = algorithm(...);
```

Then:

```cpp
*it
```

or:

```cpp
it != container.end()
```

depending on the algorithm.

---

# 53. Iterator Range `[begin, end)`

STL algorithms usually work with a half-open range:

```text
[begin, end)
```

This means:

```text
begin → included
end   → excluded
```

Example:

```cpp
sort(v.begin(), v.end());
```

means:

```text
[v.begin(), v.end())
```

All elements are included except the `end()` position.

---

# 54. Why `[begin, end)` Is Useful

Suppose:

```text
10 20 30 40 50
```

We want:

```text
20 30 40
```

We can use:

```cpp
sort(v.begin() + 1,
     v.begin() + 4);
```

The range is:

```text
[1, 4)
```

Therefore:

```text
20 ✓
30 ✓
40 ✓
50 ✗
```

---

# 55. Iterator Arithmetic

Only random-access iterators support direct arithmetic.

For a vector:

```cpp
auto it = v.begin();

it + 3
it - 2
it += 3
it -= 2
```

Example:

```cpp
cout << *(v.begin() + 2);
```

---

# 56. Iterator Comparison

Random-access iterators can be compared using ordering operators:

```cpp
it1 < it2
it1 > it2
it1 <= it2
it1 >= it2
```

Example:

```cpp
auto a = v.begin();
auto b = v.begin() + 3;

if(a < b) {
    cout << "a comes before b";
}
```

For general iterators, prefer:

```cpp
it1 == it2
it1 != it2
```

---

# 57. Iterator Invalidation

This is an advanced but extremely important topic.

An iterator can become invalid after modifying a container.

For example:

```cpp
vector<int> v = {10, 20, 30};

auto it = v.begin();

v.push_back(40);
```

Depending on whether reallocation occurs, the old iterator may become invalid.

Therefore:

```cpp
cout << *it;
```

may be unsafe after operations that invalidate iterators.

---

# 58. Vector Iterator Invalidation

For `vector`, operations that may cause reallocation can invalidate:

```text
iterators
references
pointers
```

A common example:

```cpp
push_back()
```

when the vector needs more capacity.

Safer mental rule:

```text
vector modification
        ↓
Could reallocate?
        ↓
Old iterators may be invalid
```

---

# 59. `erase()` and Iterators

Example:

```cpp
vector<int> v = {
    10, 20, 30, 40
};

auto it = v.begin() + 1;

v.erase(it);
```

Now the old iterator should not be reused.

But `erase()` returns a valid iterator to the element following the erased element:

```cpp
it = v.erase(it);
```

This is an extremely useful pattern.

---

# 60. Erasing While Iterating

Example:

```cpp
for(auto it = v.begin();
    it != v.end(); )
{
    if(*it % 2 == 0) {
        it = v.erase(it);
    }
    else {
        ++it;
    }
}
```

Why?

Because after:

```cpp
v.erase(it);
```

the iterator may be invalidated.

So we use the iterator returned by `erase()`.

---

# 61. `const_iterator` vs `iterator`

| Type             | Read | Modify |
| ---------------- | ---- | ------ |
| `iterator`       | ✓    | ✓      |
| `const_iterator` | ✓    | ❌      |

Example:

```cpp
auto it = v.begin();

*it = 100;     // ✓
```

But:

```cpp
auto it = v.cbegin();

*it = 100;     // ❌
```

---

# 62. `iterator` vs `reverse_iterator`

Normal iterator:

```text
→ → → →
```

Reverse iterator:

```text
← ← ← ←
```

Example:

```cpp
for(auto it = v.begin();
    it != v.end();
    ++it)
```

goes:

```text
10 → 20 → 30 → 40
```

while:

```cpp
for(auto it = v.rbegin();
    it != v.rend();
    ++it)
```

goes:

```text
40 → 30 → 20 → 10
```

---

# 63. `iterator_traits`

`iterator_traits` provides information about an iterator type.

Header:

```cpp
#include <iterator>
```

It can provide:

```text
value_type
difference_type
pointer
reference
iterator_category
```

Example:

```cpp
using It = vector<int>::iterator;

using Value = iterator_traits<It>::value_type;
```

Here:

```text
Value = int
```

---

# 64. Important `iterator_traits` Types

### `value_type`

The type of the element.

```cpp
vector<int>
```

has:

```text
value_type = int
```

### `difference_type`

Type used to represent distance between iterators.

### `reference`

Reference to the element.

### `pointer`

Pointer type.

### `iterator_category`

The iterator's category.

---

# 65. Why `iterator_traits` Matters

It is especially useful when writing **generic/template code**.

For example:

```cpp
template<typename Iterator>
void print(Iterator first, Iterator last)
{
    while(first != last)
    {
        cout << *first << " ";
        ++first;
    }
}
```

This can work with many containers because the function doesn't care about the specific container.

It only needs compatible iterators.

---

# 66. Pointer as an Iterator

A very important concept:

> A raw pointer can behave as a random-access/contiguous iterator.

Example:

```cpp
int arr[] = {10, 20, 30};

int* first = arr;
int* last = arr + 3;

sort(first, last);
```

This works because pointers provide the operations required by random-access/contiguous traversal.

So:

```text
Pointer
   ↓
Iterator-like interface
```

This is one reason iterators are called **generalized pointers**.

---

# 67. Iterator vs Pointer

| Pointer                     | Iterator                     |
| --------------------------- | ---------------------------- |
| Works with memory addresses | Works with containers/ranges |
| `*p`                        | `*it`                        |
| `p++`                       | `it++`                       |
| `p + n`                     | `it + n` when supported      |
| Low-level                   | Generic STL abstraction      |

Example:

```cpp
int* p = arr;

cout << *p;
```

and:

```cpp
auto it = v.begin();

cout << *it;
```

Both use dereferencing.

---

# 68. Common Iterator Mistakes

## Mistake 1 — Dereferencing `end()`

```cpp
cout << *v.end(); // ❌
```

Correct:

```cpp
auto it = v.begin();

while(it != v.end()) {
    cout << *it;
    ++it;
}
```

---

## Mistake 2 — Using `+` with list

```cpp
list<int> l;

auto it = l.begin();

it + 3; // ❌
```

Use:

```cpp
advance(it, 3);
```

---

## Mistake 3 — Using `sort()` on list

```cpp
sort(l.begin(), l.end()); // ❌
```

Use:

```cpp
l.sort();
```

---

## Mistake 4 — Using an invalidated iterator

```cpp
auto it = v.begin();

v.push_back(100);

// it may now be invalid
```

Do not blindly reuse old iterators after operations that can invalidate them.

---

## Mistake 5 — Forgetting `*`

Wrong:

```cpp
cout << it;
```

Usually you want:

```cpp
cout << *it;
```

The iterator identifies the position; dereferencing accesses the element.

---

# 69. Range-Based `for` vs Iterator Loop

Traditional iterator loop:

```cpp
for(auto it = v.begin();
    it != v.end();
    ++it)
{
    cout << *it;
}
```

Modern range-based loop:

```cpp
for(auto x : v)
{
    cout << x;
}
```

Your STL source recommends the range-based form when you simply need to visit every element.

Use explicit iterators when you need:

* Position
* Iterator returned by an algorithm
* Erasing elements
* Inserting elements
* Traversing partially
* Reverse traversal
* Iterator arithmetic

---

# 70. The Most Important Iterator Pattern

Memorize this:

```cpp
auto it = algorithm(
    container.begin(),
    container.end()
);
```

Then:

```cpp
if(it != container.end())
{
    cout << *it;
}
```

This pattern appears everywhere in STL.

---

# 71. Iterator Mental Model

Think of an iterator as:

```text
             Iterator
                │
                ↓
          Position in range
                │
                ↓
             *it
                │
                ↓
            Element
```

Movement:

```text
++it
 ↓
Next element

--it
 ↓
Previous element
```

Random access:

```text
it + n
 ↓
Jump forward

it - n
 ↓
Jump backward
```

---

# 72. Complete Iterator Map

```text
                         ITERATORS
                             │
             ┌───────────────┴───────────────┐
             │                               │
        Read / Traverse                 Write / Output
             │                               │
        Input Iterator                Output Iterator
             │
        Forward Iterator
             │
      Bidirectional Iterator
             │
       Random Access Iterator
             │
        Contiguous Iterator
```

---

# 73. Iterator Utilities

The important utilities to remember:

```cpp
advance()
next()
prev()
distance()
```

Output iterator helpers:

```cpp
back_inserter()
front_inserter()
inserter()
```

Type information:

```cpp
iterator_traits
```

Traversal functions:

```cpp
begin()
end()
cbegin()
cend()
rbegin()
rend()
crbegin()
crend()
```

---

# 74. Quick Reference Table

| Function / Concept | Purpose                   |
| ------------------ | ------------------------- |
| `begin()`          | First element             |
| `end()`            | One past last             |
| `cbegin()`         | Const beginning           |
| `cend()`           | Const ending              |
| `rbegin()`         | Reverse beginning         |
| `rend()`           | Reverse ending            |
| `crbegin()`        | Const reverse beginning   |
| `crend()`          | Const reverse ending      |
| `advance()`        | Move iterator             |
| `next()`           | Return advanced iterator  |
| `prev()`           | Return previous iterator  |
| `distance()`       | Calculate distance        |
| `back_inserter()`  | Insert at back            |
| `front_inserter()` | Insert at front           |
| `inserter()`       | Insert at position        |
| `iterator_traits`  | Iterator type information |

---

# 75. Iterator Categories — Final Comparison

```text
Input
 │
 ├── Can read
 ├── Move forward
 └── Single-pass style

Output
 │
 ├── Can write
 └── Move forward

Forward
 │
 ├── Read
 ├── Write when permitted
 ├── Move forward
 └── Multi-pass

Bidirectional
 │
 ├── Everything from Forward
 └── ++ and --

Random Access
 │
 ├── Everything from Bidirectional
 ├── + n
 ├── - n
 ├── [n]
 └── iterator difference

Contiguous
 │
 └── Random access + guaranteed contiguous storage
```

---

# 76. What You Should Memorize

For STL/competitive programming, prioritize these:

### Level 1 — Must Know

```cpp
begin()
end()
*it
++it
--it
```

### Level 2 — Very Important

```cpp
rbegin()
rend()

cbegin()
cend()

advance()
next()
prev()
distance()
```

### Level 3 — Algorithms

```cpp
find()
find_if()
sort()
count()
count_if()
lower_bound()
upper_bound()
min_element()
max_element()
```

Understand that these algorithms work through iterator ranges.

### Level 4 — Advanced

```cpp
iterator_traits
back_inserter()
front_inserter()
inserter()
iterator invalidation
```

---

# 77. Final STL Mental Model

The most important concept is not memorizing iterator functions individually.

Understand this:

```text
                 STL
                  │
       ┌──────────┼──────────┐
       │          │          │
   Container   Iterator   Algorithm
       │          │          │
       │          │          │
     stores    provides    processes
     data      positions    range
       │          │          │
       └──────────┼──────────┘
                  │
               Callable
                  │
             Lambda/Functor
```

For example:

```cpp
vector<int> numbers = {
    1, 2, 3, 4, 5, 6
};

count_if(
    numbers.begin(),
    numbers.end(),
    [](int x) {
        return x % 2 == 0;
    }
);
```

Here:

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

lambda
   ↓
Callable
```

This is the **core STL design pattern** emphasized in your source material.

---

# 78. Final Memory Tricks

### `begin / end`

```text
begin → first
end   → one past last
```

### `rbegin / rend`

```text
rbegin → last
rend   → one before first
```

### Iterator movement

```text
++it → forward
--it → backward
```

### Iterator utilities

```text
advance  → move
next     → next copy
prev     → previous copy
distance → count gap
```

### Iterator categories

```text
Input
 ↓
Forward
 ↓
Bidirectional
 ↓
Random Access
 ↓
Contiguous
```

with:

```text
Output
→ write-oriented iterator category
```

### STL philosophy

```text
Container
    ↓
Iterator
    ↓
Algorithm
    ↓
Callable
    ↓
Generic + reusable code
```

This is the key idea to carry into every STL algorithm you learn.
