# STL Min-Max & Element Algorithms

This section covers STL functions used to:

* find minimum and maximum values
* find the positions of minimum/maximum elements
* find both minimum and maximum together
* compare elements
* swap elements
* access elements through iterators
* manipulate individual elements
* work with custom objects and comparators

Main functions:

```text
min()
max()

min_element()
max_element()
minmax()
minmax_element()

clamp()

swap()
iter_swap()
iter_swap()

exchange()

equal()
lexicographical_compare()
```

---

# 1. `min()`

`min()` returns the smaller of two values.

### Header

```cpp
#include <algorithm>
```

### Example

```cpp
int a = 10;
int b = 20;

cout << min(a, b);
```

Output:

```text
10
```

Think:

```text
min(a, b)
    ↓
smaller value
```

---

# 2. `max()`

`max()` returns the larger of two values.

```cpp
int a = 10;
int b = 20;

cout << max(a, b);
```

Output:

```text
20
```

So:

```text
min()
 ↓
smaller

max()
 ↓
larger
```

---

# 3. `min()` with More Than Two Values

You can use an initializer list.

```cpp
int result = min({
    10,
    5,
    20,
    3,
    15
});
```

Result:

```text
3
```

Similarly:

```cpp
int result = max({
    10,
    5,
    20,
    3,
    15
});
```

Result:

```text
20
```

---

# 4. `min()` with Custom Comparator

You can provide your own comparison rule.

```cpp
int result = min(
    10,
    20,
    [](int a, int b)
    {
        return a > b;
    }
);
```

The comparator changes how the two values are compared.

For normal use, the default comparison is usually enough.

---

# 5. `min_element()`

`min_element()` finds the **smallest element in a range**.

Unlike `min()`, which compares values directly:

```cpp
min(a, b)
```

`min_element()` works with a container/range.

### Example

```cpp
vector<int> v = {
    10, 5, 30, 2, 20
};

auto it = min_element(
    v.begin(),
    v.end()
);
```

Access the minimum:

```cpp
cout << *it;
```

Output:

```text
2
```

---

# 6. Why Does `min_element()` Return an Iterator?

Because the algorithm needs to tell you **where the minimum element is**.

For:

```text
10 5 30 2 20
```

the result is:

```text
10 5 30 [2] 20
         ↑
         it
```

Therefore:

```cpp
*it
```

gives:

```text
2
```

But the iterator itself points to the element's position.

---

# 7. Checking `min_element()`

Always remember that an empty range has no minimum.

A safe pattern:

```cpp
auto it = min_element(
    v.begin(),
    v.end()
);

if (it != v.end())
{
    cout << *it;
}
```

---

# 8. `max_element()`

`max_element()` finds the largest element in a range.

```cpp
vector<int> v = {
    10, 5, 30, 2, 20
};

auto it = max_element(
    v.begin(),
    v.end()
);

cout << *it;
```

Output:

```text
30
```

Visual:

```text
10 5 [30] 2 20
      ↑
      it
```

---

# 9. `min_element()` vs `max_element()`

| Function        | Purpose                   | Return   |
| --------------- | ------------------------- | -------- |
| `min_element()` | Smallest element in range | Iterator |
| `max_element()` | Largest element in range  | Iterator |

Remember:

```text
min_element()
      ↓
minimum position

max_element()
      ↓
maximum position
```

---

# 10. Finding Index of Minimum Element

Because `min_element()` returns an iterator, we can calculate its index.

```cpp
vector<int> v = {
    10, 5, 30, 2, 20
};

auto it = min_element(
    v.begin(),
    v.end()
);

int index = it - v.begin();

cout << index;
```

Output:

```text
3
```

Because:

```text
10  5  30  2  20
0   1   2   3   4
            ↑
          minimum
```

---

# 11. Finding Index of Maximum

Similarly:

```cpp
auto it = max_element(
    v.begin(),
    v.end()
);

int index = it - v.begin();
```

For:

```text
10 5 30 2 20
```

the index is:

```text
2
```

---

# 12. `minmax()`

`minmax()` finds both the minimum and maximum between values.

Example:

```cpp
auto result = minmax(
    10,
    20
);
```

It returns:

```text
result.first
result.second
```

Therefore:

```cpp
cout << result.first << endl;
cout << result.second << endl;
```

Output:

```text
10
20
```

Conceptually:

```text
minmax(a, b)
     ↓
┌────┴────┐
↓         ↓
minimum  maximum
```

---

# 13. `minmax()` with Multiple Values

Using an initializer list:

```cpp
auto result = minmax({
    10,
    5,
    30,
    2,
    20
});
```

Then:

```cpp
cout << result.first;
cout << result.second;
```

Output:

```text
2
30
```

---

# 14. `minmax_element()`

`minmax_element()` finds both the minimum and maximum **elements in a range**.

```cpp
vector<int> v = {
    10, 5, 30, 2, 20
};

auto result = minmax_element(
    v.begin(),
    v.end()
);
```

It returns a pair of iterators:

```cpp
result.first
result.second
```

Therefore:

```cpp
cout << *result.first << endl;
cout << *result.second << endl;
```

Output:

```text
2
30
```

---

# 15. Understanding `minmax_element()`

For:

```text
10 5 30 2 20
```

we get:

```text
10 5 [30] [2] 20
        ↑    ↑
       max  min
```

But the returned pair is:

```text
result.first
      ↓
minimum iterator

result.second
      ↓
maximum iterator
```

So:

```cpp
*result.first
```

is the minimum.

And:

```cpp
*result.second
```

is the maximum.

---

# 16. `min_element()` vs `minmax_element()`

If you only need the minimum:

```cpp
auto it = min_element(
    v.begin(),
    v.end()
);
```

If you need both:

```cpp
auto result = minmax_element(
    v.begin(),
    v.end()
);
```

Then:

```cpp
*result.first
```

→ minimum

```cpp
*result.second
```

→ maximum

---

# 17. `minmax_element()` and Index

You can also find both indexes.

```cpp
auto result = minmax_element(
    v.begin(),
    v.end()
);

int minIndex =
    result.first - v.begin();

int maxIndex =
    result.second - v.begin();
```

---

# 18. Duplicate Minimum/Maximum Values

For:

```text
2 5 2 8 8
```

`min_element()` points to the first minimum:

```text
[2] 5 2 8 8
 ↑
```

Similarly, `max_element()` points to the first maximum according to the algorithm's comparison behavior.

This is important when duplicate values exist.

---

# 19. `clamp()`

`clamp()` keeps a value inside a specified range.

Conceptually:

```text
minimum ≤ value ≤ maximum
```

Example:

```cpp
int x = 15;

int result = clamp(
    x,
    0,
    10
);
```

Result:

```text
10
```

Because `15` is greater than the maximum `10`.

---

# 20. `clamp()` Examples

### Value inside range

```cpp
clamp(5, 0, 10);
```

Result:

```text
5
```

### Value below range

```cpp
clamp(-5, 0, 10);
```

Result:

```text
0
```

### Value above range

```cpp
clamp(20, 0, 10);
```

Result:

```text
10
```

Think:

```text
          0 -------- 10
          |          |
-5  ------|          |------ 20
 ↓                    ↓
 0                    10
```

---

# 21. `swap()`

`swap()` exchanges the values of two objects.

```cpp
int a = 10;
int b = 20;

swap(a, b);
```

Now:

```text
a = 20
b = 10
```

Visual:

```text
Before:

a → 10
b → 20

swap()

After:

a → 20
b → 10
```

---

# 22. `swap()` with Containers

You can swap entire containers.

```cpp
vector<int> a = {
    1, 2, 3
};

vector<int> b = {
    10, 20
};

swap(a, b);
```

Now:

```text
a → 10 20

b → 1 2 3
```

---

# 23. `iter_swap()`

`iter_swap()` swaps the elements pointed to by two iterators.

Example:

```cpp
vector<int> v = {
    10, 20, 30
};

auto it1 = v.begin();
auto it2 = v.begin() + 2;

iter_swap(
    it1,
    it2
);
```

Result:

```text
30 20 10
```

Because:

```text
it1 → 10
it2 → 30

iter_swap()

it1 → 30
it2 → 10
```

---

# 24. `swap()` vs `iter_swap()`

This distinction is important.

### `swap()`

Swaps objects/values:

```cpp
swap(a, b);
```

### `iter_swap()`

Swaps the elements pointed to by iterators:

```cpp
iter_swap(it1, it2);
```

Think:

```text
swap()
  ↓
objects

iter_swap()
  ↓
iterator → element
```

---

# 25. `equal()`

`equal()` checks whether two ranges contain equal corresponding elements.

Example:

```cpp
vector<int> a = {
    1, 2, 3
};

vector<int> b = {
    1, 2, 3
};

bool result = equal(
    a.begin(),
    a.end(),
    b.begin()
);
```

Result:

```text
true
```

Because:

```text
1 == 1
2 == 2
3 == 3
```

---

# 26. `equal()` Example

```cpp
vector<int> a = {
    1, 2, 3
};

vector<int> b = {
    1, 2, 4
};
```

Then:

```cpp
equal(
    a.begin(),
    a.end(),
    b.begin()
);
```

returns:

```text
false
```

because:

```text
3 != 4
```

---

# 27. `equal()` with a Predicate

You can provide a custom comparison.

```cpp
bool result = equal(
    a.begin(),
    a.end(),
    b.begin(),
    [](int x, int y)
    {
        return x == y;
    }
);
```

This becomes useful when comparing custom objects.

---

# 28. `lexicographical_compare()`

This compares two ranges lexicographically, similar to dictionary order.

Example:

```cpp
vector<int> a = {
    1, 2, 3
};

vector<int> b = {
    1, 2, 4
};

bool result = lexicographical_compare(
    a.begin(),
    a.end(),
    b.begin(),
    b.end()
);
```

Result:

```text
true
```

Because at the first different position:

```text
3 < 4
```

So:

```text
[1, 2, 3]
     <
[1, 2, 4]
```

---

# 29. String Example

Lexicographical comparison is similar to dictionary ordering.

```cpp
string a = "apple";
string b = "banana";
```

Conceptually:

```text
apple < banana
```

This concept is also used by STL containers such as ordered associative containers.

---

# 30. Element Access Through Iterators

When an algorithm returns an iterator:

```cpp
auto it = min_element(
    v.begin(),
    v.end()
);
```

The iterator itself represents a position.

To get the value:

```cpp
*it
```

To modify the element:

```cpp
*it = 100;
```

Example:

```cpp
vector<int> v = {
    10, 20, 5, 30
};

auto it = min_element(
    v.begin(),
    v.end()
);

*it = 100;
```

Now:

```text
10 20 100 30
```

---

# 31. `->` with Iterator to Objects

Suppose:

```cpp
struct Student
{
    string name;
    int marks;
};
```

And:

```cpp
vector<Student> students = {
    {"Rahim", 80},
    {"Karim", 90},
    {"Hasan", 70}
};
```

Find the student with minimum marks:

```cpp
auto it = min_element(
    students.begin(),
    students.end(),
    [](const Student& a, const Student& b)
    {
        return a.marks < b.marks;
    }
);
```

Then:

```cpp
cout << it->name;
```

Output:

```text
Hasan
```

Here:

```text
it
 ↓
Student object
 ↓
it->marks
it->name
```

---

# 32. `min_element()` with Custom Objects

This is a very important real-world use.

```cpp
auto it = min_element(
    students.begin(),
    students.end(),
    [](const Student& a, const Student& b)
    {
        return a.marks < b.marks;
    }
);
```

The algorithm itself doesn't know what "minimum" means for a `Student`.

You tell it:

```text
minimum = student with smaller marks
```

This is done using a comparator.

---

# 33. `max_element()` with Custom Objects

Similarly:

```cpp
auto it = max_element(
    students.begin(),
    students.end(),
    [](const Student& a, const Student& b)
    {
        return a.marks < b.marks;
    }
);
```

Now the iterator points to the student with maximum marks.

---

# 34. `minmax_element()` with Custom Objects

You can find both:

```cpp
auto result = minmax_element(
    students.begin(),
    students.end(),
    [](const Student& a, const Student& b)
    {
        return a.marks < b.marks;
    }
);
```

Then:

```cpp
cout << result.first->name;
cout << result.second->name;
```

---

# 35. Important Difference: Value vs Element

This is one of the most important concepts.

### `min()`

Works directly with values:

```cpp
min(10, 20);
```

Returns:

```text
10
```

### `min_element()`

Works on a range:

```cpp
min_element(
    v.begin(),
    v.end()
);
```

Returns:

```text
iterator
```

Therefore:

```text
min()
   ↓
value

min_element()
   ↓
iterator
```

Same idea for maximum:

```text
max()
   ↓
value

max_element()
   ↓
iterator
```

---

# 36. Quick Reference

| Function                    | Purpose                     | Works On  | Return            |
| --------------------------- | --------------------------- | --------- | ----------------- |
| `min()`                     | Find smaller value          | Values    | Value             |
| `max()`                     | Find larger value           | Values    | Value             |
| `minmax()`                  | Find min + max              | Values    | Pair              |
| `min_element()`             | Find minimum in range       | Range     | Iterator          |
| `max_element()`             | Find maximum in range       | Range     | Iterator          |
| `minmax_element()`          | Find min + max in range     | Range     | Pair of iterators |
| `clamp()`                   | Keep value within range     | Value     | Value             |
| `swap()`                    | Swap two objects            | Objects   | `void`            |
| `iter_swap()`               | Swap pointed elements       | Iterators | `void`            |
| `equal()`                   | Compare ranges              | Ranges    | `bool`            |
| `lexicographical_compare()` | Dictionary-style comparison | Ranges    | `bool`            |

---

# 37. Important Mental Map

```text
MIN / MAX
│
├── Values
│   ├── min()
│   ├── max()
│   └── minmax()
│
└── Ranges
    ├── min_element()
    ├── max_element()
    └── minmax_element()
```

And:

```text
ELEMENT MANIPULATION
│
├── swap()
├── iter_swap()
└── clamp()
```

And:

```text
COMPARISON
│
├── equal()
└── lexicographical_compare()
```

---

# 38. Most Important to Memorize

```text
min(a, b)
→ smaller VALUE

max(a, b)
→ larger VALUE

min_element(begin, end)
→ iterator to minimum ELEMENT

max_element(begin, end)
→ iterator to maximum ELEMENT

minmax(a, b)
→ minimum + maximum VALUES

minmax_element(begin, end)
→ minimum + maximum ITERATORS

clamp(x, low, high)
→ keeps x inside [low, high]

swap(a, b)
→ swaps objects

iter_swap(it1, it2)
→ swaps pointed elements

equal()
→ checks whether ranges match

lexicographical_compare()
→ dictionary-style range comparison
```

## STL Progression

So far, your STL algorithm learning path can now be organized as:

```text
STL ALGORITHMS
│
├── Iterators + General Algorithms
│   ├── for_each()
│   ├── sort()
│   ├── reverse()
│   ├── rotate()
│   ├── unique()
│   ├── partition()
│   └── remove()
│
├── Searching & Finding
│   ├── find()
│   ├── find_if()
│   ├── binary_search()
│   ├── lower_bound()
│   ├── upper_bound()
│   ├── equal_range()
│   ├── search()
│   └── search_n()
│
├── Numeric Algorithms
│   ├── accumulate()
│   ├── reduce()
│   ├── inner_product()
│   ├── partial_sum()
│   ├── iota()
│   ├── adjacent_difference()
│   ├── inclusive_scan()
│   └── exclusive_scan()
│
└── Min / Max & Element Operations
    ├── min()
    ├── max()
    ├── minmax()
    ├── min_element()
    ├── max_element()
    ├── minmax_element()
    ├── clamp()
    ├── swap()
    ├── iter_swap()
    ├── equal()
    └── lexicographical_compare()
```
