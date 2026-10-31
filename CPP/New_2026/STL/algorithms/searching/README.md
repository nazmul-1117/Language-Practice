# STL Searching & Finding Algorithms

Searching algorithms in C++ STL are used to locate elements, check whether elements exist, and find positions/ranges of elements.

Most of these functions work with iterators:

```cpp
algorithm(
    begin,
    end
);
```

The major searching functions are:

```text
find()
find_if()
find_if_not()

binary_search()

lower_bound()
upper_bound()
equal_range()

find_first_of()
adjacent_find()

search()
search_n()
```

---

# 1. `find()`

`find()` searches for a specific value in a range.

### Header

```cpp
#include <algorithm>
```

### Syntax

```cpp
find(begin, end, value);
```

### Example

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};

auto it = find(
    v.begin(),
    v.end(),
    30
);
```

`find()` returns an iterator.

Check whether the element was found:

```cpp
if (it != v.end())
{
    cout << "Found: " << *it;
}
else
{
    cout << "Not Found";
}
```

Output:

```text
Found: 30
```

### If not found

```cpp
it == v.end()
```

So the standard pattern is:

```cpp
auto it = find(v.begin(), v.end(), value);

if (it != v.end())
{
    // found
}
```

---

# 2. `find_if()`

`find_if()` finds the **first element satisfying a condition**.

### Example

Find the first even number:

```cpp
vector<int> v = {
    1, 3, 7, 8, 10
};

auto it = find_if(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

Result:

```text
8
```

Because `8` is the first element satisfying:

```cpp
x % 2 == 0
```

### Mental model

```text
find()
   ↓
find this VALUE

find_if()
   ↓
find first element satisfying CONDITION
```

---

# 3. `find_if_not()`

This is the opposite of `find_if()`.

It finds the first element that **does not satisfy** the condition.

Example:

```cpp
vector<int> v = {
    2, 4, 6, 7, 8
};

auto it = find_if_not(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

Result:

```text
7
```

Because `7` is the first element that is **not even**.

---

# 4. `binary_search()`

`binary_search()` checks whether a value exists in a **sorted range**.

### Syntax

```cpp
binary_search(
    begin,
    end,
    value
);
```

### Example

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};

bool found = binary_search(
    v.begin(),
    v.end(),
    30
);
```

Result:

```text
true
```

If we search for:

```cpp
60
```

Result:

```text
false
```

### Important

`binary_search()` returns:

```text
bool
```

It does **not** return an iterator.

```text
binary_search()
       ↓
   true / false
```

---

# 5. Why Binary Search?

Suppose we have:

```text
10 20 30 40 50 60 70 80
```

Instead of checking every element, binary search repeatedly divides the search range.

```text
10 20 30 40 | 50 60 70 80
             ↑
           middle
```

If searching for `70`, it eliminates half of the range at each step.

Complexity:

```text
O(log n)
```

But the range must be sorted.

---

# 6. `find()` vs `binary_search()`

| Feature                 | `find()`      | `binary_search()` |
| ----------------------- | ------------- | ----------------- |
| Works on unsorted range | Yes           | No                |
| Requires sorted range   | No            | Yes               |
| Search method           | Linear search | Binary search     |
| Complexity              | O(n)          | O(log n)          |
| Return                  | Iterator      | `bool`            |
| Gives position          | Yes           | No                |

Example:

```cpp
auto it = find(v.begin(), v.end(), 30);
```

gives the iterator.

Whereas:

```cpp
bool found = binary_search(
    v.begin(),
    v.end(),
    30
);
```

only tells you whether it exists.

---

# 7. `lower_bound()`

`lower_bound()` is one of the most important STL searching functions.

It works on a **sorted range**.

It returns an iterator pointing to the **first element greater than or equal to the given value**.

In mathematical form:

```text
first element >= value
```

### Example

```cpp
vector<int> v = {
    10, 20, 30, 30, 40, 50
};

auto it = lower_bound(
    v.begin(),
    v.end(),
    30
);
```

The result points to the first `30`.

```text
10 20 [30] 30 40 50
        ↑
        it
```

---

# 8. `lower_bound()` When Value Doesn't Exist

Consider:

```text
10 20 30 40 50
```

Search for:

```text
35
```

There is no `35`.

`lower_bound(35)` returns the first element:

```text
>= 35
```

which is:

```text
40
```

Visual:

```text
10 20 30 [40] 50
          ↑
       lower_bound
```

So:

```text
lower_bound(x)
      ↓
first element >= x
```

---

# 9. `upper_bound()`

`upper_bound()` returns an iterator pointing to the **first element greater than the given value**.

In mathematical form:

```text
first element > value
```

Example:

```cpp
vector<int> v = {
    10, 20, 30, 30, 40, 50
};

auto it = upper_bound(
    v.begin(),
    v.end(),
    30
);
```

Result:

```text
10 20 30 30 [40] 50
             ↑
             it
```

So:

```text
upper_bound(x)
      ↓
first element > x
```

---

# 10. `lower_bound()` vs `upper_bound()`

This is extremely important.

For:

```text
10 20 30 30 30 40 50
```

Searching for `30`:

### `lower_bound(30)`

Finds the first:

```text
>= 30
```

```text
10 20 [30] 30 30 40 50
        ↑
```

### `upper_bound(30)`

Finds the first:

```text
> 30
```

```text
10 20 30 30 30 [40] 50
                 ↑
```

Remember:

```text
lower_bound → >=

upper_bound → >
```

---

# 11. Finding Frequency Using Bounds

Suppose:

```cpp
vector<int> v = {
    10, 20, 30, 30, 30, 40, 50
};
```

We want to count how many times `30` occurs.

Use:

```cpp
auto first = lower_bound(
    v.begin(),
    v.end(),
    30
);

auto last = upper_bound(
    v.begin(),
    v.end(),
    30
);
```

Then:

```cpp
int count = last - first;
```

Result:

```text
3
```

Why?

```text
10 20 [30 30 30] 40 50
        ↑       ↑
      first    last
```

The distance between them is `3`.

---

# 12. `equal_range()`

`equal_range()` combines:

```text
lower_bound()
+
upper_bound()
```

It returns both iterators together.

### Example

```cpp
vector<int> v = {
    10, 20, 30, 30, 30, 40, 50
};

auto range = equal_range(
    v.begin(),
    v.end(),
    30
);
```

It returns:

```cpp
range.first
range.second
```

Conceptually:

```text
10 20 [30 30 30] 40 50
        ↑       ↑
      first    second
```

Equivalent to:

```cpp
auto first = lower_bound(
    v.begin(),
    v.end(),
    30
);

auto second = upper_bound(
    v.begin(),
    v.end(),
    30
);
```

---

# 13. `equal_range()` for Frequency

```cpp
auto range = equal_range(
    v.begin(),
    v.end(),
    30
);

int frequency =
    range.second - range.first;
```

Result:

```text
3
```

This is a very common competitive-programming technique.

---

# 14. `equal_range()` Concept

Think:

```text
              equal_range(30)
                     |
             ┌───────┴───────┐
             ↓               ↓
       lower_bound      upper_bound
             ↓               ↓
       first 30        after last 30
```

So:

```text
equal_range()
       ↓
[first occurrence, position after last occurrence)
```

---

# 15. `find_first_of()`

`find_first_of()` searches one range for the first element that matches **any element in another range**.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

vector<int> targets = {
    4, 8
};

auto it = find_first_of(
    v.begin(),
    v.end(),
    targets.begin(),
    targets.end()
);
```

`4` is found.

```text
1 2 3 [4] 5
        ↑
```

Because `4` exists in:

```text
targets = {4, 8}
```

---

# 16. `adjacent_find()`

`adjacent_find()` finds the first pair of adjacent elements that are equal.

Example:

```cpp
vector<int> v = {
    1, 2, 2, 3, 4
};

auto it = adjacent_find(
    v.begin(),
    v.end()
);
```

Result:

```text
1 [2] 2 3 4
  ↑
  it
```

The iterator points to the first element of the matching adjacent pair.

---

# 17. `adjacent_find()` with Condition

You can provide a custom comparison.

Example:

```cpp
auto it = adjacent_find(
    v.begin(),
    v.end(),
    [](int a, int b)
    {
        return abs(a - b) == 1;
    }
);
```

Now it searches for adjacent elements whose difference is `1`.

---

# 18. `search()`

`search()` searches for one sequence inside another sequence.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4, 5, 6
};

vector<int> target = {
    3, 4, 5
};

auto it = search(
    v.begin(),
    v.end(),
    target.begin(),
    target.end()
);
```

Result:

```text
1 2 [3 4 5] 6
    ↑
    it
```

So `search()` finds a **subsequence**.

---

# 19. `search_n()`

`search_n()` searches for `n` consecutive occurrences of a value.

Example:

```cpp
vector<int> v = {
    1, 2, 2, 2, 3
};

auto it = search_n(
    v.begin(),
    v.end(),
    3,
    2
);
```

Meaning:

```text
Find 3 consecutive 2s.
```

Result:

```text
1 [2 2 2] 3
  ↑
  it
```

---

# 20. Searching With Custom Comparators

Many searching algorithms allow custom conditions.

For example:

```cpp
auto it = find_if(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x > 100;
    }
);
```

This allows you to search based on:

* greater than
* less than
* even/odd
* string properties
* object fields
* custom logic

---

# 21. Searching `vector<pair<int,int>>`

STL searching becomes even more useful with custom objects.

Example:

```cpp
vector<pair<int, int>> v = {
    {1, 100},
    {2, 200},
    {3, 300}
};
```

You can use `find_if()`:

```cpp
auto it = find_if(
    v.begin(),
    v.end(),
    [](pair<int,int> p)
    {
        return p.first == 2;
    }
);
```

Then:

```cpp
cout << it->second;
```

Output:

```text
200
```

---

# 22. Main Categories

## Linear Searching

These generally examine elements sequentially:

```text
find()
find_if()
find_if_not()
find_first_of()
adjacent_find()
search()
search_n()
```

Typical complexity:

```text
O(n)
```

---

## Binary Searching

These are designed for sorted ranges:

```text
binary_search()
lower_bound()
upper_bound()
equal_range()
```

Typical complexity:

```text
O(log n)
```

---

# 23. Critical Requirement: Sorted Range

These functions require a sorted range:

```text
binary_search()
lower_bound()
upper_bound()
equal_range()
```

Example:

```cpp
vector<int> v = {
    10, 20, 30, 40, 50
};
```

Good.

But:

```cpp
vector<int> v = {
    40, 10, 50, 20, 30
};
```

is not sorted.

You should first do:

```cpp
sort(
    v.begin(),
    v.end()
);
```

Then use:

```cpp
binary_search()
lower_bound()
upper_bound()
equal_range()
```

---

# 24. The Most Important Search Pattern

For a sorted vector:

```cpp
vector<int> v = {
    10, 20, 30, 30, 30, 40, 50
};
```

### Does `30` exist?

```cpp
binary_search(
    v.begin(),
    v.end(),
    30
);
```

### First `30`?

```cpp
lower_bound(
    v.begin(),
    v.end(),
    30
);
```

### Position after last `30`?

```cpp
upper_bound(
    v.begin(),
    v.end(),
    30
);
```

### Complete range of `30`s?

```cpp
equal_range(
    v.begin(),
    v.end(),
    30
);
```

### Number of `30`s?

```cpp
auto range = equal_range(
    v.begin(),
    v.end(),
    30
);

int frequency =
    range.second - range.first;
```

---

# 25. Visual Summary

For:

```text
10 20 30 30 30 40 50
```

```text
             30 30 30
             ↓      ↓
        lower_bound  upper_bound

10 20 [30 30 30] 40 50
        ↑       ↑
      first    last
```

Therefore:

```text
lower_bound(30)
       ↓
first 30

upper_bound(30)
       ↓
position after last 30

equal_range(30)
       ↓
[first 30, after last 30)
```

---

# 26. Complete Quick Reference

| Algorithm         | Purpose                                      | Sorted Required? | Return            |
| ----------------- | -------------------------------------------- | ---------------: | ----------------- |
| `find()`          | Find exact value                             |               No | Iterator          |
| `find_if()`       | Find by condition                            |               No | Iterator          |
| `find_if_not()`   | Find first failing condition                 |               No | Iterator          |
| `binary_search()` | Check existence                              |              Yes | `bool`            |
| `lower_bound()`   | First `>= value`                             |              Yes | Iterator          |
| `upper_bound()`   | First `> value`                              |              Yes | Iterator          |
| `equal_range()`   | Range of equal values                        |              Yes | Pair of iterators |
| `find_first_of()` | Find first matching value from another range |               No | Iterator          |
| `adjacent_find()` | Find adjacent matching elements              |               No | Iterator          |
| `search()`        | Find a subsequence                           |               No | Iterator          |
| `search_n()`      | Find repeated values                         |               No | Iterator          |

---

# 27. Easy Memory Trick

Remember these four together:

```text
binary_search
     ↓
Does it exist?

lower_bound
     ↓
Where does >= x start?

upper_bound
     ↓
Where does > x start?

equal_range
     ↓
Where are all x values?
```

For example:

```text
10 20 30 30 30 40 50
```

```text
binary_search(30)
        ↓
      true

lower_bound(30)
        ↓
      first 30

upper_bound(30)
        ↓
      after last 30

equal_range(30)
        ↓
      [first 30, after last 30)
```

---

# 28. Overall Searching & Finding Map

```text
SEARCHING & FINDING
│
├── Linear Search
│   ├── find()
│   ├── find_if()
│   ├── find_if_not()
│   ├── find_first_of()
│   ├── adjacent_find()
│   ├── search()
│   └── search_n()
│
└── Binary Search
    │
    ├── binary_search()
    │
    ├── lower_bound()
    │      └── first >= x
    │
    ├── upper_bound()
    │      └── first > x
    │
    └── equal_range()
           ├── lower_bound()
           └── upper_bound()
```

## Final Things to Memorize

```text
find()
→ exact value

find_if()
→ condition

binary_search()
→ existence

lower_bound()
→ first >= x

upper_bound()
→ first > x

equal_range()
→ all x positions

adjacent_find()
→ adjacent matching elements

search()
→ find a sequence

search_n()
→ find repeated consecutive values
```

The **most important STL searching concept** is the relationship:

```text
lower_bound()
      +
upper_bound()
      ↓
equal_range()
```

and:

```text
lower_bound()
      ↓
frequency = upper_bound(x) - lower_bound(x)
```

This pattern is extremely useful in competitive programming and is one of the key STL techniques to master.
