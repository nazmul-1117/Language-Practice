# STL Set Algorithms

C++ STL provides several algorithms for working with **sets/ranges**.

Header:

```cpp
#include <algorithm>
```

The most important set algorithms are:

```text
set_union()
set_intersection()
set_difference()
set_symmetric_difference()
includes()
merge()
inplace_merge()
```

> **Important:** STL set algorithms generally expect the input ranges to be **sorted**.

---

# 1. What Are Set Algorithms?

Suppose we have:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
```

We can perform:

### Union

Everything from both:

```text
{1, 2, 3, 4, 5, 6}
```

### Intersection

Common elements:

```text
{3, 4}
```

### Difference

Elements in A but not B:

```text
{1, 2}
```

### Symmetric Difference

Elements present in exactly one set:

```text
{1, 2, 5, 6}
```

---

# 2. Important Set Algorithms

| Algorithm                    | Meaning                                   |
| ---------------------------- | ----------------------------------------- |
| `set_union()`                | A ∪ B                                     |
| `set_intersection()`         | A ∩ B                                     |
| `set_difference()`           | A − B                                     |
| `set_symmetric_difference()` | A △ B                                     |
| `includes()`                 | Checks whether one range contains another |
| `merge()`                    | Merges two sorted ranges                  |
| `inplace_merge()`            | Merges two consecutive sorted ranges      |

Mental map:

```text
              SET ALGORITHMS
                    │
       ┌────────────┼────────────┐
       │            │            │
     UNION    INTERSECTION   DIFFERENCE
       │            │            │
       └────────────┼────────────┘
                    │
          SYMMETRIC DIFFERENCE
                    │
                 INCLUDES
```

---

# 3. `set_union()`

`set_union()` creates the union of two sorted ranges.

Mathematically:

```text
A ∪ B
```

Example:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

Union = {1, 2, 3, 4, 5, 6}
```

## Syntax

```cpp
set_union(
    first1, last1,
    first2, last2,
    result
);
```

---

## Example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {

    vector<int> a = {1, 2, 3, 4};
    vector<int> b = {3, 4, 5, 6};

    vector<int> result;

    set_union(
        a.begin(), a.end(),
        b.begin(), b.end(),
        back_inserter(result)
    );

    for(int x : result)
        cout << x << " ";

}
```

Output:

```text
1 2 3 4 5 6
```

---

# 4. Why `back_inserter()`?

Notice:

```cpp
vector<int> result;
```

Initially it has size `0`.

So we cannot safely write directly into:

```cpp
result.begin()
```

Instead:

```cpp
back_inserter(result)
```

automatically performs:

```cpp
result.push_back(...)
```

for every generated element.

You will see this pattern frequently with STL algorithms.

---

# 5. `set_intersection()`

Returns elements common to both sorted ranges.

Mathematically:

```text
A ∩ B
```

Example:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

Intersection = {3, 4}
```

## Example

```cpp
vector<int> a = {1, 2, 3, 4};
vector<int> b = {3, 4, 5, 6};

vector<int> result;

set_intersection(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);
```

Result:

```text
3 4
```

---

# 6. `set_difference()`

Returns elements that exist in the **first range but not in the second**.

Mathematically:

```text
A - B
```

Example:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

A - B = {1, 2}
```

## Example

```cpp
vector<int> a = {1, 2, 3, 4};
vector<int> b = {3, 4, 5, 6};

vector<int> result;

set_difference(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);
```

Output:

```text
1 2
```

---

# 7. Direction of `set_difference()`

This is very important.

```cpp
set_difference(A, B)
```

means:

```text
A - B
```

But:

```cpp
set_difference(B, A)
```

means:

```text
B - A
```

Example:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
```

### A − B

```text
{1, 2}
```

### B − A

```text
{5, 6}
```

So the order matters.

---

# 8. `set_symmetric_difference()`

Returns elements that are present in **one range but not both**.

Mathematically:

```text
A △ B
```

Example:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
```

Result:

```text
{1, 2, 5, 6}
```

The common elements:

```text
3, 4
```

are removed.

---

## Example

```cpp
vector<int> a = {1, 2, 3, 4};
vector<int> b = {3, 4, 5, 6};

vector<int> result;

set_symmetric_difference(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);
```

Output:

```text
1 2 5 6
```

---

# 9. Four Main Set Operations

Suppose:

```text
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}
```

| Algorithm                    | Result          |
| ---------------------------- | --------------- |
| `set_union()`                | `{1,2,3,4,5,6}` |
| `set_intersection()`         | `{3,4}`         |
| `set_difference(A,B)`        | `{1,2}`         |
| `set_difference(B,A)`        | `{5,6}`         |
| `set_symmetric_difference()` | `{1,2,5,6}`     |

Easy memory trick:

```text
UNION
→ Everything

INTERSECTION
→ Common

DIFFERENCE
→ Only first

SYMMETRIC DIFFERENCE
→ Only one side
```

---

# 10. `includes()`

`includes()` checks whether all elements of one sorted range exist in another sorted range.

Think:

```text
Is B inside A?
```

It returns:

```cpp
true
```

or:

```cpp
false
```

---

## Example

```cpp
vector<int> a = {1, 2, 3, 4, 5};
vector<int> b = {2, 3, 4};

bool result = includes(
    a.begin(), a.end(),
    b.begin(), b.end()
);

cout << result;
```

Output:

```text
1
```

Because:

```text
2 ✓
3 ✓
4 ✓
```

all exist in `A`.

---

# 11. `includes()` False Example

```cpp
vector<int> a = {1, 2, 3, 4, 5};
vector<int> b = {2, 4, 6};

bool result = includes(
    a.begin(), a.end(),
    b.begin(), b.end()
);
```

Result:

```text
false
```

because:

```text
6
```

does not exist in `A`.

---

# 12. `merge()`

`merge()` merges two **sorted ranges** into another sorted range.

Example:

```text
A = {1, 3, 5}
B = {2, 4, 6}

Merged:

{1, 2, 3, 4, 5, 6}
```

## Example

```cpp
vector<int> a = {1, 3, 5};
vector<int> b = {2, 4, 6};

vector<int> result;

merge(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);
```

Output:

```text
1 2 3 4 5 6
```

---

# 13. `merge()` vs `set_union()`

This difference is important.

### `merge()`

Keeps elements from both ranges.

```text
A = {1, 2, 3}
B = {2, 3, 4}

merge
→ {1, 2, 2, 3, 3, 4}
```

### `set_union()`

Handles them like a set operation.

```text
set_union
→ {1, 2, 3, 4}
```

So:

```text
merge()
→ combines

set_union()
→ set operation
```

---

# 14. Duplicate Elements

STL set algorithms are slightly different from mathematical sets when duplicate values exist.

Example:

```text
A = {1, 2, 2, 3}
B = {2, 2, 4}
```

The algorithms operate on **sorted ranges and multiplicities**, rather than simply converting the containers into mathematical sets.

So remember:

> These algorithms work on ranges; they do not automatically remove every duplicate from your input containers.

If you need unique elements first:

```cpp
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());
```

---

# 15. `inplace_merge()`

Suppose a single range contains two consecutive sorted sections:

```text
{1, 3, 5 | 2, 4, 6}
```

The first section is sorted:

```text
1 3 5
```

The second section is sorted:

```text
2 4 6
```

`inplace_merge()` combines them into:

```text
1 2 3 4 5 6
```

---

## Example

```cpp
vector<int> v = {
    1, 3, 5,
    2, 4, 6
};

inplace_merge(
    v.begin(),
    v.begin() + 3,
    v.end()
);
```

Result:

```text
1 2 3 4 5 6
```

The middle iterator tells the algorithm where the second sorted range starts.

---

# 16. `merge()` vs `inplace_merge()`

### `merge()`

Works with:

```text
two separate ranges
```

```cpp
merge(
    a.begin(), a.end(),
    b.begin(), b.end(),
    result
);
```

### `inplace_merge()`

Works with:

```text
two consecutive sorted ranges
```

inside the same container.

```cpp
inplace_merge(
    v.begin(),
    middle,
    v.end()
);
```

---

# 17. Sorted Range Requirement

This is one of the most important rules.

For algorithms such as:

```cpp
set_union()
set_intersection()
set_difference()
set_symmetric_difference()
includes()
merge()
inplace_merge()
```

the input ranges should be sorted according to the same ordering.

Example:

```cpp
vector<int> a = {1, 3, 5};
vector<int> b = {2, 4, 6};
```

Good.

But:

```cpp
vector<int> a = {5, 1, 3};
```

is not properly sorted.

So first:

```cpp
sort(a.begin(), a.end());
```

---

# 18. Using Custom Ordering

You can also use a custom comparator.

For example, descending order:

```cpp
sort(a.begin(), a.end(), greater<int>());
sort(b.begin(), b.end(), greater<int>());
```

Then the set algorithm must use the **same comparator**.

Example:

```cpp
set_union(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result),
    greater<int>()
);
```

The important rule is:

```text
Input ordering
      ↓
Comparator
      ↓
Set algorithm
```

They should be consistent.

---

# 19. Complexity

For two sorted ranges of sizes `N` and `M`:

| Algorithm                    | Complexity |
| ---------------------------- | ---------: |
| `set_union()`                |   O(N + M) |
| `set_intersection()`         |   O(N + M) |
| `set_difference()`           |   O(N + M) |
| `set_symmetric_difference()` |   O(N + M) |
| `includes()`                 |   O(N + M) |
| `merge()`                    |   O(N + M) |

These are efficient because the ranges are already sorted.

---

# 20. Practical Example

Suppose two groups of students selected different subjects.

```text
Math:
1 2 3 4

Physics:
3 4 5 6
```

### Students in either subject

```cpp
set_union(...)
```

Result:

```text
1 2 3 4 5 6
```

### Students taking both

```cpp
set_intersection(...)
```

Result:

```text
3 4
```

### Students taking only Math

```cpp
set_difference(math, physics, ...)
```

Result:

```text
1 2
```

### Students taking exactly one subject

```cpp
set_symmetric_difference(...)
```

Result:

```text
1 2 5 6
```

This is exactly how to think about these algorithms.

---

# 21. Complete Example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {

    vector<int> a = {1, 2, 3, 4};
    vector<int> b = {3, 4, 5, 6};

    vector<int> result;

    // Union
    set_union(
        a.begin(), a.end(),
        b.begin(), b.end(),
        back_inserter(result)
    );

    cout << "Union: ";

    for(int x : result)
        cout << x << " ";

    cout << endl;

    result.clear();

    // Intersection
    set_intersection(
        a.begin(), a.end(),
        b.begin(), b.end(),
        back_inserter(result)
    );

    cout << "Intersection: ";

    for(int x : result)
        cout << x << " ";

    cout << endl;

    result.clear();

    // Difference
    set_difference(
        a.begin(), a.end(),
        b.begin(), b.end(),
        back_inserter(result)
    );

    cout << "Difference: ";

    for(int x : result)
        cout << x << " ";

    cout << endl;

    result.clear();

    // Symmetric difference
    set_symmetric_difference(
        a.begin(), a.end(),
        b.begin(), b.end(),
        back_inserter(result)
    );

    cout << "Symmetric Difference: ";

    for(int x : result)
        cout << x << " ";
}
```

Output:

```text
Union: 1 2 3 4 5 6
Intersection: 3 4
Difference: 1 2
Symmetric Difference: 1 2 5 6
```

---

# 22. Quick Reference

```cpp
// A ∪ B
set_union(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);

// A ∩ B
set_intersection(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);

// A - B
set_difference(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);

// A △ B
set_symmetric_difference(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);

// Check B ⊆ A
includes(
    a.begin(), a.end(),
    b.begin(), b.end()
);

// Merge two sorted ranges
merge(
    a.begin(), a.end(),
    b.begin(), b.end(),
    back_inserter(result)
);

// Merge two sorted sections of one range
inplace_merge(
    v.begin(),
    middle,
    v.end()
);
```

---

# 23. Final Memory Map

```text
                SET ALGORITHMS
                     │
       ┌─────────────┼─────────────┐
       │             │             │
     UNION      INTERSECTION    DIFFERENCE
       │             │             │
     A OR B        A AND B        A - B
       │
       └──────────────┐
                      │
             SYMMETRIC DIFFERENCE
                      │
                 A XOR B
                      │
                   INCLUDES
                      │
                "Is B inside A?"
```

### The 4 most important

```text
set_union()
      ↓
Everything

set_intersection()
      ↓
Common

set_difference()
      ↓
Only from first

set_symmetric_difference()
      ↓
Only from one side
```

### One golden rule

```text
SET ALGORITHMS
      ↓
SORTED INPUT
      ↓
ITERATOR RANGES
      ↓
OUTPUT ITERATOR
```

For competitive programming, make sure you are comfortable with **`set_union`, `set_intersection`, `set_difference`, `set_symmetric_difference`, and `includes`** first. These are the core of this topic.
