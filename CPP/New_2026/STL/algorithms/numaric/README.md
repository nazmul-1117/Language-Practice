# STL Numeric Algorithms

Numeric algorithms are STL functions mainly provided through the:

```cpp
#include <numeric>
```

They are used for:

* summation
* accumulation
* multiplication
* inner product
* prefix sums
* partial sums
* generating sequences
* adjacent differences
* numeric transformations
* GCD/LCM
* reducing ranges

The most important numeric algorithms are:

```text
accumulate()
reduce()

inner_product()
transform_reduce()

partial_sum()
inclusive_scan()
exclusive_scan()

adjacent_difference()

iota()

gcd()
lcm()

midpoint()
```

---

# 1. `accumulate()`

`accumulate()` is one of the most important numeric algorithms.

It processes all elements of a range and produces a single accumulated result.

### Header

```cpp
#include <numeric>
```

### Syntax

```cpp
accumulate(begin, end, initial_value);
```

---

## Basic Example

```cpp
vector<int> v = {1, 2, 3, 4, 5};

int sum = accumulate(
    v.begin(),
    v.end(),
    0
);
```

Result:

```text
15
```

Internally:

```text
initial = 0

0 + 1 = 1
1 + 2 = 3
3 + 3 = 6
6 + 4 = 10
10 + 5 = 15
```

So:

```text
accumulate()
      ↓
initial value
      ↓
process every element
      ↓
one final result
```

---

# 2. Why Initial Value Is Important

Consider:

```cpp
vector<int> v = {1, 2, 3, 4, 5};

int sum = accumulate(
    v.begin(),
    v.end(),
    0
);
```

The `0` is the initial value.

You can also use:

```cpp
int sum = accumulate(
    v.begin(),
    v.end(),
    10
);
```

Now:

```text
10 + 1 + 2 + 3 + 4 + 5
= 25
```

---

# 3. `accumulate()` for Multiplication

`accumulate()` is not limited to addition.

Syntax with an operation:

```cpp
accumulate(
    begin,
    end,
    initial,
    operation
);
```

Example:

```cpp
vector<int> v = {1, 2, 3, 4};

int product = accumulate(
    v.begin(),
    v.end(),
    1,
    multiplies<int>()
);
```

Result:

```text
24
```

Because:

```text
1 × 1 × 2 × 3 × 4 = 24
```

The initial value should be `1` for multiplication.

---

# 4. `accumulate()` with Lambda

You can provide your own operation.

Example:

```cpp
vector<int> v = {1, 2, 3, 4, 5};

int result = accumulate(
    v.begin(),
    v.end(),
    0,
    [](int sum, int x)
    {
        return sum + x * x;
    }
);
```

This calculates:

```text
1² + 2² + 3² + 4² + 5²
```

Result:

```text
55
```

---

# 5. `accumulate()` with Strings

`accumulate()` can also concatenate strings.

```cpp
vector<string> v = {
    "Hello",
    " ",
    "World"
};

string result = accumulate(
    v.begin(),
    v.end(),
    string("")
);
```

Result:

```text
Hello World
```

This demonstrates that `accumulate()` is a **general accumulation algorithm**, not only a mathematical sum.

---

# 6. `inner_product()`

`inner_product()` calculates the sum of pairwise products between two ranges.

### Syntax

```cpp
inner_product(
    first1,
    last1,
    first2,
    initial
);
```

Example:

```cpp
vector<int> a = {1, 2, 3};
vector<int> b = {4, 5, 6};

int result = inner_product(
    a.begin(),
    a.end(),
    b.begin(),
    0
);
```

Calculation:

```text
1 × 4
+
2 × 5
+
3 × 6
```

Therefore:

```text
4 + 10 + 18 = 32
```

Result:

```text
32
```

---

# 7. Understanding `inner_product()`

Think of:

```text
A = [1, 2, 3]

B = [4, 5, 6]

     ↓  ↓  ↓

    1×4
    2×5
    3×6

     ↓

   4 + 10 + 18

     ↓

     32
```

This is called the **inner product**.

It is commonly used in:

* vector mathematics
* dot product
* numerical calculations
* linear algebra
* weighted calculations

---

# 8. `inner_product()` with Custom Operations

The full form allows two operations:

```cpp
inner_product(
    first1,
    last1,
    first2,
    init,
    op1,
    op2
);
```

Example:

```cpp
int result = inner_product(
    a.begin(),
    a.end(),
    b.begin(),
    0,
    plus<int>(),
    multiplies<int>()
);
```

Here:

```text
op1 = plus
op2 = multiply
```

So:

```text
result =
0 + (1×4) + (2×5) + (3×6)
```

---

# 9. `partial_sum()`

`partial_sum()` calculates prefix/partial results.

### Header

```cpp
#include <numeric>
```

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

vector<int> result(5);

partial_sum(
    v.begin(),
    v.end(),
    result.begin()
);
```

Result:

```text
1 3 6 10 15
```

Because:

```text
1
1 + 2 = 3
1 + 2 + 3 = 6
1 + 2 + 3 + 4 = 10
1 + 2 + 3 + 4 + 5 = 15
```

---

# 10. Visualizing `partial_sum()`

Input:

```text
1  2  3  4  5
```

Output:

```text
1  3  6  10  15
```

Think:

```text
output[0] = 1

output[1] = 1 + 2

output[2] = 1 + 2 + 3

output[3] = 1 + 2 + 3 + 4

output[4] = 1 + 2 + 3 + 4 + 5
```

---

# 11. `partial_sum()` with Multiplication

You can provide a custom operation.

```cpp
vector<int> v = {
    1, 2, 3, 4
};

vector<int> result(4);

partial_sum(
    v.begin(),
    v.end(),
    result.begin(),
    multiplies<int>()
);
```

Result:

```text
1 2 6 24
```

Because:

```text
1
1 × 2 = 2
1 × 2 × 3 = 6
1 × 2 × 3 × 4 = 24
```

---

# 12. `iota()`

`iota()` fills a range with sequentially increasing values.

### Syntax

```cpp
iota(begin, end, starting_value);
```

Example:

```cpp
vector<int> v(5);

iota(
    v.begin(),
    v.end(),
    1
);
```

Result:

```text
1 2 3 4 5
```

---

# 13. `iota()` with Another Starting Value

```cpp
vector<int> v(5);

iota(
    v.begin(),
    v.end(),
    10
);
```

Result:

```text
10 11 12 13 14
```

So:

```text
iota(begin, end, start)
```

means:

```text
start
start + 1
start + 2
start + 3
...
```

---

# 14. `iota()` vs `fill()`

### `fill()`

```cpp
fill(
    v.begin(),
    v.end(),
    5
);
```

Result:

```text
5 5 5 5 5
```

### `iota()`

```cpp
iota(
    v.begin(),
    v.end(),
    5
);
```

Result:

```text
5 6 7 8 9
```

Remember:

```text
fill()
  ↓
same value

iota()
  ↓
increasing values
```

---

# 15. `adjacent_difference()`

`adjacent_difference()` calculates the difference between consecutive elements.

Example:

```cpp
vector<int> v = {
    1, 3, 6, 10
};

vector<int> result(4);

adjacent_difference(
    v.begin(),
    v.end(),
    result.begin()
);
```

Result:

```text
1 2 3 4
```

Why?

```text
result[0] = 1

3 - 1 = 2

6 - 3 = 3

10 - 6 = 4
```

So:

```text
Input:
1 3 6 10

Output:
1 2 3 4
```

---

# 16. Relationship Between `partial_sum()` and `adjacent_difference()`

These two algorithms are conceptually related.

### `partial_sum()`

```text
1 2 3 4
↓
1 3 6 10
```

### `adjacent_difference()`

```text
1 3 6 10
↓
1 2 3 4
```

They can be viewed as opposite-style operations.

---

# 17. `gcd()`

`gcd()` calculates the Greatest Common Divisor.

It is available through:

```cpp
#include <numeric>
```

Example:

```cpp
int result = gcd(12, 18);
```

Result:

```text
6
```

Because:

```text
Divisors of 12:
1, 2, 3, 4, 6, 12

Divisors of 18:
1, 2, 3, 6, 9, 18
```

Greatest common divisor:

```text
6
```

---

# 18. `lcm()`

`lcm()` calculates the Least Common Multiple.

```cpp
int result = lcm(12, 18);
```

Result:

```text
36
```

Because:

```text
Multiples of 12:
12, 24, 36, 48...

Multiples of 18:
18, 36, 54...

LCM = 36
```

---

# 19. `gcd()` and `lcm()` Together

```cpp
int a = 12;
int b = 18;

cout << gcd(a, b) << endl;
cout << lcm(a, b) << endl;
```

Output:

```text
6
36
```

These are especially useful in competitive programming.

---

# 20. `reduce()`

`reduce()` is another numeric algorithm used to combine elements into a single result.

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

int result = reduce(
    v.begin(),
    v.end(),
    0
);
```

Result:

```text
15
```

It looks similar to:

```cpp
accumulate()
```

but there is an important difference.

`reduce()` is designed to allow the implementation to perform the operation in a different order, which makes it useful for parallel execution.

For ordinary sequential accumulation where order matters, `accumulate()` is usually the simpler choice.

---

# 21. `transform_reduce()`

`transform_reduce()` combines:

```text
transform
+
reduce
```

It can transform elements and reduce them into one result.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4
};

int result = transform_reduce(
    v.begin(),
    v.end(),
    0,
    plus<int>(),
    [](int x)
    {
        return x * x;
    }
);
```

Calculation:

```text
1² + 2² + 3² + 4²
```

```text
1 + 4 + 9 + 16 = 30
```

Result:

```text
30
```

---

# 22. `transform_reduce()` with Two Ranges

This is useful for calculating a dot product.

```cpp
vector<int> a = {1, 2, 3};
vector<int> b = {4, 5, 6};

int result = transform_reduce(
    a.begin(),
    a.end(),
    b.begin(),
    0
);
```

Conceptually:

```text
1×4 + 2×5 + 3×6

= 4 + 10 + 18

= 32
```

This is closely related to:

```cpp
inner_product()
```

---

# 23. `inclusive_scan()`

`inclusive_scan()` calculates prefix results where the current element is included.

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

vector<int> result(5);

inclusive_scan(
    v.begin(),
    v.end(),
    result.begin()
);
```

Result:

```text
1 3 6 10 15
```

Conceptually:

```text
1
1 + 2
1 + 2 + 3
1 + 2 + 3 + 4
1 + 2 + 3 + 4 + 5
```

---

# 24. `exclusive_scan()`

`exclusive_scan()` calculates prefix results but excludes the current element.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

vector<int> result(5);

exclusive_scan(
    v.begin(),
    v.end(),
    result.begin(),
    0
);
```

Result:

```text
0 1 3 6 10
```

Compare:

```text
Input:

1 2 3 4 5

inclusive:

1 3 6 10 15

exclusive:

0 1 3 6 10
```

---

# 25. `inclusive_scan()` vs `exclusive_scan()`

This is very important.

### Inclusive

Current element is included:

```text
Input:
1 2 3 4

Output:
1 3 6 10
```

### Exclusive

Current element is excluded:

```text
Input:
1 2 3 4

Output:
0 1 3 6
```

Think:

```text
inclusive:
include current

exclusive:
exclude current
```

---

# 26. `exclusive_scan()` with Custom Operation

```cpp
exclusive_scan(
    v.begin(),
    v.end(),
    result.begin(),
    1,
    multiplies<int>()
);
```

For:

```text
1 2 3 4
```

the prefix multiplication is generated according to the exclusive-scan rule.

The initial value is important because it becomes the first output.

---

# 27. Numeric Algorithm Family

A useful way to remember these algorithms:

```text
                    <numeric>
                       |
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
   Accumulation     Prefix          Generation
       |               |                |
 accumulate()      partial_sum()      iota()
 reduce()          inclusive_scan()
                   exclusive_scan()
```

Another group:

```text
        Mathematical / Pair Operations
                    |
       ┌────────────┼─────────────┐
       ↓            ↓             ↓
inner_product  transform_reduce  gcd/lcm
```

And:

```text
          Difference Operations
                  |
                  ↓
       adjacent_difference()
```

---

# 28. Quick Reference Table

| Algorithm               | Purpose                            | Return          |
| ----------------------- | ---------------------------------- | --------------- |
| `accumulate()`          | Accumulate a range                 | Value           |
| `reduce()`              | Reduce range to one value          | Value           |
| `inner_product()`       | Pairwise multiply + accumulate     | Value           |
| `transform_reduce()`    | Transform + reduce                 | Value           |
| `partial_sum()`         | Generate partial/prefix sums       | Output iterator |
| `inclusive_scan()`      | Inclusive prefix operation         | Output iterator |
| `exclusive_scan()`      | Exclusive prefix operation         | Output iterator |
| `adjacent_difference()` | Difference between adjacent values | Output iterator |
| `iota()`                | Generate sequential values         | Output iterator |
| `gcd()`                 | Greatest common divisor            | Value           |
| `lcm()`                 | Least common multiple              | Value           |

---

# 29. Most Important for STL Learning

For the STL course, prioritize these first:

### Level 1 — Must Know

```text
accumulate()
iota()
partial_sum()
inner_product()
adjacent_difference()
```

### Level 2 — Important

```text
reduce()
transform_reduce()
inclusive_scan()
exclusive_scan()
```

### Level 3 — Mathematical Utilities

```text
gcd()
lcm()
```

---

# 30. The Core Mental Model

Remember this distinction:

### `<algorithm>`

Mostly works with:

```text
search
sort
modify
partition
compare
find
remove
```

Examples:

```cpp
sort()
find()
find_if()
count()
reverse()
rotate()
unique()
partition()
```

### `<numeric>`

Mostly works with:

```text
accumulation
prefix operations
numeric generation
mathematical operations
```

Examples:

```cpp
accumulate()
iota()
partial_sum()
inner_product()
adjacent_difference()
reduce()
transform_reduce()
```

---

# 31. Header Summary

For most algorithm functions:

```cpp
#include <algorithm>
```

For numeric algorithms:

```cpp
#include <numeric>
```

Example:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

int main()
{
    vector<int> v = {1, 2, 3, 4, 5};

    int sum = accumulate(
        v.begin(),
        v.end(),
        0
    );

    cout << sum;
}
```

Output:

```text
15
```

---

# Final Revision Sheet

```text
NUMERIC STL
│
├── Accumulation
│   ├── accumulate()
│   └── reduce()
│
├── Pair / Transform
│   ├── inner_product()
│   └── transform_reduce()
│
├── Prefix / Scan
│   ├── partial_sum()
│   ├── inclusive_scan()
│   └── exclusive_scan()
│
├── Difference
│   └── adjacent_difference()
│
├── Sequence Generation
│   └── iota()
│
└── Mathematical Utilities
    ├── gcd()
    └── lcm()
```

### The most important relationships

```text
accumulate()
    ↓
many values → ONE result

iota()
    ↓
start value → sequence

partial_sum()
    ↓
values → prefix sums

adjacent_difference()
    ↓
values → differences

inner_product()
    ↓
two ranges → one result

inclusive_scan()
    ↓
prefix + current element

exclusive_scan()
    ↓
prefix without current element

gcd() / lcm()
    ↓
mathematical operations
```
