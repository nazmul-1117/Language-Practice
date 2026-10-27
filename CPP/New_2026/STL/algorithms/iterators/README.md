# STL Algorithms & Iterators

C++ STL algorithms are reusable functions provided mainly through the `<algorithm>` header.

Most algorithms work on a **range of elements** represented by two iterators:

```cpp
algorithm(begin, end);
```

The range is:

```text
[begin, end)
```

This means:

* `begin` → included
* `end` → excluded
* `end` points one position after the last element

Example:

```cpp
vector<int> v = {10, 20, 30, 40, 50};

sort(v.begin(), v.end());
```

Here:

```text
begin()
   ↓
[10] [20] [30] [40] [50]
                           ↑
                          end()
```

---

# 1. `for_each()`

`for_each()` applies a function to every element in a range.

### Syntax

```cpp
for_each(begin, end, function);
```

### Example

```cpp
vector<int> v = {1, 2, 3, 4, 5};

for_each(
    v.begin(),
    v.end(),
    [](int x)
    {
        cout << x << " ";
    }
);
```

Output:

```text
1 2 3 4 5
```

### Using a normal function

```cpp
void print(int x)
{
    cout << x << " ";
}

for_each(v.begin(), v.end(), print);
```

### Important

`for_each()` processes every element.

Think:

```text
for_each
   ↓
element 1 → function
element 2 → function
element 3 → function
...
```

---

# 2. `find()`

`find()` searches for a specific value.

### Syntax

```cpp
find(begin, end, value);
```

### Example

```cpp
vector<int> v = {10, 20, 30, 40, 50};

auto it = find(
    v.begin(),
    v.end(),
    30
);
```

`find()` returns an **iterator**.

If the value is found:

```cpp
if (it != v.end())
{
    cout << "Found";
}
```

Access the value:

```cpp
cout << *it;
```

Output:

```text
30
```

If the value is not found:

```cpp
it == v.end()
```

### Important

```text
find()
  ↓
returns iterator
  ↓
found → iterator to element
not found → end()
```

---

# 3. `find_if()`

`find_if()` searches for the first element satisfying a condition.

### Example

Find the first even number:

```cpp
vector<int> v = {1, 3, 7, 8, 10};

auto it = find_if(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

Check:

```cpp
if (it != v.end())
{
    cout << *it;
}
```

Output:

```text
8
```

### Difference

```text
find()
    → searches for a value

find_if()
    → searches according to a condition
```

---

# 4. `count()`

`count()` counts how many times a particular value appears.

### Example

```cpp
vector<int> v = {
    10, 20, 10, 30, 10
};

int result = count(
    v.begin(),
    v.end(),
    10
);
```

Result:

```text
3
```

### Important

`count()` does **not** return an iterator.

It returns a number:

```text
count()
   ↓
number of matching elements
```

---

# 5. `count_if()`

`count_if()` counts elements satisfying a condition.

Example: count even numbers.

```cpp
vector<int> v = {
    1, 2, 3, 4, 5, 6
};

int result = count_if(
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
3
```

Because:

```text
2 4 6
```

### Difference

```text
count()
    → exact value

count_if()
    → condition
```

---

# 6. `sort()`

`sort()` sorts elements in a range.

### Example

```cpp
vector<int> v = {
    5, 2, 8, 1, 3
};

sort(
    v.begin(),
    v.end()
);
```

Result:

```text
1 2 3 5 8
```

### Descending order

```cpp
sort(
    v.begin(),
    v.end(),
    greater<int>()
);
```

Result:

```text
8 5 3 2 1
```

### Custom comparator

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

### Complexity

```text
O(n log n)
```

### Important

`sort()` uses iterators as the range but does not return an iterator.

```text
sort(begin, end)
      ↓
modifies the container
```

---

# 7. `reverse()`

`reverse()` reverses the order of elements.

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

reverse(
    v.begin(),
    v.end()
);
```

Result:

```text
5 4 3 2 1
```

It modifies the original range.

---

# 8. `rotate()`

`rotate()` rotates elements around a selected middle position.

### Example

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

rotate(
    v.begin(),
    v.begin() + 2,
    v.end()
);
```

Result:

```text
3 4 5 1 2
```

The second iterator specifies the element that becomes the new first element.

```text
Before:

1 2 | 3 4 5
      ↑
     middle

After:

3 4 5 | 1 2
```

### Another example

```cpp
rotate(
    v.begin(),
    v.begin() + 1,
    v.end()
);
```

Result:

```text
2 3 4 5 1
```

---

# 9. `unique()`

`unique()` removes **consecutive duplicate elements** by moving duplicates toward the end.

### Example

```cpp
vector<int> v = {
    1, 1, 2, 2, 3, 3
};

auto it = unique(
    v.begin(),
    v.end()
);
```

After `unique()`:

```text
1 2 3 ? ? ?
```

The returned iterator points to the new logical end.

Therefore, to actually remove the unwanted elements:

```cpp
v.erase(
    unique(v.begin(), v.end()),
    v.end()
);
```

Now:

```text
1 2 3
```

### Important

`unique()` does **not** actually resize the container.

This is why it is commonly used with `erase()`.

```text
unique()
   ↓
moves duplicates
   ↓
returns new logical end
   ↓
erase()
   ↓
actually removes them
```

---

# 10. `partition()`

`partition()` rearranges elements according to a condition.

Elements satisfying the condition are placed before elements that do not.

### Example

```cpp
vector<int> v = {
    1, 2, 3, 4, 5, 6
};

auto it = partition(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

The even elements are placed before the odd elements.

Possible result:

```text
6 2 4 | 3 5 1
       ↑
      it
```

The exact order inside each group is not guaranteed.

### What does it return?

`partition()` returns an iterator pointing to the first element of the second group.

```text
true group | false group
            ↑
        returned iterator
```

---

# 11. `stable_partition()`

`stable_partition()` works like `partition()`, but preserves the relative order of elements within each group.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4, 5, 6
};

stable_partition(
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
2 4 6 1 3 5
```

The original order of the even elements:

```text
2 → 4 → 6
```

is preserved.

The original order of the odd elements:

```text
1 → 3 → 5
```

is also preserved.

---

# 12. `remove()`

`remove()` removes elements logically from a range.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 2, 4
};

auto it = remove(
    v.begin(),
    v.end(),
    2
);
```

The matching elements are moved toward the end.

To actually erase them:

```cpp
v.erase(
    remove(v.begin(), v.end(), 2),
    v.end()
);
```

Result:

```text
1 3 4
```

This is called the:

**Erase-remove idiom**

---

# 13. `remove_if()`

`remove_if()` removes elements according to a condition.

Example: remove all even numbers.

```cpp
vector<int> v = {
    1, 2, 3, 4, 5, 6
};

v.erase(
    remove_if(
        v.begin(),
        v.end(),
        [](int x)
        {
            return x % 2 == 0;
        }
    ),
    v.end()
);
```

Result:

```text
1 3 5
```

---

# 14. `min_element()`

Returns an iterator pointing to the smallest element.

```cpp
vector<int> v = {
    5, 2, 8, 1, 3
};

auto it = min_element(
    v.begin(),
    v.end()
);
```

Access:

```cpp
cout << *it;
```

Output:

```text
1
```

Important:

```text
min_element()
      ↓
   iterator
      ↓
smallest element
```

---

# 15. `max_element()`

Returns an iterator pointing to the largest element.

```cpp
auto it = max_element(
    v.begin(),
    v.end()
);

cout << *it;
```

For:

```text
5 2 8 1 3
```

Output:

```text
8
```

---

# 16. `lower_bound()`

`lower_bound()` works on a **sorted range**.

It returns an iterator to the first element that is:

```text
>= value
```

Example:

```cpp
vector<int> v = {
    1, 3, 5, 5, 7, 9
};

auto it = lower_bound(
    v.begin(),
    v.end(),
    5
);
```

The iterator points to the first `5`.

```text
1 3 [5] 5 7 9
      ↑
     it
```

Think:

```text
lower_bound
     ↓
first position >= value
```

---

# 17. `upper_bound()`

`upper_bound()` returns an iterator to the first element:

```text
> value
```

Example:

```cpp
auto it = upper_bound(
    v.begin(),
    v.end(),
    5
);
```

For:

```text
1 3 5 5 7 9
```

It points to:

```text
1 3 5 5 [7] 9
          ↑
         it
```

Think:

```text
upper_bound
     ↓
first position > value
```

---

# 18. `binary_search()`

`binary_search()` checks whether an element exists in a sorted range.

```cpp
vector<int> v = {
    1, 3, 5, 7, 9
};

bool found = binary_search(
    v.begin(),
    v.end(),
    7
);
```

Result:

```text
true
```

Unlike `find()`, it returns a `bool`, not an iterator.

```text
find()
   → iterator

binary_search()
   → bool
```

The range must be sorted.

---

# 19. `equal_range()`

`equal_range()` returns the range containing all elements equivalent to a value.

It is effectively related to:

```cpp
lower_bound()
upper_bound()
```

Example:

```cpp
vector<int> v = {
    1, 3, 5, 5, 5, 7, 9
};

auto range = equal_range(
    v.begin(),
    v.end(),
    5
);
```

Conceptually:

```text
1 3 [5 5 5] 7 9
     ↑     ↑
 lower   upper
```

Access:

```cpp
auto first = range.first;
auto last = range.second;
```

---

# 20. `find_first_of()`

Searches one range for the first element matching any element from another range.

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

The first matching element is:

```text
4
```

So `it` points to `4`.

---

# 21. `adjacent_find()`

Finds the first pair of adjacent equal elements.

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
1 2 2 3 4
  ↑
 it
```

`it` points to the first `2`.

---

# 22. `all_of()`

Checks whether **all** elements satisfy a condition.

```cpp
vector<int> v = {
    2, 4, 6, 8
};

bool result = all_of(
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
true
```

---

# 23. `any_of()`

Checks whether **at least one** element satisfies a condition.

```cpp
bool result = any_of(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x > 5;
    }
);
```

Result:

```text
true
```

---

# 24. `none_of()`

Checks whether **no** element satisfies a condition.

```cpp
bool result = none_of(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x < 0;
    }
);
```

Result:

```text
true
```

---

# 25. `is_sorted()`

Checks whether a range is sorted.

```cpp
vector<int> v = {
    1, 2, 3, 4, 5
};

bool result = is_sorted(
    v.begin(),
    v.end()
);
```

Result:

```text
true
```

---

# 26. `is_sorted_until()`

Returns an iterator pointing to the first element that breaks sorted order.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 5, 4, 6
};

auto it = is_sorted_until(
    v.begin(),
    v.end()
);
```

Conceptually:

```text
1 2 3 5 [4] 6
          ↑
          it
```

The range is sorted until `4`.

---

# 27. `swap()`

Swaps two values.

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

It can also work with iterators:

```cpp
swap(*it1, *it2);
```

---

# 28. `fill()`

Fills every element in a range with a value.

```cpp
vector<int> v(5);

fill(
    v.begin(),
    v.end(),
    10
);
```

Result:

```text
10 10 10 10 10
```

---

# 29. `replace()`

Replaces every occurrence of a value.

```cpp
vector<int> v = {
    1, 2, 2, 3, 2
};

replace(
    v.begin(),
    v.end(),
    2,
    100
);
```

Result:

```text
1 100 100 3 100
```

---

# 30. `replace_if()`

Replaces elements satisfying a condition.

```cpp
replace_if(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x < 0;
    },
    0
);
```

All negative values become `0`.

---

# 31. `copy()`

Copies elements from one range to another.

```cpp
vector<int> source = {
    1, 2, 3, 4, 5
};

vector<int> destination(5);

copy(
    source.begin(),
    source.end(),
    destination.begin()
);
```

Now:

```text
destination:
1 2 3 4 5
```

---

# 32. `copy_if()`

Copies only elements satisfying a condition.

```cpp
vector<int> source = {
    1, 2, 3, 4, 5, 6
};

vector<int> destination;

copy_if(
    source.begin(),
    source.end(),
    back_inserter(destination),
    [](int x)
    {
        return x % 2 == 0;
    }
);
```

Result:

```text
2 4 6
```

---

# 33. `transform()`

`transform()` applies an operation to every element.

Example:

```cpp
vector<int> v = {
    1, 2, 3, 4
};

transform(
    v.begin(),
    v.end(),
    v.begin(),
    [](int x)
    {
        return x * 2;
    }
);
```

Result:

```text
2 4 6 8
```

---

# 34. `accumulate()`

`accumulate()` belongs to `<numeric>` rather than `<algorithm>`.

It processes a range and accumulates a result.

```cpp
#include <numeric>

vector<int> v = {
    10, 20, 30
};

int sum = accumulate(
    v.begin(),
    v.end(),
    0
);
```

Result:

```text
60
```

---

# 35. Important Iterator-Returning Algorithms

These algorithms commonly return iterators:

```text
find()
find_if()
find_if_not()

find_first_of()
adjacent_find()

min_element()
max_element()
minmax_element()

lower_bound()
upper_bound()
equal_range()

remove()
remove_if()

unique()

partition()
stable_partition()

rotate()

is_sorted_until()
```

The general pattern is:

```cpp
auto it = algorithm(
    container.begin(),
    container.end()
);
```

Then:

```cpp
if (it != container.end())
{
    cout << *it;
}
```

---

# 36. Algorithms That Return Counts

These return a numeric count:

```text
count()
count_if()
```

Example:

```cpp
int c = count(
    v.begin(),
    v.end(),
    10
);
```

---

# 37. Algorithms That Return `bool`

These return `true` or `false`:

```text
binary_search()
all_of()
any_of()
none_of()
is_sorted()
```

Example:

```cpp
if (all_of(
    v.begin(),
    v.end(),
    [](int x)
    {
        return x > 0;
    }
))
{
    cout << "All positive";
}
```

---

# 38. Algorithms That Modify the Range

Examples:

```text
sort()
reverse()
rotate()

fill()
replace()
replace_if()

remove()
remove_if()
unique()

partition()
stable_partition()

transform()
```

These generally change the elements in the supplied range.

---

# 39. The Most Important STL Pattern

A large part of STL can be understood using this pattern:

```text
              Container
                  ↓
        ┌─────────────────┐
        │ begin()  end()  │
        └─────────────────┘
                  ↓
             Iterator
                  ↓
             Algorithm
                  ↓
              Result
```

Example:

```cpp
auto it = find(
    v.begin(),
    v.end(),
    10
);
```

Breakdown:

```text
v
↓
begin(), end()
↓
iterator range
↓
find()
↓
iterator result
```

---

# 40. Quick Reference Table

| Algorithm          | Main Purpose                            | Return            |
| ------------------ | --------------------------------------- | ----------------- |
| `for_each`         | Apply operation to every element        | Function object   |
| `find`             | Find value                              | Iterator          |
| `find_if`          | Find by condition                       | Iterator          |
| `count`            | Count value                             | Integer           |
| `count_if`         | Count by condition                      | Integer           |
| `sort`             | Sort range                              | `void`            |
| `reverse`          | Reverse range                           | `void`            |
| `rotate`           | Rotate range                            | Iterator          |
| `unique`           | Remove consecutive duplicates logically | Iterator          |
| `remove`           | Remove value logically                  | Iterator          |
| `remove_if`        | Remove by condition logically           | Iterator          |
| `partition`        | Divide by condition                     | Iterator          |
| `stable_partition` | Stable partition                        | Iterator          |
| `min_element`      | Find minimum                            | Iterator          |
| `max_element`      | Find maximum                            | Iterator          |
| `lower_bound`      | First `>= value`                        | Iterator          |
| `upper_bound`      | First `> value`                         | Iterator          |
| `equal_range`      | Equal-value range                       | Pair of iterators |
| `binary_search`    | Search sorted range                     | `bool`            |
| `all_of`           | All satisfy condition                   | `bool`            |
| `any_of`           | Any satisfies condition                 | `bool`            |
| `none_of`          | None satisfy condition                  | `bool`            |
| `is_sorted`        | Check sorted                            | `bool`            |
| `is_sorted_until`  | Find where sorting stops                | Iterator          |
| `fill`             | Fill range                              | `void`            |
| `replace`          | Replace value                           | `void`            |
| `replace_if`       | Replace by condition                    | `void`            |
| `copy`             | Copy range                              | Output iterator   |
| `copy_if`          | Copy by condition                       | Output iterator   |
| `transform`        | Transform elements                      | Output iterator   |

---

# 41. What to Remember

The most important concepts from this section are:

### 1. Algorithms work on ranges

```cpp
algorithm(v.begin(), v.end());
```

### 2. `begin()` points to the first element

```cpp
v.begin()
```

### 3. `end()` points one position after the last element

```cpp
v.end()
```

### 4. Many algorithms return iterators

```cpp
auto it = find(...);
```

### 5. Check returned iterators

```cpp
if (it != v.end())
```

### 6. Dereference an iterator to access the element

```cpp
*it
```

### 7. `unique()` and `remove()` do not directly reduce a vector's size

Use:

```cpp
v.erase(
    unique(v.begin(), v.end()),
    v.end()
);
```

or:

```cpp
v.erase(
    remove(v.begin(), v.end(), value),
    v.end()
);
```

### 8. Algorithms + iterators are the core STL pattern

```text
Container
    ↓
Iterators
    ↓
Algorithm
    ↓
Result
```

Once this pattern becomes familiar, learning the rest of the STL becomes much easier.
