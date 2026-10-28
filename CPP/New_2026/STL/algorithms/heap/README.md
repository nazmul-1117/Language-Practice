# STL Heap Algorithms

Heap algorithms are part of the C++ STL `<algorithm>` library.

They are mainly used to **create, maintain, inspect, and sort heaps**.

```cpp
#include <algorithm>
#include <vector>
using namespace std;
```

---

# 1. What is a Heap?

A **heap** is a special binary-tree-based data structure stored efficiently inside an array/vector.

There are two common types:

### Max Heap

The largest element is always at the top.

```text
        50
       /  \
     30    40
    /  \
   10   20
```

```cpp
vector<int> v = {50, 30, 40, 10, 20};
```

### Min Heap

The smallest element is always at the top.

```text
        10
       /  \
     20    30
    /  \
   40   50
```

C++'s basic heap algorithms create a **max heap by default**.

---

# 2. Important Heap Algorithms

The main STL heap algorithms are:

| Algorithm         | Purpose                        |
| ----------------- | ------------------------------ |
| `make_heap()`     | Convert range into heap        |
| `push_heap()`     | Add a new element to heap      |
| `pop_heap()`      | Move largest element to end    |
| `sort_heap()`     | Sort a heap                    |
| `is_heap()`       | Check whether range is a heap  |
| `is_heap_until()` | Find where heap property stops |

Mental map:

```text
make_heap
    ↓
Create Heap

push_heap
    ↓
Insert into Heap

pop_heap
    ↓
Remove Top

sort_heap
    ↓
Sort Heap

is_heap
    ↓
Check Heap

is_heap_until
    ↓
Find Invalid Position
```

---

# 3. `make_heap()`

`make_heap()` converts a normal range into a heap.

## Syntax

```cpp
make_heap(begin, end);
```

By default, it creates a **max heap**.

## Example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {

    vector<int> v = {10, 30, 20, 5, 40};

    make_heap(v.begin(), v.end());

    for(int x : v)
        cout << x << " ";

    return 0;
}
```

The internal order is not necessarily sorted.

For example, you may get:

```text
40 30 20 5 10
```

The important property is:

```text
parent >= children
```

### Complexity

```text
O(N)
```

---

# 4. `push_heap()`

`push_heap()` is used to insert a new element into an existing heap.

### Important

You must first add the element at the end.

Then call:

```cpp
push_heap()
```

## Example

```cpp
vector<int> v = {10, 30, 20, 5, 40};

make_heap(v.begin(), v.end());

v.push_back(50);

push_heap(v.begin(), v.end());
```

Now:

```text
50
```

will become the top element.

### Complete example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {

    vector<int> v = {10, 30, 20, 5, 40};

    make_heap(v.begin(), v.end());

    v.push_back(50);

    push_heap(v.begin(), v.end());

    cout << v.front();

    return 0;
}
```

Output:

```text
50
```

### Complexity

```text
O(log N)
```

---

# 5. `pop_heap()`

`pop_heap()` removes the top element logically from the heap.

But there is an important detail:

> `pop_heap()` does NOT actually erase the element from the vector.

It moves the largest element to the end.

## Example

```cpp
vector<int> v = {10, 30, 20, 5, 40};

make_heap(v.begin(), v.end());

pop_heap(v.begin(), v.end());
```

After `pop_heap()`:

```text
40
```

is moved to the last position.

Conceptually:

```text
Heap:

        30
       /  \
     10    20
    /
   5

End:
40
```

Now actually remove it:

```cpp
v.pop_back();
```

### Complete pattern

```cpp
pop_heap(v.begin(), v.end());
v.pop_back();
```

### Complexity

```text
pop_heap() → O(log N)
pop_back() → O(1)
```

---

# 6. `sort_heap()`

`sort_heap()` sorts an existing heap.

## Example

```cpp
vector<int> v = {10, 30, 20, 5, 40};

make_heap(v.begin(), v.end());

sort_heap(v.begin(), v.end());
```

Result:

```text
5 10 20 30 40
```

For a max heap, `sort_heap()` produces ascending order.

### Complexity

```text
O(N log N)
```

---

# 7. `is_heap()`

`is_heap()` checks whether a range satisfies the heap property.

It returns:

```cpp
true
```

or

```cpp
false
```

## Example

```cpp
vector<int> v = {50, 30, 40, 10, 20};

cout << is_heap(v.begin(), v.end());
```

Output:

```text
1
```

Because:

```text
50 >= 30
50 >= 40
30 >= 10
30 >= 20
```

So it is a valid max heap.

---

## Invalid heap

```cpp
vector<int> v = {10, 50, 30, 20, 40};

cout << is_heap(v.begin(), v.end());
```

Output:

```text
0
```

because the parent-child relationship is violated.

---

# 8. `is_heap_until()`

`is_heap_until()` finds the first position where the heap property becomes invalid.

It returns an **iterator**.

## Example

```cpp
vector<int> v = {
    50, 30, 40, 10, 20, 60
};

auto it = is_heap_until(v.begin(), v.end());

cout << *it;
```

The returned iterator points to the first element that breaks the heap property.

---

# 9. Max Heap with Custom Comparator

By default:

```cpp
make_heap(v.begin(), v.end());
```

creates a max heap.

You can also explicitly use:

```cpp
greater<int>()
```

to create a min heap.

## Min Heap

```cpp
vector<int> v = {10, 30, 20, 5, 40};

make_heap(v.begin(), v.end(), greater<int>());
```

Now the smallest element is at:

```cpp
v.front()
```

So:

```cpp
cout << v.front();
```

gives:

```text
5
```

---

# 10. Max Heap vs Min Heap

### Max Heap

```cpp
make_heap(v.begin(), v.end());
```

Top:

```text
largest element
```

### Min Heap

```cpp
make_heap(
    v.begin(),
    v.end(),
    greater<int>()
);
```

Top:

```text
smallest element
```

---

# 11. Complete Heap Example

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {

    vector<int> v = {10, 30, 20, 5, 40};

    // Create heap
    make_heap(v.begin(), v.end());

    cout << "Top: " << v.front() << endl;

    // Insert
    v.push_back(50);
    push_heap(v.begin(), v.end());

    cout << "After insertion: "
         << v.front() << endl;

    // Remove top
    pop_heap(v.begin(), v.end());

    cout << "Removed: "
         << v.back() << endl;

    v.pop_back();

    // Sort heap
    sort_heap(v.begin(), v.end());

    cout << "Sorted: ";

    for(int x : v)
        cout << x << " ";

    return 0;
}
```

---

# 12. Heap vs `priority_queue`

This is extremely important.

C++ already provides:

```cpp
priority_queue
```

which internally uses heap operations.

Instead of manually doing:

```cpp
make_heap()
push_heap()
pop_heap()
```

you can usually use:

```cpp
priority_queue<int> pq;
```

Example:

```cpp
priority_queue<int> pq;

pq.push(10);
pq.push(50);
pq.push(20);

cout << pq.top();
```

Output:

```text
50
```

So:

```text
Heap Algorithms
      ↓
Low-level heap operations

priority_queue
      ↓
High-level heap-based container
```

---

# 13. Heap Algorithms vs Priority Queue

| Feature       | Heap Algorithms          | `priority_queue` |
| ------------- | ------------------------ | ---------------- |
| Header        | `<algorithm>`            | `<queue>`        |
| Works on      | Existing container/range | Own container    |
| `make_heap()` | Yes                      | Internal         |
| `push_heap()` | Yes                      | Internal         |
| `pop_heap()`  | Yes                      | Internal         |
| Access top    | `v.front()`              | `pq.top()`       |
| Remove        | `pop_heap + pop_back`    | `pq.pop()`       |
| Insert        | `push_back + push_heap`  | `pq.push()`      |

---

# 14. Heap Sort

Heap algorithms can be used to perform **Heap Sort**.

Basic process:

```text
1. Build heap
      ↓
2. Move maximum to end
      ↓
3. Restore heap
      ↓
4. Repeat
      ↓
5. Sorted array
```

STL makes this simple:

```cpp
make_heap(v.begin(), v.end());

sort_heap(v.begin(), v.end());
```

Complexity:

```text
O(N log N)
```

---

# 15. Important Difference: `sort()` vs `sort_heap()`

### `sort()`

Works directly on a normal range:

```cpp
sort(v.begin(), v.end());
```

### `sort_heap()`

Requires the range to already be a heap:

```cpp
make_heap(v.begin(), v.end());

sort_heap(v.begin(), v.end());
```

So:

```text
sort()
→ normal range

sort_heap()
→ heap range
```

---

# 16. Important Iterator Concept

Heap algorithms work on iterator ranges:

```cpp
[begin, end)
```

For example:

```cpp
make_heap(v.begin(), v.end());
```

means:

```text
v.begin()
   ↓
[ 10  30  20  5  40 ]
                      ↑
                   v.end()
```

`end()` itself is not part of the range.

---

# 17. Complexity Table

| Function          | Complexity |
| ----------------- | ---------: |
| `make_heap()`     |       O(N) |
| `push_heap()`     |   O(log N) |
| `pop_heap()`      |   O(log N) |
| `sort_heap()`     | O(N log N) |
| `is_heap()`       |       O(N) |
| `is_heap_until()` |       O(N) |

---

# 18. Quick Reference

```cpp
// Create max heap
make_heap(v.begin(), v.end());

// Insert
v.push_back(x);
push_heap(v.begin(), v.end());

// Remove top
pop_heap(v.begin(), v.end());
v.pop_back();

// Sort heap
sort_heap(v.begin(), v.end());

// Check heap
is_heap(v.begin(), v.end());

// Find first invalid position
is_heap_until(v.begin(), v.end());
```

---

# 19. Memory Trick

Remember:

```text
MAKE
 ↓
Create heap

PUSH
 ↓
Add element

POP
 ↓
Move top to end

SORT
 ↓
Sort heap

IS_HEAP
 ↓
Check heap

IS_HEAP_UNTIL
 ↓
Find where heap breaks
```

---

# 20. Most Important for Competitive Programming

Focus especially on:

```cpp
make_heap()
push_heap()
pop_heap()
sort_heap()
```

and understand:

```cpp
priority_queue
```

because in most competitive-programming problems, you will use:

```cpp
priority_queue<int> pq;
```

more frequently than manually manipulating heap algorithms.

### Core relationship

```text
                 HEAP
                  │
        ┌─────────┴─────────┐
        │                   │
 Heap Algorithms       priority_queue
        │
 ┌──────┼────────┐
 │      │        │
make   push     pop
heap   heap     heap
 │      │        │
 └──────┴────────┘
          │
       sort_heap
```
