# `std::queue` — Complete Guide

`std::queue` is a **container adaptor** in the C++ Standard Library that provides **FIFO (First-In, First-Out)** behavior.

```cpp
#include <queue>
```

The element that enters first is the element that leaves first.

```text
First In                         First Out
   ↓                                ↓
┌──────┬──────┬──────┬──────┐
│  10  │  20  │  30  │  40  │
└──────┴──────┴──────┴──────┘
   ↑                         ↑
 front                      back
```

Example:

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);
```

Queue:

```text
front → 10 → 20 → 30 ← back
```

Calling:

```cpp
q.pop();
```

removes `10`.

---

# 1. What Is a Queue?

A queue follows:

> **FIFO — First In, First Out**

Think about a real-world queue:

```text
Person A → Person B → Person C → Person D
   ↑
First person served
```

The first person who joins the queue is the first person served.

In C++:

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);
```

Then:

```cpp
q.front(); // 10
```

and:

```cpp
q.pop();
```

removes `10`.

---

# 2. Basic Syntax

```cpp
std::queue<T> name;
```

Examples:

```cpp
std::queue<int> numbers;
std::queue<std::string> names;
std::queue<double> prices;
```

---

# 3. Header

Include:

```cpp
#include <queue>
```

Usually:

```cpp
#include <iostream>
#include <queue>
```

---

# 4. Basic Example

```cpp
#include <iostream>
#include <queue>

int main() {

    std::queue<int> q;

    q.push(10);
    q.push(20);
    q.push(30);

    std::cout << q.front() << '\n';

    q.pop();

    std::cout << q.front() << '\n';

    return 0;
}
```

Output:

```text
10
20
```

---

# 5. Queue Structure

Suppose:

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);
q.push(40);
```

Conceptually:

```text
front                              back
  ↓                                  ↓
┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │
└────┴────┴────┴────┘
```

The next element removed is:

```text
10
```

Then:

```text
20
```

Then:

```text
30
```

Then:

```text
40
```

---

# 6. `push()`

Adds an element to the **back** of the queue.

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);
```

Queue:

```text
10 → 20 → 30
```

Complexity:

```text
O(1)
```

---

# 7. `emplace()`

Constructs an element directly at the back of the queue.

```cpp
std::queue<std::string> q;

q.emplace("Alice");
q.emplace("Bob");
```

For custom objects:

```cpp
std::queue<Player> players;

players.emplace("Alice", 100);
```

`emplace()` can avoid constructing a temporary object before insertion.

---

# 8. `pop()`

Removes the element at the **front**.

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);

q.pop();
```

Before:

```text
front
 ↓
10 → 20 → 30
```

After:

```text
front
 ↓
20 → 30
```

Complexity:

```text
O(1)
```

---

# 9. Important: `pop()` Does Not Return the Element

A common mistake is:

```cpp
int x = q.pop(); // ERROR
```

`pop()` returns `void`.

Correct:

```cpp
int x = q.front();

q.pop();
```

Example:

```cpp
while (!q.empty()) {

    int x = q.front();

    std::cout << x << '\n';

    q.pop();
}
```

---

# 10. `front()`

Returns a reference to the first element.

```cpp
std::queue<int> q;

q.push(10);
q.push(20);

std::cout << q.front();
```

Output:

```text
10
```

---

# 11. Modifying the Front Element

Because `front()` returns a reference, you can modify the element.

```cpp
q.front() = 100;
```

Example:

```cpp
std::queue<int> q;

q.push(10);
q.push(20);

q.front() = 100;
```

Queue:

```text
100 → 20
```

---

# 12. `back()`

Returns a reference to the last element.

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);

std::cout << q.back();
```

Output:

```text
30
```

---

# 13. Modifying the Back Element

```cpp
q.back() = 100;
```

Queue:

```text
10 → 20 → 100
```

---

# 14. `empty()`

Checks whether the queue contains no elements.

```cpp
if (q.empty()) {
    std::cout << "Queue is empty";
}
```

Returns:

```text
true
```

or:

```text
false
```

---

# 15. `size()`

Returns the number of elements.

```cpp
std::cout << q.size();
```

Example:

```cpp
std::queue<int> q;

q.push(10);
q.push(20);
q.push(30);

std::cout << q.size();
```

Output:

```text
3
```

---

# 16. `swap()`

Swaps two queues.

```cpp
std::queue<int> a;
std::queue<int> b;

a.push(1);
a.push(2);

b.push(3);
b.push(4);

a.swap(b);
```

Now:

```text
a:
3 → 4

b:
1 → 2
```

You can also use:

```cpp
std::swap(a, b);
```

---

# 17. Complete `std::queue` Member Functions

The primary member functions are:

```cpp
push()
emplace()
pop()

front()
back()

empty()
size()

swap()
```

Unlike containers such as `vector` and `list`, `queue` intentionally provides a very small interface.

This is because `queue` is designed to enforce FIFO behavior.

---

# 18. Function Summary

| Function    | Purpose            | Complexity |
| ----------- | ------------------ | ---------: |
| `push()`    | Add at back        |       O(1) |
| `emplace()` | Construct at back  |       O(1) |
| `pop()`     | Remove front       |       O(1) |
| `front()`   | Access first       |       O(1) |
| `back()`    | Access last        |       O(1) |
| `empty()`   | Check empty        |       O(1) |
| `size()`    | Number of elements |       O(1) |
| `swap()`    | Exchange queues    |       O(1) |

---

# 19. Queue Does Not Support Random Access

You cannot do:

```cpp
q[2]; // ERROR
```

You also cannot use:

```cpp
q.at(2); // ERROR
```

And you cannot directly iterate through a queue using:

```cpp
for (auto x : q) {
}
```

because `std::queue` does not expose `begin()` and `end()`.

This is intentional.

The queue abstraction only exposes:

```text
front
back
push
pop
```

---

# 20. Why Can't We Access the Middle?

Suppose:

```text
front
 ↓
10 → 20 → 30 → 40
                 ↑
                back
```

The queue only promises access to:

```cpp
q.front();
q.back();
```

You cannot directly access:

```text
20
30
```

This protects the FIFO abstraction.

If you need random access, use:

```cpp
std::vector
```

or:

```cpp
std::deque
```

instead.

---

# 21. Processing an Entire Queue

A very common pattern:

```cpp
while (!q.empty()) {

    int x = q.front();

    std::cout << x << ' ';

    q.pop();
}
```

Example:

```text
Queue:
10 → 20 → 30 → 40

Processing:
10
20
30
40

Queue:
empty
```

Important:

This **destroys the queue's contents**.

---

# 22. Checking Before `front()`

Never do:

```cpp
q.front();
```

if the queue may be empty.

Use:

```cpp
if (!q.empty()) {
    std::cout << q.front();
}
```

Similarly:

```cpp
if (!q.empty()) {
    q.pop();
}
```

Calling `front()`, `back()`, or `pop()` on an empty queue is invalid.

---

# 23. Queue Using `std::deque`

By default:

```cpp
std::queue<int> q;
```

is based on:

```cpp
std::deque<int>
```

Conceptually:

```cpp
std::queue<T, std::deque<T>>
```

The underlying container provides the actual storage.

---

# 24. Queue Template Structure

The full template is conceptually:

```cpp
std::queue<
    T,
    Container
>
```

Example:

```cpp
std::queue<int, std::deque<int>> q;
```

Here:

```text
T         = int
Container = deque<int>
```

---

# 25. Using `std::list` as the Underlying Container

You can use:

```cpp
std::queue<int, std::list<int>> q;
```

Example:

```cpp
std::queue<int, std::list<int>> q;

q.push(10);
q.push(20);
q.push(30);
```

It still behaves exactly like a queue:

```text
10 → 20 → 30
```

The underlying container is hidden by the queue interface.

---

# 26. Requirements for the Underlying Container

The underlying container must provide the operations needed by `queue`.

It needs to support:

```text
back()
front()
push_back()
pop_front()
```

This is why common choices are:

```cpp
std::deque
std::list
```

---

# 27. Why Vector Is Not Normally Used

A queue needs efficient:

```text
push_back()
pop_front()
```

`std::vector` has:

```text
push_back() → O(1) amortized
pop_front() → O(n)
```

Therefore, vector is not an appropriate default underlying container for `std::queue`.

A `deque` provides efficient operations at both ends.

---

# 28. `std::queue` vs `std::deque`

These are different abstractions.

### `std::deque`

Provides:

* Random access
* Iterators
* Front insertion
* Back insertion
* Front removal
* Back removal

### `std::queue`

Provides only the FIFO interface:

* `front()`
* `back()`
* `push()`
* `emplace()`
* `pop()`
* `empty()`
* `size()`

Think:

```text
deque = general-purpose container

queue = restricted FIFO interface
```

---

# 29. Queue vs Stack

This is an important distinction.

## Queue

FIFO:

```text
First In → First Out
```

```text
10 → 20 → 30

pop:
10
```

## Stack

LIFO:

```text
Last In → First Out
```

```text
10
20
30 ← top

pop:
30
```

---

# 30. Queue vs Priority Queue

Normal queue:

```text
10 → 20 → 30
```

Processing order:

```text
10
20
30
```

Priority queue:

```text
10, 20, 30
```

Processing depends on priority.

For a max-priority queue:

```text
30
20
10
```

Processing order:

```text
30
20
10
```

---

# 31. Queue Complexity

Typical complexity:

```text
push()       O(1)
emplace()    O(1)
pop()        O(1)
front()      O(1)
back()       O(1)
empty()      O(1)
size()       O(1)
```

This makes queue operations highly efficient.

---

# 32. Queue with Strings

```cpp
std::queue<std::string> names;

names.push("Alice");
names.push("Bob");
names.push("Charlie");
```

Process:

```cpp
while (!names.empty()) {

    std::cout << names.front() << '\n';

    names.pop();
}
```

Output:

```text
Alice
Bob
Charlie
```

---

# 33. Queue with Pairs

```cpp
std::queue<std::pair<int, int>> positions;

positions.push({10, 20});
positions.push({30, 40});
```

Access:

```cpp
auto [x, y] = positions.front();

std::cout << x << ' ' << y;

positions.pop();
```

This is commonly useful for grid/BFS problems.

---

# 34. Queue with Structs

```cpp
struct Task {
    std::string name;
    int priority;
};
```

Then:

```cpp
std::queue<Task> tasks;

tasks.push({"Download", 1});
tasks.push({"Process", 2});
```

Process:

```cpp
while (!tasks.empty()) {

    Task task = tasks.front();

    std::cout << task.name << '\n';

    tasks.pop();
}
```

---

# 35. `emplace()` with Objects

Instead of:

```cpp
tasks.push({"Download", 1});
```

you can use:

```cpp
tasks.emplace("Download", 1);
```

This constructs the object directly in the underlying container.

---

# 36. Classic BFS Pattern

One of the most important applications of `std::queue` is:

**Breadth-First Search (BFS)**.

Example graph traversal:

```cpp
std::queue<int> q;

q.push(start);

while (!q.empty()) {

    int node = q.front();
    q.pop();

    for (int next : graph[node]) {

        if (!visited[next]) {

            visited[next] = true;
            q.push(next);
        }
    }
}
```

Conceptually:

```text
Level 0
   ↓
Level 1
   ↓
Level 2
   ↓
Level 3
```

Queue ensures nodes are processed level by level.

---

# 37. BFS on a Grid

For a grid:

```cpp
std::queue<std::pair<int, int>> q;
```

Start:

```cpp
q.push({startRow, startCol});
```

Then:

```cpp
while (!q.empty()) {

    auto [r, c] = q.front();
    q.pop();

    // process current cell
}
```

This is one of the most common competitive-programming uses of `std::queue`.

---

# 38. Level Order Traversal

Queues are also commonly used for binary tree level-order traversal.

```cpp
std::queue<Node*> q;

q.push(root);

while (!q.empty()) {

    Node* current = q.front();
    q.pop();

    std::cout << current->value;

    if (current->left)
        q.push(current->left);

    if (current->right)
        q.push(current->right);
}
```

The tree is processed level by level.

---

# 39. Task Processing

Queues are useful for jobs/tasks:

```cpp
std::queue<std::string> tasks;

tasks.push("Compile");
tasks.push("Test");
tasks.push("Deploy");
```

Processing:

```cpp
while (!tasks.empty()) {

    std::string task = tasks.front();
    tasks.pop();

    std::cout << "Processing: "
              << task << '\n';
}
```

---

# 40. Customer Service Simulation

```cpp
std::queue<std::string> customers;

customers.push("Customer A");
customers.push("Customer B");
customers.push("Customer C");
```

Service:

```cpp
while (!customers.empty()) {

    std::cout
        << "Serving "
        << customers.front()
        << '\n';

    customers.pop();
}
```

Output:

```text
Serving Customer A
Serving Customer B
Serving Customer C
```

---

# 41. Producer-Consumer Concept

Queues are commonly used to represent work waiting to be processed:

```text
Producer
   ↓
┌───────────────┐
│     Queue     │
│ Task A        │
│ Task B        │
│ Task C        │
└───────────────┘
        ↓
     Consumer
```

Examples include:

* Job systems
* Event processing
* Message processing
* Network packets
* Game events
* Background tasks

For multithreaded applications, `std::queue` itself is **not thread-safe**; synchronization is required when multiple threads access the same queue.

---

# 42. Queue in Game Development

Queues are useful for:

### Event processing

```cpp
std::queue<GameEvent> events;
```

### AI actions

```cpp
std::queue<Action> actions;
```

### Spawn requests

```cpp
std::queue<SpawnRequest> requests;
```

### Network messages

```cpp
std::queue<Message> messages;
```

Conceptually:

```text
Events arrive
     ↓
   Queue
     ↓
Process one by one
```

---

# 43. Queue in Networking

A queue can represent packets waiting for processing:

```text
Incoming packets
       ↓
┌──────────────┐
│ Packet Queue │
└──────────────┘
       ↓
 Processing
```

For example:

```cpp
std::queue<Packet> packetQueue;
```

New packet:

```cpp
packetQueue.push(packet);
```

Process:

```cpp
Packet packet = packetQueue.front();
packetQueue.pop();
```

For production networking systems, synchronization, bounded capacity, backpressure, and thread safety become additional concerns.

---

# 44. Queue and `const`

A const queue can be read but not modified:

```cpp
const std::queue<int> q;
```

You can call:

```cpp
q.empty();
q.size();
q.front();
q.back();
```

But not:

```cpp
q.push(10);
q.pop();
```

---

# 45. Passing Queue to Functions

## By value

```cpp
void process(std::queue<int> q) {
}
```

This copies the queue.

Avoid this if the queue is large and you don't need a copy.

---

## By const reference

```cpp
void inspect(const std::queue<int>& q) {
    std::cout << q.front();
}
```

No copy.

---

## By reference

```cpp
void process(std::queue<int>& q) {

    if (!q.empty()) {
        q.pop();
    }
}
```

The original queue can be modified.

---

# 46. Returning a Queue

You can return a queue from a function:

```cpp
std::queue<int> createQueue() {

    std::queue<int> q;

    q.push(10);
    q.push(20);
    q.push(30);

    return q;
}
```

Modern C++ can return it efficiently through move semantics and return-value optimization.

---

# 47. Copying a Queue

```cpp
std::queue<int> a;

a.push(10);
a.push(20);

std::queue<int> b = a;
```

Now both queues contain:

```text
10 → 20
```

They are independent copies.

---

# 48. Moving a Queue

```cpp
std::queue<int> a;

a.push(10);
a.push(20);

std::queue<int> b = std::move(a);
```

This can transfer the underlying container efficiently.

Include:

```cpp
#include <utility>
```

The moved-from queue remains valid, but don't rely on it retaining its old contents.

---

# 49. Queue Does Not Have Iterators

You cannot do:

```cpp
q.begin(); // ERROR
q.end();   // ERROR
```

This is intentional.

If you need to inspect every element without removing them, consider whether `queue` is the correct abstraction.

If you need iteration and random access, use the underlying container directly, such as:

```cpp
std::deque
```

---

# 50. Queue Does Not Have `clear()`

There is no:

```cpp
q.clear();
```

member function.

A common way to empty a queue is:

```cpp
while (!q.empty()) {
    q.pop();
}
```

However, if you simply want to discard the whole queue object, another approach is:

```cpp
std::queue<int> empty;
q.swap(empty);
```

Or:

```cpp
q = {};
```

when appropriate.

---

# 51. Clearing a Queue

### Method 1

```cpp
while (!q.empty()) {
    q.pop();
}
```

### Method 2

```cpp
std::queue<int> empty;

q.swap(empty);
```

### Method 3

```cpp
q = {};
```

The appropriate approach depends on whether you need to process/destruct elements individually or simply discard the queue.

---

# 52. Common Mistake: `pop()` Before `front()`

Wrong:

```cpp
q.pop();

std::cout << q.front();
```

You may remove the element you wanted to inspect.

Correct:

```cpp
std::cout << q.front();

q.pop();
```

The usual processing order is:

```text
front()
   ↓
process
   ↓
pop()
```

---

# 53. Common Mistake: Calling `front()` on Empty Queue

Wrong:

```cpp
while (true) {
    std::cout << q.front();
    q.pop();
}
```

Eventually the queue becomes empty.

Correct:

```cpp
while (!q.empty()) {

    std::cout << q.front();

    q.pop();
}
```

---

# 54. Common Mistake: Expecting `pop()` to Return a Value

Wrong:

```cpp
int x = q.pop();
```

Correct:

```cpp
int x = q.front();
q.pop();
```

---

# 55. Common Mistake: Expecting Random Access

Wrong:

```cpp
q[5];
```

A queue intentionally doesn't provide this.

If you need indexed access:

```cpp
std::vector
```

or:

```cpp
std::deque
```

may be more appropriate.

---

# 56. Common Mistake: Using Queue for Priority

A normal queue processes based on arrival order.

```text
A arrives
B arrives
C arrives

Processing:
A
B
C
```

If you want:

```text
highest priority first
```

use:

```cpp
std::priority_queue
```

instead.

---

# 57. Queue Mental Model

Think of `std::queue` as:

```text
                  std::queue
                      │
                      ↓
                 FIFO ADAPTER
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
       FRONT                     BACK
          │                       │
       remove                    add
          │                       │
       pop()                    push()
```

The most important operations:

```text
push()
   ↓
back

front()
   ↓
process

pop()
   ↓
remove
```

---

# 58. Queue Workflow

The standard pattern is:

```cpp
while (!q.empty()) {

    auto value = q.front();

    // Process value

    q.pop();
}
```

Memorize this pattern.

It appears constantly in:

* BFS
* Task processing
* Event systems
* Simulations
* Scheduling
* Message processing

---

# 59. `std::queue` Function Reference

```text
┌───────────────────────────────────────┐
│           std::queue API              │
├───────────────────────────────────────┤
│ push()                                │
│ emplace()                             │
│ pop()                                 │
│ front()                               │
│ back()                                │
│ empty()                               │
│ size()                                │
│ swap()                                │
└───────────────────────────────────────┘
```

That's intentionally much smaller than the API of `vector` or `list`.

---

# 60. Practical Example

```cpp
#include <iostream>
#include <queue>
#include <string>

int main() {

    std::queue<std::string> tasks;

    tasks.push("Load Game");
    tasks.push("Connect Server");
    tasks.push("Load Player");
    tasks.push("Start Match");

    while (!tasks.empty()) {

        const std::string& task = tasks.front();

        std::cout << "Processing: "
                  << task
                  << '\n';

        tasks.pop();
    }

    return 0;
}
```

Output:

```text
Processing: Load Game
Processing: Connect Server
Processing: Load Player
Processing: Start Match
```

---

# 61. Practice Problems

## Beginner

1. Create a queue of integers.
2. Add 10 numbers.
3. Print the front element.
4. Print the back element.
5. Remove elements one by one.
6. Check whether the queue is empty.
7. Count the number of elements.
8. Process a queue of strings.

---

## Intermediate

1. Implement BFS on a graph.
2. Implement BFS on a 2D grid.
3. Implement binary-tree level-order traversal.
4. Simulate a customer service queue.
5. Simulate a printer queue.
6. Simulate task processing.
7. Process incoming messages.
8. Use `std::pair` with a queue.

---

# 62. Important Concepts Checklist

Before moving on from `std::queue`, understand:

* [ ] Container adaptor
* [ ] FIFO
* [ ] `push()`
* [ ] `emplace()`
* [ ] `pop()`
* [ ] `front()`
* [ ] `back()`
* [ ] `empty()`
* [ ] `size()`
* [ ] `swap()`
* [ ] No random access
* [ ] No iterators
* [ ] No `clear()` member
* [ ] Underlying container
* [ ] `deque` as default underlying container
* [ ] `list` as an alternative underlying container
* [ ] Why vector is not suitable as the default
* [ ] Queue vs stack
* [ ] Queue vs priority queue
* [ ] FIFO processing
* [ ] BFS
* [ ] Level-order traversal
* [ ] Task processing
* [ ] Event queues
* [ ] Packet/message queues
* [ ] Thread-safety considerations
* [ ] Complexity

---

# 63. Recommended Learning Order

```text
std::queue
    │
    ├── Container Adaptor
    │
    ├── FIFO
    │
    ├── push()
    │
    ├── emplace()
    │
    ├── front()
    │
    ├── back()
    │
    ├── pop()
    │
    ├── empty()
    │
    ├── size()
    │
    ├── swap()
    │
    ├── Underlying Container
    │
    ├── deque vs list
    │
    ├── Queue vs Stack
    │
    ├── Queue vs Priority Queue
    │
    └── BFS / Real-world Applications
```

---

# 64. Key Takeaway

The core idea of `std::queue` is extremely simple:

```text
              FIFO
               │
       First In → First Out
               │
               ↓
        ┌──────────────┐
        │    QUEUE     │
        └──────────────┘
          ↑          ↓
        push        pop
        back       front
```

Remember:

```cpp
q.push(x);      // Add to back
q.front();      // See first
q.back();       // See last
q.pop();        // Remove first
q.empty();      // Check empty
q.size();       // Number of elements
```

The most important processing pattern is:

```cpp
while (!q.empty()) {

    auto value = q.front();

    // process value

    q.pop();
}
```

And the most important application to master is:

```text
Queue
  ↓
BFS
  ↓
Level-by-level processing
```

`std::queue` is not designed to be a general-purpose container. It is a **restricted interface that enforces FIFO behavior**, making it ideal whenever elements must be processed in the same order they arrive.
