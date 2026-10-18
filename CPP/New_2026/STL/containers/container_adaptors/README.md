# STL Container Adaptors

> **Path:** `stl/container_adaptors/README.md`

Container adaptors provide a simple interface for managing data in different ways.

For your **C++ → Networking → Unreal Engine** roadmap, the most important ones are:

* `std::queue` → FIFO
* `std::deque` → both ends
* `std::stack` → LIFO
* `std::priority_queue` → highest priority first

You don't need to memorize every function. Focus on **how they work, when to use them, and their complexity**.

---

# 1. Queue

Header:

```cpp
#include <queue>
```

A queue follows:

```text
FIFO
First In → First Out
```

Example:

```text
Input:

10 → 20 → 30

Remove:

10 → 20 → 30
```

The first item added is the first item removed.

## Basic Usage

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
}
```

Output:

```text
10
20
```

## Important Functions

| Function    | Purpose              | Complexity |
| ----------- | -------------------- | ---------: |
| `push()`    | Add to back          |       O(1) |
| `emplace()` | Construct at back    |       O(1) |
| `front()`   | Get first element    |       O(1) |
| `back()`    | Get last element     |       O(1) |
| `pop()`     | Remove first element |       O(1) |
| `empty()`   | Check empty          |       O(1) |
| `size()`    | Number of elements   |       O(1) |

### Important

`pop()` does **not** return the removed value.

Wrong:

```cpp
int x = q.pop();
```

Correct:

```cpp
int x = q.front();
q.pop();
```

## Typical Processing Pattern

```cpp
while (!q.empty()) {
    auto item = q.front();

    // Process item

    q.pop();
}
```

This pattern is extremely important.

---

# 2. Queue in Networking

Queues are very useful when data must be processed in order.

For example:

```text
Incoming Messages

Message 1
Message 2
Message 3
Message 4

        ↓

      Queue

        ↓

Process Message 1
Process Message 2
Process Message 3
Process Message 4
```

A simplified example:

```cpp
std::queue<std::string> messages;

messages.push("Login");
messages.push("Move");
messages.push("Attack");

while (!messages.empty()) {
    std::string message = messages.front();
    messages.pop();

    std::cout << "Processing: " << message << '\n';
}
```

### Networking concepts to connect with

Think about queues when you learn:

* Incoming messages
* Outgoing messages
* Packets waiting for processing
* Events waiting to be handled
* Request processing
* Producer/consumer systems

> **Networking mental model:**
> Data arrives → Queue → Process data.

---

# 3. Deque

Header:

```cpp
#include <deque>
```

`deque` means:

> **Double-Ended Queue**

It allows insertion and removal from **both ends**.

```text
        Front
          ↓
      10 20 30 40
          ↑
         Back
```

You can:

```text
push_front()
push_back()

pop_front()
pop_back()
```

## Basic Usage

```cpp
#include <deque>

std::deque<int> d;

d.push_back(10);
d.push_back(20);

d.push_front(5);

d.pop_front();
d.pop_back();
```

## Important Functions

| Function       | Purpose            | Complexity |
| -------------- | ------------------ | ---------: |
| `push_front()` | Add to front       |       O(1) |
| `push_back()`  | Add to back        |       O(1) |
| `pop_front()`  | Remove front       |       O(1) |
| `pop_back()`   | Remove back        |       O(1) |
| `front()`      | Access front       |       O(1) |
| `back()`       | Access back        |       O(1) |
| `operator[]`   | Random access      |       O(1) |
| `empty()`      | Check empty        |       O(1) |
| `size()`       | Number of elements |       O(1) |

---

# 4. Queue vs Deque

This distinction is important.

### Queue

```text
Add → Back
Remove → Front
```

```text
10 20 30

push(40)

10 20 30 40
↑           ↑
front       back
```

You normally don't care about both ends.

### Deque

```text
Add → Front OR Back
Remove → Front OR Back
```

```text
10 20 30

push_front(5)

5 10 20 30

push_back(40)

5 10 20 30 40
```

### Simple rule

> Need strict FIFO? → `queue`

> Need both ends? → `deque`

---

# 5. Deque in Networking / Game Systems

A deque can be useful when recent and older data need different handling.

For example, conceptually:

```text
Recent events
      ↓
[ newest | ... | oldest ]
                     ↑
              remove old data
```

It can be useful for things such as:

* Sliding windows
* Recent event history
* Buffered data
* Message/event buffers
* Time-based data
* Gameplay history

Don't force `deque` into every problem. Use it when **both ends matter**.

---

# 6. Stack

Header:

```cpp
#include <stack>
```

A stack follows:

```text
LIFO
Last In → First Out
```

Example:

```text
push(10)
push(20)
push(30)

        ↓

30 ← first out
20
10
```

The last element added is removed first.

## Basic Usage

```cpp
#include <stack>

std::stack<int> s;

s.push(10);
s.push(20);
s.push(30);

std::cout << s.top() << '\n';

s.pop();
```

Output:

```text
30
```

## Important Functions

| Function    | Purpose            | Complexity |
| ----------- | ------------------ | ---------: |
| `push()`    | Add element        |       O(1) |
| `emplace()` | Construct element  |       O(1) |
| `top()`     | Access top         |       O(1) |
| `pop()`     | Remove top         |       O(1) |
| `empty()`   | Check empty        |       O(1) |
| `size()`    | Number of elements |       O(1) |

### Important

Like `queue`, `stack::pop()` returns `void`.

Correct:

```cpp
int value = s.top();
s.pop();
```

---

# 7. Stack Use Cases

Stacks are useful when the **most recent item should be handled first**.

Common examples:

* Undo operations
* Backtracking
* Function-call style processing
* Expression evaluation
* Depth-First Search (DFS)
* Temporary state management

For example:

```text
Game State A
     ↓
Game State B
     ↓
Game State C
```

If you want to return to the most recent state:

```text
C → B → A
```

That's stack behavior.

---

# 8. Stack and Networking

Stack is generally **less important than queue for networking**.

However, understanding LIFO is important because some algorithms and state-based systems naturally use it.

For example:

```text
Packet/Event processing
        ↓
Need newest item first
        ↓
      Stack
```

For normal incoming packet/message processing, a queue is usually the more natural mental model.

---

# 9. Priority Queue

Header:

```cpp
#include <queue>
```

A priority queue is different from a normal queue.

Normal queue:

```text
First In → First Out
```

Priority queue:

```text
Highest Priority → First Out
```

Example:

```text
Task A → priority 2
Task B → priority 10
Task C → priority 5
```

Processing order:

```text
Task B
Task C
Task A
```

Because:

```text
10 > 5 > 2
```

---

# 10. Basic Priority Queue

```cpp
#include <queue>

std::priority_queue<int> pq;

pq.push(10);
pq.push(30);
pq.push(20);

std::cout << pq.top() << '\n';

pq.pop();
```

Output:

```text
30
```

By default, the **largest element has the highest priority**.

## Important Functions

| Function    | Purpose                  | Complexity |
| ----------- | ------------------------ | ---------: |
| `push()`    | Add element              |   O(log n) |
| `emplace()` | Add element              |   O(log n) |
| `top()`     | Highest-priority element |       O(1) |
| `pop()`     | Remove highest priority  |   O(log n) |
| `empty()`   | Check empty              |       O(1) |
| `size()`    | Number of elements       |       O(1) |

---

# 11. Min Priority Queue

By default:

```cpp
std::priority_queue<int> pq;
```

gives the **largest value first**.

If you want the smallest value first:

```cpp
std::priority_queue<
    int,
    std::vector<int>,
    std::greater<int>
> pq;
```

Now:

```text
10
20
30
```

will be processed as:

```text
10 → 20 → 30
```

---

# 12. Priority Queue in Networking

Priority queues are particularly useful when different tasks/data have different importance.

Imagine:

```text
Incoming Tasks

Normal data       Priority 2
Game update       Priority 5
Important event   Priority 10
```

A priority queue allows the important task to be processed first.

Conceptually:

```text
                Incoming Data
                      ↓
               Priority Queue
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
     High Priority          Low Priority
          ↓                       ↓
       Process                Wait
```

This is useful to understand for:

* Network events
* Task scheduling
* Game events
* AI tasks
* Pathfinding
* Resource management
* Event prioritization

---

# 13. Priority Queue in Game Development

Suppose a game has events:

```text
Player Death       Priority 100
Boss Attack        Priority 80
Enemy Movement     Priority 30
Ambient Sound      Priority 10
```

The priority queue can process:

```text
Player Death
Boss Attack
Enemy Movement
Ambient Sound
```

This is the basic idea behind **priority-based processing**.

You don't need to implement an entire game system with `priority_queue` yet. Just understand the concept.

---

# 14. Queue vs Stack vs Deque vs Priority Queue

This is the most important comparison.

| Container        | Rule             | Add               | Remove           | Main Use            |
| ---------------- | ---------------- | ----------------- | ---------------- | ------------------- |
| `queue`          | FIFO             | Back              | Front            | Ordered processing  |
| `stack`          | LIFO             | Top               | Top              | Backtracking/state  |
| `deque`          | Both ends        | Front/Back        | Front/Back       | Flexible buffering  |
| `priority_queue` | Highest priority | According to heap | Highest priority | Priority processing |

### Remember

```text
queue
FIFO
↓
First comes → First processed


stack
LIFO
↓
Last comes → First processed


deque
Both ends
↓
Front + Back


priority_queue
Priority
↓
Most important → First processed
```

---

# 15. Complexity Cheat Sheet

```text
                Insert       Access       Remove
---------------------------------------------------
queue            O(1)        O(1)          O(1)

deque            O(1)        O(1)          O(1)
                 ends        ends          ends

stack            O(1)        O(1)          O(1)

priority_queue   O(log n)    O(1)          O(log n)
```

For `priority_queue`:

```text
top()  → O(1)
push() → O(log n)
pop()  → O(log n)
```

---

# 16. Choosing the Right One

Use this simple decision process.

```text
Do you need ordered processing?
        |
       Yes
        |
   FIFO required?
      /     \
    Yes      No
     |        |
   queue   Need priority?
             /     \
           Yes      No
            |        |
    priority_queue  stack/deque
```

For both-end operations:

```text
Need to add/remove from both ends?
        |
       Yes
        |
      deque
```

---

# 17. Networking Mental Model

For your networking learning path, remember these four ideas:

### Normal message processing

```text
Messages
   ↓
 queue
   ↓
Process in arrival order
```

### Priority-based processing

```text
Messages
   ↓
priority_queue
   ↓
Process important messages first
```

### Recent/old data management

```text
Data
 ↓
deque
 ↓
Manage both ends
```

### Reverse / backtracking logic

```text
States
  ↓
stack
  ↓
Most recent state first
```

---

# 18. Unreal Engine Connection

You will encounter similar ideas in game programming even when Unreal Engine uses its own types and systems.

The important concepts to carry into Unreal are:

```text
FIFO
LIFO
Priority
Buffering
Event processing
Task processing
State management
```

For example:

### Event Queue

```text
Input/Event
     ↓
Queue
     ↓
Process events
```

### Task Priority

```text
Tasks
 ↓
Priority Queue
 ↓
Important task first
```

### State History

```text
Current State
     ↓
Previous State
     ↓
Older State

Stack-like thinking
```

The goal is **not** to memorize `std::queue` specifically for Unreal.

The goal is to understand the underlying data-structure concepts so you can recognize them when working with Unreal's containers and systems.

---

# 19. Common Mistakes

## Mistake 1 — Expecting `pop()` to return a value

Wrong:

```cpp
int x = q.pop();
```

Correct:

```cpp
int x = q.front();
q.pop();
```

For stack:

```cpp
int x = s.top();
s.pop();
```

---

## Mistake 2 — Accessing an empty container

Don't do:

```cpp
q.front();
```

without checking.

Better:

```cpp
if (!q.empty()) {
    std::cout << q.front();
}
```

For stack:

```cpp
if (!s.empty()) {
    std::cout << s.top();
}
```

---

## Mistake 3 — Using `queue` when you need priorities

If you have:

```text
Task A → Priority 1
Task B → Priority 10
```

A normal queue processes:

```text
A → B
```

A priority queue can process:

```text
B → A
```

---

## Mistake 4 — Using `stack` when you need FIFO

If messages must be processed in arrival order:

```text
Message 1
Message 2
Message 3
```

Don't use a stack.

Use:

```cpp
std::queue
```

---

# 20. Practical Example

Imagine a simple game server receiving events:

```cpp
#include <iostream>
#include <queue>
#include <string>

int main() {
    std::queue<std::string> events;

    events.push("Player Connected");
    events.push("Player Moved");
    events.push("Player Attacked");

    while (!events.empty()) {
        std::cout << events.front() << '\n';
        events.pop();
    }

    return 0;
}
```

Processing order:

```text
Player Connected
Player Moved
Player Attacked
```

This is the basic idea of **FIFO event processing**.

---

# 21. What You Actually Need to Learn

For your **C++ → Networking → Unreal Engine** roadmap, prioritize these:

### Must Know

```text
✓ FIFO vs LIFO
✓ queue
✓ stack
✓ deque
✓ priority_queue
✓ push / pop
✓ front / back
✓ top
✓ empty / size
✓ Basic complexity
✓ When to choose each one
```

### Important for Networking

```text
✓ Message queues
✓ Packet/event processing
✓ FIFO processing
✓ Priority-based processing
✓ Buffering
```

### Important for Game Development

```text
✓ Event processing
✓ Task scheduling
✓ State management
✓ Priority systems
✓ Buffers
```

### Don't spend too much time on

```text
✗ Memorizing every overload
✗ Rare advanced tricks
✗ Implementing STL containers from scratch
✗ Using a container without understanding why
```

---

# 22. Quick Revision

```text
┌──────────────────────────────────────────┐
│              QUEUE                       │
│                                          │
│ FIFO                                     │
│ First In → First Out                     │
│                                          │
│ push() → back                            │
│ pop()  → front                           │
│ front() → first element                  │
└──────────────────────────────────────────┘


┌──────────────────────────────────────────┐
│              STACK                       │
│                                          │
│ LIFO                                     │
│ Last In → First Out                      │
│                                          │
│ push() → top                             │
│ pop()  → top                             │
│ top()  → top element                     │
└──────────────────────────────────────────┘


┌──────────────────────────────────────────┐
│              DEQUE                       │
│                                          │
│ Double-ended                             │
│                                          │
│ push_front()                             │
│ push_back()                              │
│ pop_front()                              │
│ pop_back()                               │
└──────────────────────────────────────────┘


┌──────────────────────────────────────────┐
│          PRIORITY QUEUE                  │
│                                          │
│ Highest priority → First                 │
│                                          │
│ push() → O(log n)                        │
│ top()  → O(1)                            │
│ pop()  → O(log n)                        │
└──────────────────────────────────────────┘
```

---

# 23. Final Mental Model

Don't memorize four separate containers.

Remember this:

```text
                 How should data leave?

                        │
          ┌─────────────┼─────────────┐
          │             │             │
         FIFO          LIFO        Priority
          │             │             │
        queue         stack    priority_queue
          │
          │
      Both ends?
          │
         deque
```

### The four words to remember

```text
queue          → FIFO
stack          → LIFO
deque          → BOTH ENDS
priority_queue → PRIORITY
```

For your networking and Unreal Engine journey, **understanding these four concepts is much more important than memorizing every STL function.**
