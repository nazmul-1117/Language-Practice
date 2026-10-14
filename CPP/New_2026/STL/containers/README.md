# STL Containers

STL containers are standard C++ data structures used to store and organize collections of objects.

The main STL containers can be divided into four categories:

```text
STL Containers
│
├── Sequence Containers
│   ├── vector
│   ├── array
│   ├── deque
│   ├── list
│   └── forward_list
│
├── Associative Containers
│   ├── map
│   ├── set
│   ├── multimap
│   └── multiset
│
├── Unordered Associative Containers
│   ├── unordered_map
│   ├── unordered_set
│   ├── unordered_multimap
│   └── unordered_multiset
│
└── Container Adaptors
    ├── stack
    ├── queue
    └── priority_queue
```

---

# 1. Sequence Containers

Sequence containers store elements in a linear sequence.

| Container           | Main Idea             | Random Access |
| ------------------- | --------------------- | ------------- |
| `std::vector`       | Dynamic array         | ✅             |
| `std::array`        | Fixed-size array      | ✅             |
| `std::deque`        | Double-ended sequence | ✅             |
| `std::list`         | Doubly linked list    | ❌             |
| `std::forward_list` | Singly linked list    | ❌             |

### `std::vector`

Dynamic, contiguous array.

```cpp
std::vector<int> numbers;
```

**Best for:** General-purpose dynamic collections and fast random access.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── vector/
        └── README.md
```

---

### `std::array`

Fixed-size container.

```cpp
std::array<int, 5> numbers;
```

**Best for:** Collections whose size is known at compile time.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── array/
        └── README.md
```

---

### `std::deque`

Double-ended queue.

```cpp
std::deque<int> numbers;
```

**Best for:** Efficient insertion and removal at both ends.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── deque/
        └── README.md
```

---

### `std::list`

Doubly linked list.

```cpp
std::list<int> numbers;
```

**Best for:** Frequent insertion/removal when you already have an iterator to the position.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── list/
        └── README.md
```

---

### `std::forward_list`

Singly linked list.

```cpp
std::forward_list<int> numbers;
```

**Best for:** Lightweight forward-only linked-list operations.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── forward_list/
        └── README.md
```

---

# 2. Associative Containers

Associative containers organize elements using keys and maintain sorted order.

| Container       | Stores      | Duplicate Keys |
| --------------- | ----------- | -------------- |
| `std::map`      | Key → Value | ❌              |
| `std::set`      | Keys        | ❌              |
| `std::multimap` | Key → Value | ✅              |
| `std::multiset` | Keys        | ✅              |

---

### `std::map`

Stores key-value pairs with unique keys.

```cpp
std::map<std::string, int> ages;
```

Example:

```text
Alice   → 25
Bob     → 30
Charlie → 22
```

**Best for:** Ordered key-value data with unique keys.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── map/
        └── README.md
```

---

### `std::set`

Stores unique values in sorted order.

```cpp
std::set<int> numbers;
```

**Best for:** Maintaining a collection of unique sorted values.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── set/
        └── README.md
```

---

### `std::multimap`

Stores key-value pairs where multiple elements can have the same key.

```cpp
std::multimap<std::string, int> students;
```

**Best for:** One-to-many relationships with ordered keys.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── multimap/
        └── README.md
```

---

### `std::multiset`

Stores values in sorted order and allows duplicates.

```cpp
std::multiset<int> numbers;
```

**Best for:** Ordered collections where duplicate values are allowed.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── multiset/
        └── README.md
```

---

# 3. Unordered Associative Containers

Unordered containers use hashing and do **not** maintain sorted order.

| Container                 | Stores      | Duplicate Keys |
| ------------------------- | ----------- | -------------- |
| `std::unordered_map`      | Key → Value | ❌              |
| `std::unordered_set`      | Keys        | ❌              |
| `std::unordered_multimap` | Key → Value | ✅              |
| `std::unordered_multiset` | Keys        | ✅              |

---

### `std::unordered_map`

Hash-based key-value container.

```cpp
std::unordered_map<std::string, int> ages;
```

**Best for:** Fast average key-based lookup when ordering is not required.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── unordered_map/
        └── README.md
```

---

### `std::unordered_set`

Hash-based container of unique values.

```cpp
std::unordered_set<int> numbers;
```

**Best for:** Fast average membership testing with unique values.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── unordered_set/
        └── README.md
```

---

### `std::unordered_multimap`

Hash-based key-value container that allows duplicate keys.

```cpp
std::unordered_multimap<std::string, int> data;
```

**Best for:** Unordered one-to-many key-value relationships.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── unordered_multimap/
        └── README.md
```

---

### `std::unordered_multiset`

Hash-based container that allows duplicate values.

```cpp
std::unordered_multiset<int> numbers;
```

**Best for:** Fast average lookup where duplicates are allowed and ordering is unnecessary.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── unordered_multiset/
        └── README.md
```

---

# 4. Container Adaptors

Container adaptors provide restricted interfaces for specific data structures or behaviors.

```text
Container Adaptors
│
├── stack
├── queue
└── priority_queue
```

---

### `std::stack`

Implements **LIFO**:

> Last In, First Out

```cpp
std::stack<int> numbers;
```

**Best for:** Stack-based processing such as undo operations and DFS.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── stack/
        └── README.md
```

---

### `std::queue`

Implements **FIFO**:

> First In, First Out

```cpp
std::queue<int> numbers;
```

**Best for:** First-come-first-served processing and BFS.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── queue/
        └── README.md
```

---

### `std::priority_queue`

Provides access to the highest-priority element.

```cpp
std::priority_queue<int> numbers;
```

**Best for:** Priority-based processing, scheduling, and priority algorithms.

📁 Detailed documentation:

```text
stl/
└── containers/
    └── priority_queue/
        └── README.md
```

---

# 5. Quick Comparison

```text
Need a dynamic array?
        ↓
      vector

Need a fixed-size array?
        ↓
      array

Need efficient operations at both ends?
        ↓
      deque

Need linked-list behavior?
        ↓
   list / forward_list

Need ordered key → value?
        ↓
      map

Need unique ordered values?
        ↓
      set

Need fast average key lookup?
        ↓
  unordered_map

Need fast average membership lookup?
        ↓
  unordered_set

Need LIFO?
        ↓
      stack

Need FIFO?
        ↓
      queue

Need priority-based access?
        ↓
 priority_queue
```

---

# 6. Container Selection

Choosing a container should depend on the operations your program performs.

Consider:

```text
1. Do I need ordering?
2. Do I need unique elements?
3. Do I need key-value relationships?
4. Do I need random access?
5. How often will I insert?
6. How often will I delete?
7. How often will I search?
8. Is memory layout important?
9. Do I need average O(1) lookup?
10. Do I need a specific access pattern?
```

The goal is not to memorize every container.

The goal is to understand:

> **Why would I choose one container over another?**

---

# 7. Container Directory

The detailed documentation for each container will be maintained separately:

```text
stl/
└── containers/
    │
    ├── README.md
    │
    ├── vector/
    │   └── README.md
    │
    ├── array/
    │   └── README.md
    │
    ├── deque/
    │   └── README.md
    │
    ├── list/
    │   └── README.md
    │
    ├── forward_list/
    │   └── README.md
    │
    ├── map/
    │   └── README.md
    │
    ├── set/
    │   └── README.md
    │
    ├── multimap/
    │   └── README.md
    │
    ├── multiset/
    │   └── README.md
    │
    ├── unordered_map/
    │   └── README.md
    │
    ├── unordered_set/
    │   └── README.md
    │
    ├── unordered_multimap/
    │   └── README.md
    │
    ├── unordered_multiset/
    │   └── README.md
    │
    ├── stack/
    │   └── README.md
    │
    ├── queue/
    │   └── README.md
    │
    └── priority_queue/
        └── README.md
```

---

# Summary

STL containers can be grouped into:

```text
Sequence
    vector
    array
    deque
    list
    forward_list

Associative
    map
    set
    multimap
    multiset

Unordered Associative
    unordered_map
    unordered_set
    unordered_multimap
    unordered_multiset

Container Adaptors
    stack
    queue
    priority_queue
```

This README provides the **overview and navigation**.

Detailed concepts, syntax, operations, complexity, examples, iterator behavior, and practical use cases should be documented inside each container's individual README.
