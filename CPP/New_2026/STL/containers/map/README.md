# `std::map`

> **Path:** `stl/containers/map/README.md`

`std::map` is an STL associative container that stores data as **key-value pairs**.

The most important idea is:

```text
Key → Value
```

Example:

```text
Player ID → Player Name

101 → "Alice"
102 → "Bob"
103 → "Charlie"
```

Instead of searching through every element manually, you use the **key** to find the associated value.

---

# 1. Why Do We Need `map`?

Suppose you have:

```cpp
int playerId;
std::string playerName;
```

With many players, you might need:

```text
101 → Alice
102 → Bob
103 → Charlie
104 → David
```

A `map` lets you store this relationship directly:

```cpp
std::map<int, std::string> players;
```

Then:

```cpp
players[101]
```

gives:

```text
Alice
```

This is the basic purpose of a map:

> **Use one piece of data (key) to find another piece of data (value).**

---

# 2. Include Header

```cpp
#include <map>
```

Usually you will also need:

```cpp
#include <iostream>
#include <string>
```

Example:

```cpp
#include <iostream>
#include <map>
#include <string>

int main() {
    std::map<int, std::string> players;

    players[101] = "Alice";
    players[102] = "Bob";

    std::cout << players[101] << '\n';

    return 0;
}
```

Output:

```text
Alice
```

---

# 3. Basic Syntax

```cpp
std::map<KeyType, ValueType> mapName;
```

Example:

```cpp
std::map<int, std::string> players;
```

Here:

```text
Key   = int
Value = std::string
```

Another example:

```cpp
std::map<std::string, int> scores;
```

```text
Player Name → Score

"Alice" → 100
"Bob"   → 85
```

---

# 4. Key and Value

Every element has two parts:

```text
Key → Value
```

Example:

```cpp
std::map<int, std::string> players;

players[101] = "Alice";
```

Here:

```text
101   = key
Alice = value
```

The key identifies the element.

The value is the information associated with that key.

---

# 5. Important Property: Keys Are Unique

A normal `std::map` cannot contain duplicate keys.

Example:

```cpp
std::map<int, std::string> players;

players[101] = "Alice";
players[101] = "Bob";
```

The second assignment changes the value.

The map contains:

```text
101 → Bob
```

not:

```text
101 → Alice
101 → Bob
```

If you need duplicate keys, use:

```cpp
std::multimap
```

---

# 6. `map` Is Sorted

A `std::map` keeps its elements sorted by **key**.

Example:

```cpp
std::map<int, std::string> players;

players[103] = "Charlie";
players[101] = "Alice";
players[102] = "Bob";
```

Iteration produces:

```text
101 → Alice
102 → Bob
103 → Charlie
```

Even though you inserted:

```text
103
101
102
```

the keys are stored in sorted order.

By default:

```text
smallest key → largest key
```

---

# 7. Creating a Map

### Empty map

```cpp
std::map<int, std::string> players;
```

### With initial values

```cpp
std::map<int, std::string> players = {
    {101, "Alice"},
    {102, "Bob"},
    {103, "Charlie"}
};
```

You can also write:

```cpp
std::map<int, std::string> players{
    {101, "Alice"},
    {102, "Bob"},
    {103, "Charlie"}
};
```

---

# 8. Insert Using `operator[]`

The easiest way to insert:

```cpp
std::map<int, std::string> players;

players[101] = "Alice";
players[102] = "Bob";
```

Now:

```text
101 → Alice
102 → Bob
```

### Important behavior

If the key already exists:

```cpp
players[101] = "Charlie";
```

the old value is replaced.

Now:

```text
101 → Charlie
```

---

# 9. `insert()`

You can also use:

```cpp
players.insert({101, "Alice"});
```

Example:

```cpp
std::map<int, std::string> players;

players.insert({101, "Alice"});
players.insert({102, "Bob"});
```

---

# 10. Important Difference: `[]` vs `insert()`

This is very important.

### Using `[]`

```cpp
players[101] = "Alice";
```

If `101` doesn't exist:

```text
Create it
```

If it already exists:

```text
Replace its value
```

### Using `insert()`

```cpp
players.insert({101, "Alice"});
```

If `101` already exists:

```text
Do not replace it
```

Example:

```cpp
std::map<int, std::string> players;

players.insert({101, "Alice"});
players.insert({101, "Bob"});
```

Result:

```text
101 → Alice
```

---

# 11. `emplace()`

`emplace()` constructs the element directly.

```cpp
players.emplace(101, "Alice");
```

Example:

```cpp
std::map<int, std::string> players;

players.emplace(101, "Alice");
players.emplace(102, "Bob");
```

For normal usage, knowing `insert()` and `emplace()` is enough.

---

# 12. Checking Whether a Key Exists

## `find()`

```cpp
auto it = players.find(101);
```

If found:

```cpp
if (it != players.end()) {
    std::cout << it->second;
}
```

If not found:

```cpp
it == players.end()
```

### Example

```cpp
std::map<int, std::string> players = {
    {101, "Alice"},
    {102, "Bob"}
};

auto it = players.find(101);

if (it != players.end()) {
    std::cout << "Found: " << it->second << '\n';
}
```

---

# 13. `contains()`

Modern C++ provides:

```cpp
players.contains(101)
```

Example:

```cpp
if (players.contains(101)) {
    std::cout << "Player exists\n";
}
```

This is available from **C++20**.

For your modern C++ learning, remember:

```text
contains() → simple existence check
find()     → existence + access to iterator
```

---

# 14. `count()`

For a normal `map`, a key can exist only once.

Therefore:

```cpp
players.count(101)
```

returns either:

```text
0 → doesn't exist
1 → exists
```

Example:

```cpp
if (players.count(101)) {
    std::cout << "Found\n";
}
```

For `std::map`, `contains()` is generally clearer when you only need a yes/no answer.

---

# 15. Accessing Values

There are several ways.

## `operator[]`

```cpp
std::cout << players[101];
```

But there is an important danger.

If the key doesn't exist:

```cpp
players[999]
```

can create a new element with a default value.

For example:

```cpp
std::map<int, std::string> players;

std::cout << players[999];
```

This can create:

```text
999 → ""
```

So don't use `[]` just to check whether a key exists.

Use:

```cpp
players.contains(999)
```

or:

```cpp
players.find(999)
```

---

# 16. `at()`

`at()` accesses an existing key.

```cpp
std::cout << players.at(101);
```

If the key doesn't exist, `at()` throws:

```text
std::out_of_range
```

Example:

```cpp
try {
    std::cout << players.at(999);
}
catch (const std::out_of_range& e) {
    std::cout << "Player not found\n";
}
```

### Simple rule

```text
[]   → can create a key
at() → expects the key to exist
```

---

# 17. `front()` and `back()`

`std::map` provides:

```cpp
front()
back()
```

in modern C++.

They refer to the first and last elements according to key ordering.

Example:

```cpp
std::map<int, std::string> players = {
    {101, "Alice"},
    {102, "Bob"},
    {103, "Charlie"}
};
```

Conceptually:

```text
front → 101 → Alice
back  → 103 → Charlie
```

These are less commonly needed than `find()` and iteration.

---

# 18. Iterating Through a Map

The most common method:

```cpp
for (const auto& pair : players) {
    std::cout << pair.first
              << " → "
              << pair.second
              << '\n';
}
```

Output:

```text
101 → Alice
102 → Bob
103 → Charlie
```

Remember:

```text
pair.first  → key
pair.second → value
```

---

# 19. Structured Bindings

Modern C++ provides a cleaner way:

```cpp
for (const auto& [id, name] : players) {
    std::cout << id << " → " << name << '\n';
}
```

This is extremely useful.

Instead of:

```cpp
pair.first
pair.second
```

you directly get:

```text
id
name
```

For modern C++ code, prefer this style when appropriate.

---

# 20. Iterators

A map iterator points to a key-value pair.

```cpp
auto it = players.begin();
```

Access:

```cpp
it->first
it->second
```

Example:

```cpp
for (auto it = players.begin();
     it != players.end();
     ++it) {

    std::cout << it->first
              << " → "
              << it->second
              << '\n';
}
```

---

# 21. Reverse Iteration

Use:

```cpp
rbegin()
rend()
```

Example:

```cpp
for (auto it = players.rbegin();
     it != players.rend();
     ++it) {

    std::cout << it->first
              << " → "
              << it->second
              << '\n';
}
```

This goes from the largest key toward the smallest.

---

# 22. `erase()`

Remove by key:

```cpp
players.erase(101);
```

Example:

```cpp
std::map<int, std::string> players = {
    {101, "Alice"},
    {102, "Bob"},
    {103, "Charlie"}
};

players.erase(102);
```

Now:

```text
101 → Alice
103 → Charlie
```

---

# 23. Erase Using Iterator

```cpp
auto it = players.find(102);

if (it != players.end()) {
    players.erase(it);
}
```

This is useful when you already have an iterator.

---

# 24. `clear()`

Remove everything:

```cpp
players.clear();
```

After:

```cpp
players.empty()
```

returns:

```text
true
```

---

# 25. `empty()`

Check whether the map has no elements:

```cpp
if (players.empty()) {
    std::cout << "Map is empty\n";
}
```

Complexity:

```text
O(1)
```

---

# 26. `size()`

Get the number of elements:

```cpp
std::cout << players.size();
```

Example:

```text
3
```

---

# 27. `lower_bound()`

This is one of the most important advanced `map` operations.

```cpp
auto it = players.lower_bound(102);
```

It returns an iterator to the **first key that is greater than or equal to** the given key.

Example:

```text
Keys:

101
105
110
```

Then:

```cpp
players.lower_bound(105)
```

returns:

```text
105
```

But:

```cpp
players.lower_bound(103)
```

returns:

```text
105
```

because 105 is the first key `>= 103`.

---

# 28. `upper_bound()`

```cpp
auto it = players.upper_bound(105);
```

It returns the first key **greater than** the given key.

For:

```text
101
105
110
```

```cpp
upper_bound(105)
```

returns:

```text
110
```

---

# 29. `lower_bound()` vs `upper_bound()`

Remember:

```text
lower_bound(x)
→ first key >= x

upper_bound(x)
→ first key > x
```

These are useful when working with **ranges and ordered data**.

---

# 30. Custom Sorting Order

By default:

```cpp
std::map<int, std::string>
```

sorts keys in ascending order.

You can provide a comparator.

For descending order:

```cpp
std::map<
    int,
    std::string,
    std::greater<int>
> players;
```

Now:

```text
103 → Charlie
102 → Bob
101 → Alice
```

---

# 31. Custom Key Types

A map key doesn't have to be `int` or `string`.

You can use types such as:

```cpp
std::pair
```

or your own class/struct.

Example:

```cpp
std::map<std::pair<int, int>, std::string> positions;
```

This could represent:

```text
(x, y) → Object
```

Example:

```cpp
positions[{10, 20}] = "Enemy";
positions[{30, 40}] = "Player";
```

---

# 32. Map With Struct Values

This is very useful for game/networking systems.

```cpp
struct Player {
    std::string name;
    int health;
    int score;
};
```

Then:

```cpp
std::map<int, Player> players;
```

Example:

```cpp
players[101] = {"Alice", 100, 500};
players[102] = {"Bob", 80, 350};
```

Now:

```cpp
std::cout << players[101].name;
std::cout << players[101].health;
```

---

# 33. Map of Objects

You can store larger objects as values:

```cpp
std::map<int, Player> players;
```

Conceptually:

```text
Player ID
    ↓
Player Object
```

This is a very useful pattern.

---

# 34. Nested Maps

A map can contain another map.

```cpp
std::map<int, std::map<int, int>> data;
```

Conceptually:

```text
Player
  ↓
  Match
    ↓
    Score
```

Example:

```cpp
data[101][1] = 500;
data[101][2] = 700;
```

Meaning:

```text
Player 101
    Match 1 → 500
    Match 2 → 700
```

Don't overuse nested maps, but understand the concept.

---

# 35. Map With `std::vector`

You can combine containers:

```cpp
std::map<int, std::vector<std::string>> teams;
```

Example:

```cpp
teams[1].push_back("Alice");
teams[1].push_back("Bob");

teams[2].push_back("Charlie");
```

Conceptually:

```text
Team 1 → [Alice, Bob]
Team 2 → [Charlie]
```

This combination appears frequently in real programs.

---

# 36. Map With Smart Pointers

You can also store smart pointers:

```cpp
std::map<int, std::unique_ptr<Player>> players;
```

This is useful when the map owns dynamically allocated objects.

For example:

```cpp
players[101] = std::make_unique<Player>();
```

This is more advanced, but the pattern is worth recognizing.

---

# 37. Map for Networking

`std::map` is useful when you have an identifier and need to find associated information.

For example:

```text
Connection ID → Connection
Player ID     → Player
Packet ID     → Packet
Request ID    → Request
Session ID    → Session
```

Example:

```cpp
std::map<int, std::string> connections;

connections[1001] = "Client A";
connections[1002] = "Client B";
connections[1003] = "Client C";
```

Then:

```cpp
auto it = connections.find(1002);

if (it != connections.end()) {
    std::cout << "Connected to: "
              << it->second;
}
```

The mental model is:

```text
ID
 ↓
Find associated data
```

---

# 38. Example: Player Lookup

Imagine a multiplayer game.

Each player has an ID:

```text
101
102
103
104
```

You want to find players quickly by ID.

```cpp
std::map<int, Player> players;
```

Then:

```cpp
auto it = players.find(playerId);
```

If found:

```cpp
if (it != players.end()) {
    Player& player = it->second;

    // Work with player
}
```

This is a very common associative-container pattern.

---

# 39. Example: Network Request Tracking

Suppose each request has an ID:

```text
Request ID → Request data
```

You can represent it as:

```cpp
std::map<int, Request> requests;
```

When a request arrives:

```cpp
requests[request.id] = request;
```

Later:

```cpp
auto it = requests.find(requestId);
```

You can locate the corresponding request.

---

# 40. Example: Game Object Lookup

```cpp
std::map<int, std::string> objects;

objects[1] = "Player";
objects[2] = "Enemy";
objects[3] = "Weapon";
```

Now:

```cpp
int objectId = 2;

auto it = objects.find(objectId);

if (it != objects.end()) {
    std::cout << "Object: " << it->second;
}
```

This is the basic idea behind **ID → object lookup**.

---

# 41. `map` Complexity

This is extremely important.

`std::map` is normally implemented as a **balanced binary search tree**, commonly a Red-Black Tree.

Therefore the main operations are:

```text
O(log n)
```

### Complexity

| Operation       | Complexity |
| --------------- | ---------: |
| `insert()`      |   O(log n) |
| `emplace()`     |   O(log n) |
| `find()`        |   O(log n) |
| `contains()`    |   O(log n) |
| `erase(key)`    |   O(log n) |
| `lower_bound()` |   O(log n) |
| `upper_bound()` |   O(log n) |
| `operator[]`    |   O(log n) |
| `at()`          |   O(log n) |
| `size()`        |       O(1) |
| `empty()`       |       O(1) |
| `clear()`       |       O(n) |

---

# 42. Why Is `map` O(log n)?

Imagine sorted data:

```text
        50
       /  \
     25    75
    / \    / \
   10 30 60 90
```

To find:

```text
60
```

you don't need to check everything.

You compare:

```text
60 vs 50
```

Go right.

Then:

```text
60 vs 75
```

Go left.

Then:

```text
60
```

Found.

The tree keeps reducing the search space.

That's why lookup is approximately:

```text
O(log n)
```

---

# 43. `map` vs `unordered_map`

This is one of the most important STL decisions for your networking roadmap.

## `map`

```text
Ordered
Tree-based
O(log n)
```

## `unordered_map`

```text
Unordered
Hash-table based
Average O(1)
```

### Comparison

| Feature               | `map`           | `unordered_map`                |
| --------------------- | --------------- | ------------------------------ |
| Ordering              | Sorted          | No ordering                    |
| Main structure        | Tree            | Hash table                     |
| Search                | O(log n)        | Average O(1)                   |
| Insert                | O(log n)        | Average O(1)                   |
| Erase                 | O(log n)        | Average O(1)                   |
| Range operations      | Excellent       | Not naturally ordered          |
| Memory/cache behavior | Usually heavier | Often better for direct lookup |
| Custom hash needed    | No              | Sometimes                      |
| Predictable ordering  | Yes             | No                             |

---

# 44. When Should You Use `map`?

Use `map` when you need:

### 1. Sorted keys

```text
101
102
103
104
```

### 2. Range queries

For example:

```text
Find all IDs from 100 to 200
```

### 3. `lower_bound()` / `upper_bound()`

### 4. Ordered iteration

```cpp
for (const auto& [key, value] : data)
```

### 5. You specifically need tree-based ordering

---

# 45. When Should You Use `unordered_map`?

If you mainly need:

```text
key → value
```

and don't care about ordering, `unordered_map` is often the more natural choice.

For example:

```text
Player ID → Player
```

where you simply need to find a player by ID.

For your networking learning:

> Learn both, but understand **why** you choose one over the other.

---

# 46. Important: Don't Assume `map` Is Always Better

A common beginner mistake is:

> "Map is fast, so I should use map."

Not necessarily.

The question is:

```text
Do I need ordered keys?
```

If:

```text
Yes → map
No  → consider unordered_map
```

This decision becomes very important in larger systems.

---

# 47. `map` vs `vector`

These containers solve different problems.

### Vector

Good for:

```text
Index → Value
```

Example:

```cpp
players[0]
players[1]
players[2]
```

### Map

Good for:

```text
Key → Value
```

Example:

```cpp
players[1001]
players[4502]
players[9877]
```

Use `vector` when you have sequential/index-based data.

Use `map` when you have meaningful keys.

---

# 48. `map` vs `set`

This distinction is simple.

### Map

```text
Key → Value
```

Example:

```text
101 → Alice
```

### Set

```text
Unique values only
```

Example:

```text
101
102
103
```

Think:

```text
map → association
set → membership
```

---

# 49. Iterator Invalidation

For `std::map`, iterators are relatively stable.

Inserting another element generally does **not** invalidate existing iterators.

Example:

```cpp
auto it = players.find(101);

players.insert({200, "New Player"});

// it can still be used
```

Erasing the specific element invalidates its iterator:

```cpp
players.erase(it);
```

After that, don't use that iterator.

This stability is one advantage of tree-based associative containers.

---

# 50. Swapping Maps

```cpp
map1.swap(map2);
```

Example:

```cpp
std::map<int, std::string> a;
std::map<int, std::string> b;

a.swap(b);
```

The contents are exchanged.

---

# 51. Copying a Map

```cpp
std::map<int, std::string> a;

a[1] = "Alice";
a[2] = "Bob";

auto b = a;
```

Now `b` contains a copy.

---

# 52. Moving a Map

```cpp
auto b = std::move(a);
```

This transfers the resources from `a` to `b`.

After moving:

```text
b → owns the transferred data
a → valid but unspecified state
```

You should not assume `a` still contains its previous elements.

---

# 53. `map` and `const`

If you only need to read:

```cpp
const std::map<int, std::string> players = {
    {101, "Alice"},
    {102, "Bob"}
};
```

You can iterate:

```cpp
for (const auto& [id, name] : players) {
    std::cout << id << " → " << name << '\n';
}
```

Using `const` helps prevent accidental modification.

---

# 54. A Useful Real-World Pattern

One of the most useful patterns to remember is:

```cpp
auto it = data.find(key);

if (it != data.end()) {
    // Found
    auto& value = it->second;
}
```

This works for many associative containers.

For example:

```cpp
std::map<int, Player> players;

auto it = players.find(playerId);

if (it != players.end()) {
    Player& player = it->second;

    // Use player
}
```

Memorize the **pattern**, not every variation.

---

# 55. Practical Example

```cpp
#include <iostream>
#include <map>
#include <string>

struct Player {
    std::string name;
    int health;
};

int main() {

    std::map<int, Player> players;

    players.emplace(101, Player{"Alice", 100});
    players.emplace(102, Player{"Bob", 80});
    players.emplace(103, Player{"Charlie", 90});

    for (const auto& [id, player] : players) {
        std::cout << id << " → "
                  << player.name
                  << " | HP: "
                  << player.health
                  << '\n';
    }

    int searchId = 102;

    auto it = players.find(searchId);

    if (it != players.end()) {
        std::cout << "Found: "
                  << it->second.name
                  << '\n';
    }

    return 0;
}
```

Output:

```text
101 → Alice | HP: 100
102 → Bob | HP: 80
103 → Charlie | HP: 90

Found: Bob
```

---

# 56. Common Mistakes

## Mistake 1 — Using `[]` for existence checking

Avoid:

```cpp
if (players[101]) {
    ...
}
```

For many value types this can create the key.

Prefer:

```cpp
players.contains(101)
```

or:

```cpp
players.find(101) != players.end()
```

---

## Mistake 2 — Forgetting that keys are unique

```cpp
players[101] = "Alice";
players[101] = "Bob";
```

This doesn't create two players.

It changes:

```text
Alice → Bob
```

---

## Mistake 3 — Expecting insertion order

If you insert:

```text
300
100
200
```

iteration produces:

```text
100
200
300
```

`map` is ordered by key, **not insertion order**.

---

## Mistake 4 — Choosing `map` when ordering isn't needed

If you only need fast key lookup:

```text
key → value
```

consider:

```cpp
std::unordered_map
```

---

## Mistake 5 — Thinking `map` is O(1)

Normal `std::map` operations such as lookup and insertion are:

```text
O(log n)
```

not O(1).

---

# 57. What You Need to Know for Networking

Prioritize these concepts:

```text
★★★★★

1. Key → Value
2. Unique keys
3. Sorted keys
4. find()
5. contains()
6. insert()
7. erase()
8. iteration
9. O(log n)
10. map vs unordered_map
```

Then learn:

```text
★★★★☆

11. lower_bound()
12. upper_bound()
13. Custom comparator
14. Iterator behavior
15. Map with structs/classes
```

Lower priority:

```text
★★☆☆☆

16. Allocators
17. Rare advanced comparator techniques
18. Complex nested maps
```

---

# 58. What You Need to Know for Unreal Engine

The most important concept is:

```text
ID → Object
```

Examples:

```text
Player ID    → Player
Actor ID     → Actor
Item ID      → Item
Request ID   → Request
Session ID   → Session
```

This pattern appears everywhere in software development.

You should become comfortable reading code like:

```cpp
auto it = players.find(playerId);

if (it != players.end()) {
    // use the player
}
```

Also understand the equivalent Unreal-style idea:

```text
Key → Value
```

because Unreal has its own associative containers, such as `TMap`.

The STL knowledge transfers conceptually.

---

# 59. STL `std::map` vs Unreal `TMap`

In Unreal Engine, you will commonly encounter:

```cpp
TMap<KeyType, ValueType>
```

Conceptually:

```text
std::map
    ↓
Key → Value
    ↓
TMap
```

Do not assume they are identical implementations or have identical performance characteristics.

The important transferable knowledge is:

```text
Associative container
Key
Value
Lookup
Insert
Remove
Iteration
Hash/tree concepts
Complexity
```

---

# 60. Quick Cheat Sheet

```text
std::map<Key, Value>
```

### Create

```cpp
std::map<int, std::string> data;
```

### Insert / update

```cpp
data[101] = "Alice";
```

### Insert without overwriting

```cpp
data.insert({101, "Alice"});
```

### Emplace

```cpp
data.emplace(101, "Alice");
```

### Find

```cpp
auto it = data.find(101);
```

### Check

```cpp
data.contains(101);
```

### Access

```cpp
data[101];
data.at(101);
```

### Delete

```cpp
data.erase(101);
```

### Size

```cpp
data.size();
```

### Empty

```cpp
data.empty();
```

### Clear

```cpp
data.clear();
```

### Iterate

```cpp
for (const auto& [key, value] : data) {
    // ...
}
```

### Range

```cpp
data.lower_bound(key);
data.upper_bound(key);
```

---

# 61. Final Mental Model

Don't think of `map` as just another STL container.

Think:

```text
             MAP

        Key → Value
           |
           ↓
     Find by key
           |
           ↓
      Ordered keys
           |
           ↓
      O(log n) lookup
```

For your networking/game-development path:

```text
Player ID
    ↓
Player

Request ID
    ↓
Request

Connection ID
    ↓
Connection

Object ID
    ↓
Object
```

That **Key → Value** relationship is the most important thing to understand.

---

# 62. One-Minute Revision

```text
std::map

✓ Key → Value
✓ Keys are unique
✓ Keys are sorted
✓ Usually balanced tree
✓ Lookup → O(log n)
✓ Insert → O(log n)
✓ Erase → O(log n)

Important functions:

insert()
emplace()
find()
contains()
at()
operator[]
erase()
clear()
size()
empty()
lower_bound()
upper_bound()

Main comparison:

map
→ ordered
→ O(log n)

unordered_map
→ unordered
→ average O(1)
```

### Golden Rule

> **Use `std::map` when you need a sorted key → value relationship.**

> **Use `std::unordered_map` when you mainly need key → value lookup and ordering doesn't matter.**
