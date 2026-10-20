# STL Sets

> **Path:** `stl/containers/sets/README.md`

C++ provides four main set containers:

```text
set
unordered_set
multiset
unordered_multiset
```

The most important idea is:

> A **set stores unique values**.

Unlike `map`, a set does not store:

```text
Key → Value
```

It stores only:

```text
Value
```

Example:

```text
101
102
103
```

---

# 1. The Four Set Types

| Container            | Unique? | Ordered? | Main Structure |
| -------------------- | ------- | -------- | -------------- |
| `set`                | Yes     | Yes      | Balanced tree  |
| `unordered_set`      | Yes     | No       | Hash table     |
| `multiset`           | No      | Yes      | Balanced tree  |
| `unordered_multiset` | No      | No       | Hash table     |

For your learning path, remember:

```text
set
→ unique + sorted

unordered_set
→ unique + unordered

multiset
→ duplicates + sorted

unordered_multiset
→ duplicates + unordered
```

---

# 2. Why Do We Need a Set?

Suppose you have player IDs:

```text
101
102
103
101
102
```

You only want to know:

> Which unique players exist?

A set automatically removes duplicates:

```text
101
102
103
```

This is the main purpose of a set.

---

# 3. `std::set`

Header:

```cpp
#include <set>
```

Basic syntax:

```cpp
std::set<int> ids;
```

Example:

```cpp
#include <iostream>
#include <set>

int main() {

    std::set<int> ids;

    ids.insert(30);
    ids.insert(10);
    ids.insert(20);
    ids.insert(10);

    for (int id : ids) {
        std::cout << id << '\n';
    }

    return 0;
}
```

Output:

```text
10
20
30
```

Notice two things:

1. `10` appeared only once.
2. Values are sorted.

---

# 4. Main Properties of `set`

```text
std::set
```

provides:

```text
✓ Unique values
✓ Sorted values
✓ Fast search
✓ Fast insertion
✓ Fast deletion
```

Typical complexity:

```text
O(log n)
```

---

# 5. `set` Insertion

Use:

```cpp
ids.insert(100);
```

Example:

```cpp
std::set<int> ids;

ids.insert(10);
ids.insert(20);
ids.insert(30);
```

Result:

```text
10
20
30
```

If you insert:

```cpp
ids.insert(20);
```

again, nothing new is added.

---

# 6. Checking Whether a Value Exists

## `find()`

```cpp
auto it = ids.find(20);
```

Then:

```cpp
if (it != ids.end()) {
    std::cout << "Found\n";
}
```

---

# 7. `contains()`

In C++20:

```cpp
if (ids.contains(20)) {
    std::cout << "Found\n";
}
```

This is often the cleanest way to ask:

> Does this value exist?

Complexity:

```text
O(log n)
```

for `set`.

---

# 8. `count()`

You can also use:

```cpp
if (ids.count(20)) {
    std::cout << "Found\n";
}
```

For `set`, the result is:

```text
0 → not found
1 → found
```

because duplicate values are not allowed.

---

# 9. Removing Values

Remove by value:

```cpp
ids.erase(20);
```

Example:

```cpp
std::set<int> ids = {10, 20, 30};

ids.erase(20);
```

Result:

```text
10
30
```

Complexity:

```text
O(log n)
```

---

# 10. Iterating Through a `set`

```cpp
for (int value : ids) {
    std::cout << value << '\n';
}
```

Because `set` is ordered, values appear in sorted order.

Example:

```text
Input:

50
10
30
20

Output:

10
20
30
50
```

---

# 11. `set` Does Not Have Indexing

You cannot do:

```cpp
ids[0];
```

This is invalid.

A set is not designed for:

```text
index → value
```

It is designed for:

```text
value → does it exist?
```

Think:

```text
vector
→ position/index

set
→ membership
```

---

# 12. `unordered_set`

Header:

```cpp
#include <unordered_set>
```

Syntax:

```cpp
std::unordered_set<int> ids;
```

Example:

```cpp
std::unordered_set<int> ids;

ids.insert(30);
ids.insert(10);
ids.insert(20);
ids.insert(10);
```

The values are unique:

```text
10
20
30
```

but their iteration order is **not guaranteed**.

You might see:

```text
20
10
30
```

or another order.

---

# 13. `set` vs `unordered_set`

This is one of the most important STL decisions.

### `set`

```text
Unique
+
Sorted
+
O(log n)
```

### `unordered_set`

```text
Unique
+
No ordering
+
Average O(1)
```

Comparison:

| Feature          | `set`    | `unordered_set`     |
| ---------------- | -------- | ------------------- |
| Duplicates       | No       | No                  |
| Ordering         | Sorted   | No guaranteed order |
| Search           | O(log n) | Average O(1)        |
| Insert           | O(log n) | Average O(1)        |
| Erase            | O(log n) | Average O(1)        |
| Range operations | Yes      | No natural ordering |
| Main structure   | Tree     | Hash table          |

---

# 14. When Should You Use `set`?

Use `set` when:

```text
You need unique values
+
You need them ordered
```

Example:

```text
Player IDs:

101
105
110
120
```

You may want them automatically sorted.

---

# 15. When Should You Use `unordered_set`?

Use `unordered_set` when:

```text
You need unique values
+
You don't care about ordering
```

Example:

```text
Connected player IDs
```

You mainly want to ask:

```text
Is player 105 connected?
```

You don't care whether iteration produces:

```text
101 105 110
```

or:

```text
110 101 105
```

In this situation, `unordered_set` is often the natural choice.

---

# 16. Networking Example — Connected Players

Suppose a server needs to keep track of connected player IDs.

```cpp
std::unordered_set<int> connectedPlayers;
```

When a player connects:

```cpp
connectedPlayers.insert(101);
```

When checking:

```cpp
if (connectedPlayers.contains(101)) {
    std::cout << "Player is connected\n";
}
```

When disconnecting:

```cpp
connectedPlayers.erase(101);
```

Conceptually:

```text
Player connects
      ↓
unordered_set
      ↓
Player ID stored
      ↓
Fast membership check
```

This is a very useful pattern for networking.

---

# 17. Networking Example — Unique Packet IDs

Suppose you want to remember packet IDs that have already been processed.

```cpp
std::unordered_set<int> processedPackets;
```

When a packet arrives:

```cpp
int packetId = 5001;

if (!processedPackets.contains(packetId)) {

    // Process packet

    processedPackets.insert(packetId);
}
```

Conceptually:

```text
Packet arrives
     ↓
Already processed?
   /       \
 Yes       No
 ↓          ↓
Ignore    Process
             ↓
           Store ID
```

This is a good example of **set = membership tracking**.

---

# 18. `multiset`

Header:

```cpp
#include <set>
```

A `multiset` is like a `set`, except:

> Duplicate values are allowed.

Example:

```cpp
std::multiset<int> scores;

scores.insert(100);
scores.insert(100);
scores.insert(80);
scores.insert(80);
scores.insert(80);
```

The result is:

```text
80
80
80
100
100
```

It is:

```text
Ordered
+
Duplicates allowed
```

---

# 19. `set` vs `multiset`

### `set`

```text
10
20
30
```

Duplicate:

```text
10
10
```

becomes:

```text
10
```

### `multiset`

```text
10
10
20
30
```

Duplicates remain.

---

# 20. Why Use `multiset`?

Use `multiset` when:

```text
Duplicates are meaningful
+
You still want sorted data
```

For example:

```text
Player scores:

100
100
90
80
80
```

You might want to keep every score while maintaining sorted order.

Another conceptual use:

```text
Priority values
Event values
Scores
Measurements
```

---

# 21. Counting Duplicates

This is especially useful with `multiset`.

```cpp
std::multiset<int> scores = {
    100, 100, 90, 80, 80, 80
};
```

Then:

```cpp
scores.count(80);
```

returns:

```text
3
```

Because `80` appears three times.

---

# 22. `unordered_multiset`

Header:

```cpp
#include <unordered_set>
```

Syntax:

```cpp
std::unordered_multiset<int> values;
```

It allows:

```text
Duplicates
+
No ordering
```

Example:

```cpp
std::unordered_multiset<int> values;

values.insert(10);
values.insert(10);
values.insert(20);
values.insert(20);
```

Both copies remain.

But iteration order is not guaranteed.

---

# 23. `multiset` vs `unordered_multiset`

| Feature    | `multiset` | `unordered_multiset` |
| ---------- | ---------- | -------------------- |
| Duplicates | Yes        | Yes                  |
| Ordering   | Sorted     | No                   |
| Search     | O(log n)   | Average O(1)         |
| Insert     | O(log n)   | Average O(1)         |
| Erase      | O(log n)   | Average O(1)         |
| Structure  | Tree       | Hash table           |

Remember:

```text
multiset
→ duplicates + sorted

unordered_multiset
→ duplicates + unordered
```

---

# 24. Important Set Functions

For `set` and `unordered_set`, the most important functions are:

```cpp
insert()
emplace()
find()
contains()
count()
erase()
clear()
size()
empty()
```

Example:

```cpp
std::set<int> values;

values.insert(10);
values.emplace(20);

values.contains(10);

values.find(20);

values.erase(10);

values.size();

values.empty();

values.clear();
```

---

# 25. `lower_bound()` — Ordered Sets

`set` supports:

```cpp
lower_bound()
upper_bound()
```

Example:

```cpp
std::set<int> values = {
    10, 20, 30, 40
};
```

Then:

```cpp
auto it = values.lower_bound(25);
```

returns:

```text
30
```

because `30` is the first value:

```text
>= 25
```

---

# 26. `upper_bound()` — Ordered Sets

```cpp
auto it = values.upper_bound(20);
```

returns:

```text
30
```

because it finds the first value:

```text
> 20
```

Remember:

```text
lower_bound(x)
→ first value >= x

upper_bound(x)
→ first value > x
```

These operations are important when working with **ranges and ordered data**.

---

# 27. Range Queries

Suppose:

```cpp
std::set<int> values = {
    10, 20, 30, 40, 50
};
```

You want values from:

```text
20 → 40
```

You can use:

```cpp
auto start = values.lower_bound(20);
auto end = values.upper_bound(40);

for (auto it = start; it != end; ++it) {
    std::cout << *it << '\n';
}
```

Output:

```text
20
30
40
```

This is one reason to choose an ordered `set` instead of an `unordered_set`.

---

# 28. Custom Ordering

A `set` can use a custom comparator.

Example:

```cpp
std::set<int, std::greater<int>> values;
```

Now values are stored in descending order.

```text
50
40
30
20
10
```

This is useful when you need a different ordering rule.

---

# 29. Sets of Strings

Sets aren't limited to numbers.

```cpp
std::set<std::string> names;
```

Example:

```cpp
names.insert("Charlie");
names.insert("Alice");
names.insert("Bob");
```

Iteration:

```text
Alice
Bob
Charlie
```

This is useful for maintaining a unique collection of names/identifiers.

---

# 30. Set of Pairs

You can use:

```cpp
std::set<std::pair<int, int>> positions;
```

Example:

```cpp
positions.insert({10, 20});
positions.insert({30, 40});
```

This can represent unique coordinates:

```text
(10, 20)
(30, 40)
```

A pair is compared lexicographically by default.

---

# 31. Set of Custom Structs

You can store your own types, but the set needs a way to determine ordering for `std::set`.

For example:

```cpp
struct Player {
    int id;
};
```

You could provide a comparator:

```cpp
struct PlayerCompare {
    bool operator()(const Player& a,
                    const Player& b) const {
        return a.id < b.id;
    }
};
```

Then:

```cpp
std::set<Player, PlayerCompare> players;
```

For `unordered_set`, you need a suitable hash function and equality comparison.

This is more advanced; understand the concept first before worrying about implementation details.

---

# 32. Set Complexity

## `set`

| Operation       | Complexity |
| --------------- | ---------: |
| `insert()`      |   O(log n) |
| `find()`        |   O(log n) |
| `contains()`    |   O(log n) |
| `erase()`       |   O(log n) |
| `lower_bound()` |   O(log n) |
| `upper_bound()` |   O(log n) |
| `size()`        |       O(1) |
| `empty()`       |       O(1) |

## `unordered_set`

| Operation    | Average |
| ------------ | ------: |
| `insert()`   |    O(1) |
| `find()`     |    O(1) |
| `contains()` |    O(1) |
| `erase()`    |    O(1) |
| `size()`     |    O(1) |

Worst-case hash-table operations can degrade to:

```text
O(n)
```

Don't forget that.

---

# 33. Memory and Performance

A simplified mental model:

### `set`

```text
       50
      /  \
    20    80
```

Tree-based structure.

Advantages:

```text
✓ Ordered
✓ Range queries
✓ lower_bound()
✓ upper_bound()
```

### `unordered_set`

Think:

```text
Hash
 ↓
Bucket
 ↓
Value
```

Advantages:

```text
✓ Very fast average lookup
✓ Good for membership checks
✓ No need for ordering
```

---

# 34. `set` vs `unordered_set` for Networking

This is an important decision.

### Scenario 1

You need:

```text
Is this player connected?
```

Ordering doesn't matter.

Use:

```cpp
std::unordered_set<int>
```

### Scenario 2

You need:

```text
All player IDs sorted
```

Use:

```cpp
std::set<int>
```

### Scenario 3

You need:

```text
Find all IDs in a range
```

Use:

```cpp
std::set<int>
```

because of:

```cpp
lower_bound()
upper_bound()
```

---

# 35. `set` vs `unordered_set` in Game Development

Imagine active objects:

```text
1001
1005
1010
1050
```

If you only need:

```text
Is object 1010 active?
```

use:

```cpp
std::unordered_set<int>
```

If you need:

```text
Process object IDs in sorted order
```

use:

```cpp
std::set<int>
```

The important skill is not memorizing the containers.

It is asking:

> **Do I need ordering?**

---

# 36. `set` vs `vector`

These are very different.

### Vector

```text
[10, 20, 30, 40]
 0   1   2   3
```

Designed for:

```text
Index → Value
```

### Set

```text
10
20
30
40
```

Designed for:

```text
Does this value exist?
```

Use:

```text
vector → sequence
set    → unique membership
```

---

# 37. `set` vs `map`

Remember:

### `set`

```text
Value
```

Example:

```text
101
102
103
```

### `map`

```text
Key → Value
```

Example:

```text
101 → Alice
102 → Bob
103 → Charlie
```

Mental model:

```text
set
→ "Does this exist?"

map
→ "What data belongs to this key?"
```

---

# 38. `unordered_set` vs `unordered_map`

Same basic relationship.

### `unordered_set`

```text
Value
```

Used for membership.

### `unordered_map`

```text
Key → Value
```

Used for lookup.

Example:

```text
unordered_set:

101
102
103
```

versus:

```text
unordered_map:

101 → Alice
102 → Bob
103 → Charlie
```

---

# 39. Networking Example — Active Connections

A simple conceptual server:

```cpp
std::unordered_set<int> activeConnections;
```

Connect:

```cpp
activeConnections.insert(connectionId);
```

Check:

```cpp
if (activeConnections.contains(connectionId)) {
    // Connection exists
}
```

Disconnect:

```cpp
activeConnections.erase(connectionId);
```

This is exactly the kind of problem where a set makes sense:

> We only care whether the ID exists.

---

# 40. Networking Example — Processed Packets

```cpp
std::unordered_set<int> processedPackets;
```

When a packet arrives:

```cpp
if (processedPackets.contains(packetId)) {
    // Already processed
    return;
}

processedPackets.insert(packetId);

// Process packet
```

The important idea:

```text
Packet ID
   ↓
Already seen?
   ↓
Yes → ignore
No  → process
```

---

# 41. Game Example — Collected Items

Suppose the player has collected unique item IDs:

```cpp
std::unordered_set<int> collectedItems;
```

Collect:

```cpp
collectedItems.insert(itemId);
```

Check:

```cpp
if (collectedItems.contains(itemId)) {
    std::cout << "Already collected\n";
}
```

Again:

```text
Set = unique membership
```

---

# 42. Game Example — Unique Tags

Imagine:

```text
Player tags:

"Player"
"Enemy"
"Flying"
"Boss"
```

A set can keep tags unique:

```cpp
std::unordered_set<std::string> tags;

tags.insert("Player");
tags.insert("Enemy");
tags.insert("Boss");
```

Trying to add `"Enemy"` again doesn't create another copy.

This concept is closely related to **tag/component-style systems** in game development.

---

# 43. Unreal Engine Connection

In Unreal Engine, you will encounter:

```cpp
TSet<T>
```

The STL concept that transfers is:

```text
std::set
std::unordered_set
      ↓
Unique collection
      ↓
TSet
```

Don't assume the implementations and performance characteristics are identical.

What you should carry into Unreal is the **data-structure idea**:

```text
Unique values
Membership checking
Insert
Remove
Iteration
Hashing
Ordering vs no ordering
```

---

# 44. What You Need for Unreal

Prioritize:

```text
★★★★★

✓ What is a set?
✓ Unique values
✓ Membership checking
✓ insert()
✓ erase()
✓ find()
✓ contains()
✓ set vs unordered_set
✓ O(log n) vs average O(1)
✓ Why ordering matters
```

Then:

```text
★★★★☆

✓ multiset
✓ unordered_multiset
✓ count()
✓ lower_bound()
✓ upper_bound()
✓ Custom comparators
✓ Hashing concept
```

Lower priority:

```text
★★☆☆☆

✗ Rare advanced allocator details
✗ Complex custom hashing implementation
✗ Reimplementing set from scratch
```

---

# 45. Four Set Types — Quick Decision

Use this:

```text
                 Do duplicates matter?
                       /        \
                     No          Yes
                     /            \
               Need ordering?   Need ordering?
                 /    \           /    \
               Yes    No        Yes    No
                |      |          |      |
               set  unordered_  multiset unordered_
                     set                  multiset
```

Simpler:

```text
set
→ unique + sorted

unordered_set
→ unique + unordered

multiset
→ duplicate + sorted

unordered_multiset
→ duplicate + unordered
```

---

# 46. Most Important Functions

### `set`

```cpp
insert()
emplace()
find()
contains()
count()
erase()
clear()
size()
empty()
lower_bound()
upper_bound()
```

### `unordered_set`

```cpp
insert()
emplace()
find()
contains()
count()
erase()
clear()
size()
empty()
```

Remember:

> `lower_bound()` and `upper_bound()` are for ordered associative containers such as `set`.

---

# 47. Common Mistakes

## Mistake 1 — Expecting duplicates

```cpp
std::set<int> values;

values.insert(10);
values.insert(10);
```

There is still only:

```text
10
```

---

## Mistake 2 — Expecting indexing

Wrong:

```cpp
values[0];
```

A set has no index-based access.

---

## Mistake 3 — Assuming `unordered_set` is sorted

Don't rely on:

```text
1
2
3
4
```

Iteration order is not guaranteed.

---

## Mistake 4 — Using `set` when ordering doesn't matter

If you only need:

```text
Does this exist?
```

consider:

```cpp
std::unordered_set
```

---

## Mistake 5 — Using `unordered_set` when range queries matter

If you need:

```text
values between 100 and 200
```

an ordered `set` is much more appropriate.

---

# 48. Practical Example

```cpp
#include <iostream>
#include <unordered_set>

int main() {

    std::unordered_set<int> connectedPlayers;

    connectedPlayers.insert(101);
    connectedPlayers.insert(102);
    connectedPlayers.insert(103);

    if (connectedPlayers.contains(102)) {
        std::cout << "Player 102 is connected\n";
    }

    connectedPlayers.erase(102);

    if (!connectedPlayers.contains(102)) {
        std::cout << "Player 102 disconnected\n";
    }

    return 0;
}
```

This example represents a realistic concept:

```text
Connection ID
      ↓
unordered_set
      ↓
Is connection active?
```

---

# 49. Quick Complexity Cheat Sheet

```text
                 Ordered       Unordered
                 set           unordered_set

Unique           ✓             ✓

Search           O(log n)      Average O(1)

Insert           O(log n)      Average O(1)

Erase            O(log n)      Average O(1)

Sorted           ✓             ✗

Range Query      ✓             ✗
```

For duplicates:

```text
multiset
→ same complexity style as set

unordered_multiset
→ same average complexity style as unordered_set
```

---

# 50. Final Mental Model

Don't memorize four containers separately.

Think about two questions:

```text
Question 1:
Can duplicates exist?

YES → multi
NO  → normal


Question 2:
Do I need ordering?

YES → ordered
NO  → unordered
```

Therefore:

```text
                 Duplicates?

             NO             YES
             │               │
       ┌─────┴─────┐   ┌─────┴─────┐
       │           │   │           │
    Ordered     Unordered      Ordered    Unordered
       │           │              │          │
      set    unordered_set     multiset  unordered_multiset
```

---

# 51. One-Minute Revision

```text
set
→ unique
→ sorted
→ O(log n)

unordered_set
→ unique
→ unordered
→ average O(1)

multiset
→ duplicates
→ sorted
→ O(log n)

unordered_multiset
→ duplicates
→ unordered
→ average O(1)
```

### Networking

```text
Connected IDs
Processed packet IDs
Active sessions
Unique users
Unique resources
```

### Game Development

```text
Collected item IDs
Active object IDs
Unique tags
Visited nodes
Unique states
```

### Golden Rule

> **Use a set when you care about whether a value exists, not where it is.**

> **Use `set` when you need uniqueness + ordering.**

> **Use `unordered_set` when you need uniqueness + fast average membership lookup and don't care about ordering.**
