#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;


/*
============================================================
                    STL HEAP ALGORITHMS
============================================================

This program demonstrates:

1. make_heap()
2. push_heap()
3. pop_heap()
4. sort_heap()

------------------------------------------------------------
WHAT IS A HEAP?
------------------------------------------------------------

A heap is a special data structure that follows a
specific ordering rule.

By default, C++ STL creates a:

                    MAX HEAP

In a max heap:

    The largest element is always at the front.

Example:

    Data:
        10 30 20 5 40

    After make_heap():

        40 30 20 5 10
         ↑
       largest

IMPORTANT:
    A heap is NOT completely sorted.

It only guarantees that the largest element is at:

        v.front()

------------------------------------------------------------
1. make_heap()
------------------------------------------------------------

Converts an existing range into a heap.

Syntax:

    make_heap(begin, end);

Time Complexity:

    O(n)

------------------------------------------------------------
2. push_heap()
------------------------------------------------------------

Adds the last element into an existing heap.

Typical process:

    v.push_back(value);
    push_heap(v.begin(), v.end());

IMPORTANT:
    push_heap() assumes that the range before the
    newly added element is already a valid heap.

Time Complexity:

    O(log n)

------------------------------------------------------------
3. pop_heap()
------------------------------------------------------------

Moves the largest element from the front to the end
of the heap.

IMPORTANT:

    pop_heap() does NOT remove the element from vector.

It only moves it.

Therefore, we normally use:

    pop_heap(v.begin(), v.end());
    v.pop_back();

Time Complexity:

    O(log n)

------------------------------------------------------------
4. sort_heap()
------------------------------------------------------------

Converts a heap into a sorted range.

For a max heap, the final result is:

    ascending order

Example:

    40 30 20 10 5

After sort_heap():

    5 10 20 30 40

IMPORTANT:
    After sort_heap(), the range is no longer a heap.

Time Complexity:

    O(n log n)

============================================================
*/


int main() {

    /*
    ========================================================
    STEP 1: CREATE VECTOR
    ========================================================

    Initial data:

        10 30 20 5 40
    ========================================================
    */

    vector<int> v = {
        10, 30, 20, 5, 40
    };


    /*
    --------------------------------------------------------
    Print Original Data
    --------------------------------------------------------
    */

    cout << "Original Data: ";

    for (int d : v) {

        cout << d << " ";

    }

    cout << endl;


    /*
    ========================================================
    STEP 2: make_heap()
    ========================================================

    Convert the vector into a MAX HEAP.

    Before:

        10 30 20 5 40

    After:

        40 30 20 5 10

    NOTE:

    The exact internal arrangement can vary as long as
    the heap property is satisfied.

    For a max heap:

        v.front() == largest element
    ========================================================
    */

    make_heap(
        v.begin(),
        v.end()
    );


    cout << "Make Heap: ";

    for (int x : v) {

        cout << x << " ";

    }

    cout << endl;


    /*
    ========================================================
    STEP 3: push_heap()
    ========================================================

    First add a new element normally:

        v.push_back(50);

    At this point:

        The new element is at the end.

    But the vector is no longer guaranteed to be a
    valid heap.

    Therefore:

        push_heap()

    places the new element in its correct heap position.

    Process:

        v.push_back(50);
        push_heap(v.begin(), v.end());

    After push_heap():

        50 will become the largest element and move
        toward the front.
    ========================================================
    */

    v.push_back(50);

    push_heap(
        v.begin(),
        v.end()
    );


    cout << "After Insert: ";

    for (int x : v) {

        cout << x << " ";

    }

    cout << endl;


    /*
    ========================================================
    STEP 4: pop_heap()
    ========================================================

    pop_heap() removes the largest element logically
    from the heap.

    IMPORTANT:

        pop_heap() DOES NOT reduce vector size.

    It moves the largest element:

        FROM:
            v.front()

        TO:
            v.end() - 1

    Example:

        Before:

            50 40 30 5 10 20

        After pop_heap():

            40 20 30 5 10 | 50
                           ↑
                       moved here

    Therefore, we use:

        v.pop_back();

    to actually remove it.
    ========================================================
    */

    pop_heap(
        v.begin(),
        v.end()
    );


    v.pop_back();


    cout << "After Pop: ";

    for (int x : v) {

        cout << x << " ";

    }

    cout << endl;


    /*
    ========================================================
    STEP 5: sort_heap()
    ========================================================

    Now the vector is still a heap.

    sort_heap() converts the heap into sorted order.

    Max heap:

        40 20 30 5 10

    After sort_heap():

        5 10 20 30 40

    IMPORTANT:

        After sort_heap(), the vector is sorted,
        but it is NO LONGER a heap.
    ========================================================
    */

    sort_heap(
        v.begin(),
        v.end()
    );


    cout << "After Sort: ";

    for (int x : v) {

        cout << x << " ";

    }

    cout << endl;


    /*
    ========================================================
                        FINAL SUMMARY
    ========================================================

    make_heap()
        ↓
    Convert vector → heap

        ↓

    push_back()
        ↓
    Add new element

        ↓

    push_heap()
        ↓
    Restore heap property

        ↓

    pop_heap()
        ↓
    Move largest element to the end

        ↓

    pop_back()
        ↓
    Actually remove the largest element

        ↓

    sort_heap()
        ↓
    Convert heap → sorted vector

    --------------------------------------------------------

    IMPORTANT FUNCTIONS:

    make_heap()
        → O(n)

    push_heap()
        → O(log n)

    pop_heap()
        → O(log n)

    sort_heap()
        → O(n log n)

    --------------------------------------------------------

    DEFAULT STL HEAP:

        MAX HEAP

    Therefore:

        v.front()
            ↓
        Largest element

    ========================================================
    */

    return 0;
}