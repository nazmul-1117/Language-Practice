#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;


/*
============================================================
              STL MINIMUM & MAXIMUM ALGORITHMS
============================================================

This program demonstrates:

1. min()
2. max()
3. min_element()
4. max_element()

------------------------------------------------------------
min()
------------------------------------------------------------
Returns the smaller value between two values.

Example:
    min(10, 20) -> 10

Time Complexity:
    O(1)

------------------------------------------------------------
max()
------------------------------------------------------------
Returns the larger value between two values.

Example:
    max(10, 20) -> 20

Time Complexity:
    O(1)

------------------------------------------------------------
min_element()
------------------------------------------------------------
Returns an iterator pointing to the smallest element
in a range.

Example:
    vector = {10, 5, 30, 2, 20}

    min_element()
            ↓
    10  5  30  2  20
             ↑
            min

    Result -> 2

Time Complexity:
    O(n)

------------------------------------------------------------
max_element()
------------------------------------------------------------
Returns an iterator pointing to the largest element
in a range.

Example:
    vector = {10, 5, 30, 2, 20}

    max_element()
            ↓
    10  5  30  2  20
             ↑
            max

    Result -> 30

Time Complexity:
    O(n)

============================================================
*/


int main() {

    /*
    ========================================================
    FUNCTION 1: min()
    ========================================================

    min() compares two values and returns the smaller one.

    Example:

        a = 10
        b = 20

        min(a, b) -> 10
    ========================================================
    */

    int a = 10;
    int b = 20;

    cout << "Min value: "
         << min(a, b)
         << endl;


    /*
    ========================================================
    FUNCTION 2: max()
    ========================================================

    max() compares two values and returns the larger one.

    Example:

        a = 10
        b = 20

        max(a, b) -> 20
    ========================================================
    */

    cout << "Max value: "
         << max(a, b)
         << endl;


    cout << endl;


    /*
    ========================================================
    CREATE VECTOR
    ========================================================

    The vector contains:

        10  5  30  2  20
    ========================================================
    */

    vector<int> v = {
        10, 5, 30, 2, 20
    };


    /*
    ========================================================
    PRINT ORIGINAL DATA
    ========================================================
    */

    cout << "Original Data: ";

    for (int d : v) {

        cout << d << " ";

    }

    cout << endl;


    /*
    ========================================================
    FUNCTION 3: min_element()
    ========================================================

    Syntax:

        min_element(begin, end)

    min_element() does NOT return the minimum value
    directly.

    It returns an ITERATOR pointing to the minimum
    element.

    Example:

        v = {10, 5, 30, 2, 20}

        min_element()
                    ↓
        10  5  30  2  20
                    ↑
                   2

    Therefore:

        *it

    gives us the actual value.

    IMPORTANT:
        The vector does NOT need to be sorted.

    Time Complexity:
        O(n)
    ========================================================
    */

    auto it = min_element(
        v.begin(),
        v.end()
    );


    /*
    --------------------------------------------------------
    Check whether the iterator is valid
    --------------------------------------------------------
    */

    if (it != v.end()) {

        cout << "Min Element: "
             << *it
             << endl;

    }
    else {

        cout << "Vector is empty."
             << endl;

    }


    /*
    ========================================================
    FUNCTION 4: max_element()
    ========================================================

    Syntax:

        max_element(begin, end)

    max_element() returns an ITERATOR pointing to the
    largest element.

    Example:

        v = {10, 5, 30, 2, 20}

        max_element()
                    ↓
        10  5  30  2  20
                ↑
               30

    Therefore:

        *it

    gives us the actual maximum value.

    IMPORTANT:
        The vector does NOT need to be sorted.

    Time Complexity:
        O(n)
    ========================================================
    */

    it = max_element(
        v.begin(),
        v.end()
    );


    /*
    --------------------------------------------------------
    Check whether the iterator is valid
    --------------------------------------------------------
    */

    if (it != v.end()) {

        cout << "Max Element: "
             << *it
             << endl;

    }
    else {

        cout << "Vector is empty."
             << endl;

    }


    /*
    ========================================================
                       FINAL SUMMARY
    ========================================================

    min(a, b)
        -> Minimum between TWO values
        -> Returns VALUE
        -> O(1)

    max(a, b)
        -> Maximum between TWO values
        -> Returns VALUE
        -> O(1)

    min_element(begin, end)
        -> Minimum element from a RANGE
        -> Returns ITERATOR
        -> O(n)

    max_element(begin, end)
        -> Maximum element from a RANGE
        -> Returns ITERATOR
        -> O(n)

    --------------------------------------------------------

    IMPORTANT DIFFERENCE:

        min()
        max()

            ↓
        compare values


        min_element()
        max_element()

            ↓
        search a range
            ↓
        return iterator
            ↓
        use *iterator to get value

    ========================================================
    */

    return 0;
}