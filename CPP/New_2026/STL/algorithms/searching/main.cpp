#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;


/*
============================================================
                STL SEARCHING ALGORITHMS
============================================================

This program demonstrates:

1. Linear Search       -> find()
2. Binary Search       -> binary_search()
3. Lower Bound         -> lower_bound()
4. Upper Bound         -> upper_bound()
5. Equal Range         -> equal_range()

IMPORTANT:
- find() can work on an unsorted vector.
- binary_search(), lower_bound(), upper_bound(),
  and equal_range() require the vector to be sorted.

============================================================
*/


int main() {

    /*
    --------------------------------------------------------
    STEP 1: Create the vector
    --------------------------------------------------------

    The vector must be sorted before using:

        binary_search()
        lower_bound()
        upper_bound()
        equal_range()

    --------------------------------------------------------
    */

    vector<int> v = {
        11, 12, 13, 14, 15
    };


    // Value that we want to search for
    int target = 13;


    /*
    ========================================================
    FUNCTION 1: LINEAR SEARCH
    ========================================================

    Function:
        find(begin, end, value)

    How it works:
        - Starts from the first element.
        - Checks each element one by one.
        - Stops when target is found.
        - Returns an iterator to the found element.
        - If target is not found, returns end().

    Time Complexity:
        O(n)

    Important:
        The vector does NOT need to be sorted.
    ========================================================
    */

    auto it = find(v.begin(), v.end(), target);


    if (it != v.end()) {

        cout << "Linear Search:\t";
        cout << "Data found -> " << *it << endl;

    }
    else {

        cout << "Linear Search:\t";
        cout << "Data not found" << endl;

    }


    cout << endl;


    /*
    ========================================================
    FUNCTION 2: BINARY SEARCH
    ========================================================

    Function:
        binary_search(begin, end, value)

    How it works:
        - Divides the search range repeatedly.
        - Much faster than linear search for large data.
        - Returns only true or false.

    Return:
        true  -> data found
        false -> data not found

    Time Complexity:
        O(log n)

    IMPORTANT:
        The vector MUST be sorted.
    ========================================================
    */

    bool isFound = binary_search(
        v.begin(),
        v.end(),
        target
    );


    cout << "Binary Search:\t";

    if (isFound) {

        cout << target
             << " data found -> "
             << boolalpha
             << isFound
             << endl;

    }
    else {

        cout << target
             << " data is not found -> "
             << boolalpha
             << isFound
             << endl;

    }


    cout << endl;


    /*
    ========================================================
    FUNCTION 3: LOWER BOUND
    ========================================================

    Function:
        lower_bound(begin, end, value)

    Meaning:
        Returns an iterator pointing to the FIRST element
        that is >= target.

    Example:

        Vector:
        10  20  20  30  40
                 ^
                 |
            lower_bound(20)

        Result -> first 20

    If target does not exist:
        It returns the position where target could be
        inserted while keeping the vector sorted.

    Time Complexity:
        O(log n)
    ========================================================
    */

    it = lower_bound(
        v.begin(),
        v.end(),
        target
    );


    cout << "Lower Bound:\t";

    if (it != v.end()) {

        cout << "First element >= "
             << target
             << " -> "
             << *it
             << endl;

    }
    else {

        cout << "No element >= "
             << target
             << endl;

    }


    cout << endl;


    /*
    ========================================================
    FUNCTION 4: UPPER BOUND
    ========================================================

    Function:
        upper_bound(begin, end, value)

    Meaning:
        Returns an iterator pointing to the FIRST element
        that is > target.

    Example:

        Vector:
        10  20  20  30  40
                    ^
                    |
              upper_bound(20)

        Result -> 30

    Time Complexity:
        O(log n)
    ========================================================
    */

    it = upper_bound(
        v.begin(),
        v.end(),
        target
    );


    cout << "Upper Bound:\t";

    if (it != v.end()) {

        cout << "First element > "
             << target
             << " -> "
             << *it
             << endl;

    }
    else {

        cout << "No element > "
             << target
             << endl;

    }


    cout << endl;


    /*
    ========================================================
    FUNCTION 5: EQUAL RANGE
    ========================================================

    Function:
        equal_range(begin, end, value)

    This combines:

        lower_bound()
        upper_bound()

    It returns a pair of iterators:

        range.first
            -> lower_bound
            -> first element >= target

        range.second
            -> upper_bound
            -> first element > target

    Example:

        Vector:
        10  20  20  20  30  40
            |-----------|
            first      second

        equal_range(20)

        first  -> first 20
        second -> 30

    Time Complexity:
        O(log n)
    ========================================================
    */

    auto range = equal_range(
        v.begin(),
        v.end(),
        target
    );


    /*
    --------------------------------------------------------
    Print Equal Range - First Iterator
    --------------------------------------------------------
    */

    cout << "Equal Range:\n";

    if (range.first != v.end()) {

        cout << "First bound:\t"
             << *range.first
             << endl;

    }
    else {

        cout << "First bound:\tend()" << endl;

    }


    /*
    --------------------------------------------------------
    Print Equal Range - Second Iterator
    --------------------------------------------------------
    */

    if (range.second != v.end()) {

        cout << "Second bound:\t"
             << *range.second
             << endl;

    }
    else {

        cout << "Second bound:\tend()" << endl;

    }


    /*
    ========================================================
    FINAL SUMMARY
    ========================================================

    find()
        -> Searches one by one
        -> O(n)
        -> Sorted vector NOT required

    binary_search()
        -> Returns true/false
        -> O(log n)
        -> Sorted vector REQUIRED

    lower_bound()
        -> First element >= target
        -> O(log n)
        -> Sorted vector REQUIRED

    upper_bound()
        -> First element > target
        -> O(log n)
        -> Sorted vector REQUIRED

    equal_range()
        -> lower_bound + upper_bound
        -> Returns pair of iterators
        -> O(log n)
        -> Sorted vector REQUIRED

    ========================================================
    */

    return 0;
}