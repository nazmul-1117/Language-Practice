#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>

using namespace std;

/*
============================================================
                    C++ STL — ITERATORS
============================================================

An iterator is like a generalized pointer.

It is used to:
    - Access elements
    - Traverse containers
    - Modify elements
    - Work with STL algorithms

Important iterator concepts covered:

1. begin()
2. end()
3. Dereferencing (*it)
4. ++it
5. --it
6. Iterator arithmetic
7. Reverse iterator
8. const_iterator
9. advance()
10. next()
11. prev()
12. distance()
13. back_inserter()
14. Iterator returned by algorithms
15. Iterator categories

============================================================
*/


// ----------------------------------------------------------
// FUNCTION: printUsingIterator()
// Purpose : Print vector elements using an iterator
// ----------------------------------------------------------

void printUsingIterator(const vector<int>& v)
{
    cout << "Data: ";

    for (auto it = v.begin(); it != v.end(); ++it)
    {
        cout << *it << " ";
    }

    cout << endl;
}


int main()
{
    /*
    ========================================================
    STEP 1: Create a vector
    ========================================================
    */

    vector<int> v = {10, 20, 30, 40, 50};

    cout << "Original Data: ";
    printUsingIterator(v);


    /*
    ========================================================
    STEP 2: begin()

    begin() returns an iterator pointing to
    the FIRST element.

    Vector:

        [10] [20] [30] [40] [50]
         ↑
       begin()

    ========================================================
    */

    auto it = v.begin();

    cout << "\n1. begin(): ";
    cout << *it << endl;


    /*
    ========================================================
    STEP 3: end()

    end() points ONE POSITION AFTER the last element.

        [10] [20] [30] [40] [50] [END]
                                  ↑
                                 end()

    IMPORTANT:

        *v.end()     // ❌ Invalid

    ========================================================
    */

    auto endIt = v.end();

    cout << "2. Last Element: ";
    cout << *(endIt - 1) << endl;


    /*
    ========================================================
    STEP 4: Dereferencing an Iterator

    The * operator accesses the element
    pointed to by the iterator.

        *it

    ========================================================
    */

    it = v.begin();

    cout << "3. Dereference: ";
    cout << *it << endl;


    /*
    ========================================================
    STEP 5: ++it

    Moves the iterator to the NEXT element.

        Before:

        [10] [20] [30] [40] [50]
         ↑
         it

        After ++it:

        [10] [20] [30] [40] [50]
               ↑
               it

    ========================================================
    */

    ++it;

    cout << "4. After ++it: ";
    cout << *it << endl;


    /*
    ========================================================
    STEP 6: --it

    Moves the iterator to the PREVIOUS element.

    ========================================================
    */

    --it;

    cout << "5. After --it: ";
    cout << *it << endl;


    /*
    ========================================================
    STEP 7: Iterator Traversal

    Standard iterator loop:

        begin()
           ↓
        check end()
           ↓
        use *it
           ↓
        ++it
           ↓
        repeat

    ========================================================
    */

    cout << "\n6. Forward Traversal: ";

    for (auto iter = v.begin();
         iter != v.end();
         ++iter)
    {
        cout << *iter << " ";
    }

    cout << endl;


    /*
    ========================================================
    STEP 8: Iterator Arithmetic

    Vector provides Random Access Iterators.

    Therefore we can use:

        it + n
        it - n
        it += n
        it -= n
        it[n]

    ========================================================
    */

    it = v.begin();

    cout << "7. it + 2: ";
    cout << *(it + 2) << endl;

    cout << "8. it + 4: ";
    cout << *(it + 4) << endl;

    cout << "9. it[3]: ";
    cout << it[3] << endl;


    /*
    ========================================================
    STEP 9: Iterator Difference

    Random-access iterators can be subtracted.

        v.end() - v.begin()

    gives the number of elements.

    ========================================================
    */

    auto first = v.begin();
    auto last = v.end();

    cout << "10. Iterator Distance: ";
    cout << last - first << endl;


    /*
    ========================================================
    STEP 10: Reverse Iterator

    rbegin() -> points to LAST element
    rend()   -> one position before FIRST element

    Normal:

        10 → 20 → 30 → 40 → 50

    Reverse:

        50 → 40 → 30 → 20 → 10

    ========================================================
    */

    cout << "\n11. Reverse Traversal: ";

    for (auto iter = v.rbegin();
         iter != v.rend();
         ++iter)
    {
        cout << *iter << " ";
    }

    cout << endl;


    /*
    ========================================================
    STEP 11: const_iterator

    const_iterator allows us to READ elements,
    but we cannot MODIFY elements through it.

    ========================================================
    */

    vector<int>::const_iterator cit = v.cbegin();

    cout << "12. const_iterator: ";
    cout << *cit << endl;

    // *cit = 100;    // ❌ ERROR


    /*
    ========================================================
    STEP 12: cbegin() and cend()

    cbegin() -> const beginning
    cend()   -> const ending

    They provide read-only access.

    ========================================================
    */

    cout << "13. cbegin/cend: ";

    for (auto iter = v.cbegin();
         iter != v.cend();
         ++iter)
    {
        cout << *iter << " ";
    }

    cout << endl;


    /*
    ========================================================
    STEP 13: crbegin() and crend()

    These provide CONST reverse iterators.

    ========================================================
    */

    cout << "14. Const Reverse Traversal: ";

    for (auto iter = v.crbegin();
         iter != v.crend();
         ++iter)
    {
        cout << *iter << " ";
    }

    cout << endl;


    /*
    ========================================================
    STEP 14: advance()

    advance() moves an iterator by a given number
    of positions.

    Syntax:

        advance(iterator, number);

    Example:

        begin()
           ↓
        10 20 30 40 50
           ↑
          +3

        Result = 40

    IMPORTANT:
        advance() changes the original iterator.

    ========================================================
    */

    it = v.begin();

    advance(it, 3);

    cout << "\n15. advance(it, 3): ";
    cout << *it << endl;


    /*
    ========================================================
    STEP 15: next()

    next() returns a NEW iterator.

    It does NOT change the original iterator.

    ========================================================
    */

    it = v.begin();

    auto nextIt = next(it, 2);

    cout << "16. next(it, 2): ";
    cout << *nextIt << endl;

    cout << "    Original it: ";
    cout << *it << endl;


    /*
    ========================================================
    STEP 16: prev()

    prev() returns an iterator moved backward.

    Example:

        end()
         ↓
        10 20 30 40 50
                 ↑
              prev(end())

    ========================================================
    */

    auto previousIt = prev(v.end());

    cout << "17. prev(v.end()): ";
    cout << *previousIt << endl;


    /*
    ========================================================
    STEP 17: distance()

    distance() calculates the number of positions
    between two iterators.

    ========================================================
    */

    auto start = v.begin();
    auto finish = v.end();

    cout << "18. distance(): ";
    cout << distance(start, finish) << endl;


    /*
    ========================================================
    STEP 18: Iterator with find()

    Many STL algorithms return an iterator.

    find() searches for a value.

    ========================================================
    */

    int target = 30;

    auto findIt = find(
        v.begin(),
        v.end(),
        target
    );

    cout << "\n19. find(30): ";

    if (findIt != v.end())
    {
        cout << "Found -> " << *findIt << endl;
    }
    else
    {
        cout << "Not Found" << endl;
    }


    /*
    ========================================================
    STEP 19: Iterator with min_element()

    min_element() returns an iterator
    pointing to the smallest element.

    ========================================================
    */

    auto minIt = min_element(
        v.begin(),
        v.end()
    );

    cout << "20. min_element(): ";
    cout << *minIt << endl;


    /*
    ========================================================
    STEP 20: Iterator with max_element()

    max_element() returns an iterator
    pointing to the largest element.

    ========================================================
    */

    auto maxIt = max_element(
        v.begin(),
        v.end()
    );

    cout << "21. max_element(): ";
    cout << *maxIt << endl;


    /*
    ========================================================
    STEP 21: back_inserter()

    back_inserter() creates an output iterator
    that inserts elements using push_back().

    ========================================================
    */

    vector<int> source = {1, 2, 3, 4, 5};

    vector<int> result;

    copy(
        source.begin(),
        source.end(),
        back_inserter(result)
    );

    cout << "\n22. back_inserter(): ";

    for (int x : result)
    {
        cout << x << " ";
    }

    cout << endl;


    /*
    ========================================================
    STEP 22: Iterator with list

    list provides BIDIRECTIONAL iterators.

    Therefore:

        ++it  -> works
        --it  -> works

    But:

        it + 3 -> ❌ does NOT work

    Instead use advance().

    ========================================================
    */

    list<int> numbers = {
        100, 200, 300, 400, 500
    };

    auto listIt = numbers.begin();

    advance(listIt, 3);

    cout << "\n23. List + advance(): ";
    cout << *listIt << endl;


    /*
    ========================================================
    STEP 23: Iterator Categories

    Traditional STL iterator categories:

        Input Iterator
              ↓
        Forward Iterator
              ↓
        Bidirectional Iterator
              ↓
        Random Access Iterator
              ↓
        Contiguous Iterator

    Output Iterator is a separate write-oriented category.

    ========================================================

    Examples:

        vector
            → Random Access / Contiguous

        array
            → Random Access / Contiguous

        deque
            → Random Access

        list
            → Bidirectional

        forward_list
            → Forward

        set
            → Bidirectional

        map
            → Bidirectional

    ========================================================
    */


    /*
    ========================================================
                    PROGRAM END
    ========================================================
    */

    return 0;
}