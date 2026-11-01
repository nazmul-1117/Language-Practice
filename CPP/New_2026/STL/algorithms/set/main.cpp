#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/*
============================================================
                STL SET ALGORITHMS
============================================================

This program demonstrates:

1. set_union()
2. set_intersection()
3. set_difference()
4. set_symmetric_difference()
5. includes()
6. merge()
7. inplace_merge()

IMPORTANT:
Most STL set algorithms require the input ranges to be
SORTED according to the same ordering.

Example:

A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

Union                 -> 1 2 3 4 5 6
Intersection          -> 3 4
A - B                 -> 1 2
Symmetric Difference  -> 1 2 5 6
============================================================
*/


// ----------------------------------------------------------
// FUNCTION: printVector()
// Purpose : Print all elements of a vector
// ----------------------------------------------------------
void printVector(const vector<int>& v)
{
    for (int x : v)
    {
        cout << x << " ";
    }

    cout << endl;
}


int main()
{
    /*
    ========================================================
    STEP 1: Create two sorted vectors
    ========================================================
    */

    vector<int> A = {1, 2, 3, 4};
    vector<int> B = {3, 4, 5, 6};


    cout << "A: ";
    printVector(A);

    cout << "B: ";
    printVector(B);

    cout << endl;


    /*
    ========================================================
    1. set_union()

    Meaning:
        A UNION B

    Mathematical notation:
        A ∪ B

    It returns all elements from both ranges.

    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6}

    Result:
        {1, 2, 3, 4, 5, 6}
    ========================================================
    */

    vector<int> result;

    set_union(
        A.begin(),
        A.end(),
        B.begin(),
        B.end(),
        back_inserter(result)
    );

    cout << "Union: ";
    printVector(result);


    /*
    --------------------------------------------------------
    Clear result before using it again.
    --------------------------------------------------------
    */

    result.clear();


    /*
    ========================================================
    2. set_intersection()

    Meaning:
        A INTERSECTION B

    Mathematical notation:
        A ∩ B

    It returns elements common to both ranges.

    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6}

    Result:
        {3, 4}
    ========================================================
    */

    set_intersection(
        A.begin(),
        A.end(),
        B.begin(),
        B.end(),
        back_inserter(result)
    );

    cout << "Intersection: ";
    printVector(result);


    result.clear();


    /*
    ========================================================
    3. set_difference()

    Meaning:
        A - B

    It returns elements that exist in A
    but do NOT exist in B.

    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6}

    Result:
        {1, 2}

    IMPORTANT:
        Direction matters.

        A - B != B - A
    ========================================================
    */

    set_difference(
        A.begin(),
        A.end(),
        B.begin(),
        B.end(),
        back_inserter(result)
    );

    cout << "A - B: ";
    printVector(result);


    result.clear();


    /*
    ========================================================
    4. set_symmetric_difference()

    Meaning:
        Elements present in exactly ONE of the two ranges.

    Mathematical notation:
        A △ B

    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6}

    Common elements:
        3, 4

    Remove common elements.

    Result:
        {1, 2, 5, 6}
    ========================================================
    */

    set_symmetric_difference(
        A.begin(),
        A.end(),
        B.begin(),
        B.end(),
        back_inserter(result)
    );

    cout << "Symmetric Difference: ";
    printVector(result);


    result.clear();


    /*
    ========================================================
    5. includes()

    Purpose:
        Check whether all elements of B exist in A.

    A = {1, 2, 3, 4}
    B = {3, 4}

    Since 3 and 4 exist in A:

        includes() -> true
    ========================================================
    */

    bool isIncluded = includes(
        A.begin(),
        A.end(),
        B.begin(),
        B.end()
    );

    cout << endl;

    cout << "Does A include B? ";

    if (isIncluded)
    {
        cout << "Yes";
    }
    else
    {
        cout << "No";
    }

    cout << endl;


    /*
    ========================================================
    6. merge()

    Purpose:
        Merge TWO SORTED ranges into another sorted range.

    Example:

        A = {1, 3, 5}
        B = {2, 4, 6}

        Result:
            {1, 2, 3, 4, 5, 6}

    IMPORTANT:
        merge() keeps duplicate elements.

    Example:

        A = {1, 2, 3}
        B = {2, 3, 4}

        merge() ->
        {1, 2, 2, 3, 3, 4}
    ========================================================
    */

    vector<int> X = {1, 3, 5};
    vector<int> Y = {2, 4, 6};

    vector<int> merged;

    merge(
        X.begin(),
        X.end(),
        Y.begin(),
        Y.end(),
        back_inserter(merged)
    );

    cout << endl;

    cout << "Merge: ";
    printVector(merged);


    /*
    ========================================================
    7. inplace_merge()

    Purpose:
        Merge TWO CONSECUTIVE SORTED sections
        inside the SAME vector.

    Example:

        {1, 3, 5 | 2, 4, 6}

        First sorted section:
            1 3 5

        Second sorted section:
            2 4 6

        After inplace_merge():

            1 2 3 4 5 6
    ========================================================
    */

    vector<int> V = {
        1, 3, 5,
        2, 4, 6
    };

    /*
        V.begin() + 3 points to the beginning
        of the second sorted section.
    */

    inplace_merge(
        V.begin(),
        V.begin() + 3,
        V.end()
    );

    cout << "Inplace Merge: ";
    printVector(V);


    /*
    ========================================================
                    PROGRAM END
    ========================================================
    */

    return 0;
}