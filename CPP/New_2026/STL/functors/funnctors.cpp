#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace std;

/*
============================================================
                    C++ STL — FUNCTORS
============================================================

A FUNCTOR is an object that behaves like a function.

A class/struct becomes a functor when it defines:

        operator()

Example:

struct Add
{
    int operator()(int a, int b) const
    {
        return a + b;
    }
};

Add add;

cout << add(10, 20);

Here:

    add(10, 20)

is internally similar to:

    add.operator()(10, 20);

============================================================
*/


/*
============================================================
FUNCTION 1: Basic Functor
============================================================

This functor checks whether a number is even.

operator() returns:

    true  -> even
    false -> odd

============================================================
*/

struct IsEven
{
    bool operator()(int value) const
    {
        return value % 2 == 0;
    }
};


/*
============================================================
FUNCTION 2: Add Functor
============================================================

This functor accepts two parameters and returns their sum.

============================================================
*/

struct Add
{
    int operator()(int a, int b) const
    {
        return a + b;
    }
};


/*
============================================================
FUNCTION 3: Square Functor
============================================================

This functor calculates the square of a number.

============================================================
*/

struct Square
{
    int operator()(int x) const
    {
        return x * x;
    }
};


/*
============================================================
FUNCTION 4: GreaterThan Functor
============================================================

This is a STATEFUL FUNCTOR.

It stores a value called 'limit'.

Example:

    GreaterThan check(50);

means:

    limit = 50

Then:

    check(70)

checks:

    70 > 50

============================================================
*/

struct GreaterThan
{
    int limit;

    // Constructor
    GreaterThan(int value)
        : limit(value)
    {
    }

    bool operator()(int value) const
    {
        return value > limit;
    }
};


/*
============================================================
FUNCTION 5: Descending Comparator
============================================================

This functor is used as a custom comparator for sort().

    a > b

means larger values should come first.

Therefore:

    9 7 5 3 1

============================================================
*/

struct Descending
{
    bool operator()(int a, int b) const
    {
        return a > b;
    }
};


/*
============================================================
FUNCTION 6: Print Functor
============================================================

Used with for_each().

It prints every element.

============================================================
*/

struct Print
{
    void operator()(int x) const
    {
        cout << x << " ";
    }
};


/*
============================================================
FUNCTION 7: Double Functor
============================================================

Used with transform().

It doubles each element.

Example:

    1 2 3 4

becomes:

    2 4 6 8

============================================================
*/

struct Double
{
    int operator()(int x) const
    {
        return x * 2;
    }
};


/*
============================================================
FUNCTION 8: Multiply Functor
============================================================

Used with accumulate().

Instead of adding values:

    1 + 2 + 3 + 4

we can multiply them:

    1 * 2 * 3 * 4

============================================================
*/

struct Multiply
{
    int operator()(int a, int b) const
    {
        return a * b;
    }
};


/*
============================================================
FUNCTION 9: Student Structure
============================================================

Used to demonstrate sorting user-defined objects.

============================================================
*/

struct Student
{
    string name;
    int marks;
};


/*
============================================================
FUNCTION 10: Student Comparator Functor
============================================================

Sort students according to marks.

Higher marks come first.

============================================================
*/

struct CompareStudent
{
    bool operator()(
        const Student& a,
        const Student& b
    ) const
    {
        if (a.marks == b.marks)
            return a.name > b.name;
        return a.marks > b.marks;
    }
};


/*
============================================================
                        MAIN FUNCTION
============================================================
*/

int main()
{
    /*
    ========================================================
    SECTION 1: BASIC FUNCTOR
    ========================================================
    */

    cout << "================ BASIC FUNCTOR ================\n";

    IsEven check;

    cout << "Is 10 even? "
         << boolalpha
         << check(10)
         << endl;

    cout << "Is 15 even? "
         << check(15)
         << endl;


    /*
    --------------------------------------------------------
    IMPORTANT:

    check is an OBJECT.

    But:

        check(10)

    works because the structure has:

        operator()

    Internally:

        check(10)

    behaves like:

        check.operator()(10)
    --------------------------------------------------------
    */


    /*
    ========================================================
    SECTION 2: FUNCTOR WITH MULTIPLE PARAMETERS
    ========================================================
    */

    cout << "\n================ ADD FUNCTOR ================\n";

    Add add;

    cout << "10 + 20 = "
         << add(10, 20)
         << endl;


    /*
    ========================================================
    SECTION 3: SQUARE FUNCTOR
    ========================================================
    */

    cout << "\n================ SQUARE FUNCTOR ================\n";

    Square square;

    cout << "Square of 5 = "
         << square(5)
         << endl;


    /*
    ========================================================
    SECTION 4: FUNCTOR WITH STL count_if()
    ========================================================

    count_if() counts how many elements satisfy a condition.

    Here:

        IsEven{}

    is passed as the condition.

    ========================================================
    */

    cout << "\n================ count_if() ================\n";

    vector<int> numbers = {
        10, 15, 20, 25, 30, 35
    };

    int evenCount = count_if(
        numbers.begin(),
        numbers.end(),
        IsEven{}
    );

    cout << "Number of even values: "
         << evenCount
         << endl;


    /*
    ========================================================
    SECTION 5: STATEFUL FUNCTOR
    ========================================================

    GreaterThan(20)

    creates a functor whose internal state is:

        limit = 20

    ========================================================
    */

    cout << "\n================ STATEFUL FUNCTOR ================\n";

    int greaterCount = count_if(
        numbers.begin(),
        numbers.end(),
        GreaterThan(20)
    );

    cout << "Numbers greater than 20: "
         << greaterCount
         << endl;


    /*
    ========================================================
    SECTION 6: find_if() WITH FUNCTOR
    ========================================================

    find_if() returns an iterator to the first element
    satisfying the condition.

    ========================================================
    */

    cout << "\n================ find_if() ================\n";

    auto it = find_if(
        numbers.begin(),
        numbers.end(),
        IsEven{}
    );

    if (it != numbers.end())
    {
        cout << "First even number: "
             << *it
             << endl;
    }
    else
    {
        cout << "No even number found."
             << endl;
    }


    /*
    ========================================================
    SECTION 7: for_each() WITH FUNCTOR
    ========================================================

    for_each() applies the functor to every element.

    ========================================================
    */

    cout << "\n================ for_each() ================\n";

    cout << "Numbers: ";

    for_each(
        numbers.begin(),
        numbers.end(),
        Print{}
    );

    cout << endl;


    /*
    ========================================================
    SECTION 8: transform() WITH FUNCTOR
    ========================================================

    transform() applies the functor to each element.

    Original:

        10 15 20 25 30 35

    After Double:

        20 30 40 50 60 70

    ========================================================
    */

    cout << "\n================ transform() ================\n";

    vector<int> doubled = numbers;

    transform(
        doubled.begin(),
        doubled.end(),
        doubled.begin(),
        Double{}
    );

    cout << "Doubled values: ";

    for (int x : doubled)
    {
        cout << x << " ";
    }

    cout << endl;


    /*
    ========================================================
    SECTION 9: accumulate() WITH FUNCTOR
    ========================================================

    Normal accumulate:

        1 + 2 + 3 + 4

    Custom Multiply functor:

        1 * 2 * 3 * 4

    ========================================================
    */

    cout << "\n================ accumulate() ================\n";

    vector<int> values = {
        1, 2, 3, 4
    };

    int product = accumulate(
        values.begin(),
        values.end(),
        1,
        Multiply{}
    );

    cout << "Product = "
         << product
         << endl;


    /*
    ========================================================
    SECTION 10: CUSTOM COMPARATOR
    ========================================================

    sort() normally sorts in ascending order.

    We can provide our own functor to change the rule.

    ========================================================
    */

    cout << "\n================ CUSTOM COMPARATOR ================\n";

    vector<int> data = {
        5, 2, 9, 1, 7, 3
    };

    sort(
        data.begin(),
        data.end(),
        Descending{}
    );

    cout << "Descending order: ";

    for (int x : data)
    {
        cout << x << " ";
    }

    cout << endl;


    /*
    ========================================================
    SECTION 11: BUILT-IN FUNCTOR greater<int>()
    ========================================================

    C++ already provides many functors inside:

        <functional>

    greater<int>() behaves approximately like:

        a > b

    ========================================================
    */

    cout << "\n================ BUILT-IN FUNCTOR ================\n";

    vector<int> builtIn = {
        10, 30, 20, 50, 40
    };

    sort(
        builtIn.begin(),
        builtIn.end(),
        greater<int>()
    );

    cout << "Using greater<int>(): ";

    for (int x : builtIn)
    {
        cout << x << " ";
    }

    cout << endl;


    /*
    ========================================================
    SECTION 12: BUILT-IN ARITHMETIC FUNCTOR
    ========================================================
    */

    cout << "\n================ ARITHMETIC FUNCTOR ================\n";

    plus<int> addition;

    cout << "10 + 20 = "
         << addition(10, 20)
         << endl;


    multiplies<int> multiplication;

    cout << "10 * 20 = "
         << multiplication(10, 20)
         << endl;


    /*
    ========================================================
    SECTION 13: LOGICAL FUNCTOR
    ========================================================
    */

    cout << "\n================ LOGICAL FUNCTOR ================\n";

    logical_and<bool> logicalAnd;

    cout << "true && false = "
         << logicalAnd(true, false)
         << endl;


    logical_or<bool> logicalOr;

    cout << "true || false = "
         << logicalOr(true, false)
         << endl;


    logical_not<bool> logicalNot;

    cout << "!true = "
         << logicalNot(true)
         << endl;


    /*
    ========================================================
    SECTION 14: USER-DEFINED OBJECTS
    ========================================================

    Sort students according to their marks.

    ========================================================
    */

    cout << "\n================ STUDENT FUNCTOR ================\n";

    vector<Student> students =
    {
        {"Rahim", 75},
        {"Karim", 90},
        {"Nazmul", 85},
        {"Hasan", 65},
        {"Nayeem", 85}
    };

    sort(
        students.begin(),
        students.end(),
        CompareStudent{}
    );

    cout << "Students by marks (descending):\n";

    for (const Student& student : students)
    {
        cout << student.name
             << " -> "
             << student.marks
             << endl;
    }


    /*
    ========================================================
    SECTION 15: std::function WITH FUNCTOR
    ========================================================

    std::function can store a callable object.

    It can store:

        Function
        Lambda
        Functor
        Function pointer

    ========================================================
    */

    cout << "\n================ std::function ================\n";

    function<int(int, int)> operation;

    operation = Add{};

    cout << "Using std::function: "
         << operation(100, 200)
         << endl;


    /*
    ========================================================
                        PROGRAM END
    ========================================================
    */

    return 0;
}