#include <iostream>
#include <set>

using namespace std;

int main(int argc, char const *argv[]) {

    int nDot = 50;

    // ============================================================
    // ORDERED SET
    // ============================================================
    // set stores UNIQUE elements.
    // Elements are automatically stored in SORTED order.
    // Default order is ascending.
    // Duplicate elements are automatically ignored.
    // Search / insertion / deletion: O(log n)
    // ============================================================

    set<int> numbers;

    // ------------------------------------------------------------
    // INSERT ELEMENTS
    // ------------------------------------------------------------

    numbers.insert(50);
    numbers.insert(20);
    numbers.insert(30);
    numbers.insert(40);
    numbers.insert(10);

    // Duplicate value will NOT be inserted
    numbers.insert(30);

    cout << "Ordered Set Size: " << numbers.size() << endl;

    // ------------------------------------------------------------
    // DISPLAY ALL ELEMENTS
    // ------------------------------------------------------------
    // Elements are automatically displayed in sorted order.

    cout << "Elements: ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // BEGIN()
    // ------------------------------------------------------------
    // begin() returns an iterator to the first element.
    // In an ordered set, this is the smallest element.

    cout << "First Element: " << *numbers.begin() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // END()
    // ------------------------------------------------------------
    // end() points just after the last element.
    // To access the last element, decrement the iterator.

    set<int>::iterator it;

    it = numbers.end();
    it--;

    cout << "Last Element: " << *it << endl;

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // FIND ELEMENT
    // ------------------------------------------------------------
    // find() searches for a specific element.
    // If found, it returns an iterator to that element.
    // Otherwise, it returns end().

    it = numbers.find(30);

    if (it != numbers.end()) {
        cout << "Data found: " << *it << endl;
    } else {
        cout << "Data not found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // COUNT
    // ------------------------------------------------------------
    // Since set does not allow duplicate values,
    // count() can return either 0 or 1.

    if (numbers.count(20) == 1) {
        cout << "20 is present in the set" << endl;
    } else {
        cout << "20 is not present in the set" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // ERASE ELEMENT BY VALUE
    // ------------------------------------------------------------
    // Removes the specified element from the set.

    numbers.erase(40);

    cout << "After erasing 40: ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // ERASE ELEMENT BY ITERATOR
    // ------------------------------------------------------------

    it = numbers.find(30);

    if (it != numbers.end()) {
        numbers.erase(it);
    }

    cout << "After erasing 30 using iterator: ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // LOWER_BOUND
    // ------------------------------------------------------------
    // lower_bound(x) returns the first element
    // which is greater than or equal to x.

    it = numbers.lower_bound(25);

    if (it != numbers.end()) {
        cout << "Lower bound of 25: " << *it << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // UPPER_BOUND
    // ------------------------------------------------------------
    // upper_bound(x) returns the first element
    // which is strictly greater than x.

    it = numbers.upper_bound(25);

    if (it != numbers.end()) {
        cout << "Upper bound of 25: " << *it << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // EQUAL_RANGE
    // ------------------------------------------------------------
    // equal_range(x) returns a pair of iterators:
    //
    // first  = lower_bound(x)
    // second = upper_bound(x)

    auto range = numbers.equal_range(25);

    if (range.first != numbers.end()) {
        cout << "Equal range first: "
             << *range.first << endl;
    }

    if (range.second != numbers.end()) {
        cout << "Equal range second: "
             << *range.second << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // REVERSE ITERATION
    // ------------------------------------------------------------
    // rbegin() points to the largest element.
    // rend() points before the smallest element.

    cout << "Reverse Order: ";

    for (auto i = numbers.rbegin();
         i != numbers.rend();
         i++) {

        cout << *i << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // INSERT MORE ELEMENTS
    // ------------------------------------------------------------

    numbers.insert(60);
    numbers.insert(70);

    cout << "After inserting 60 and 70: ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // SIZE
    // ------------------------------------------------------------

    cout << "Current Set Size: " << numbers.size() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // EMPTY
    // ------------------------------------------------------------

    if (numbers.empty()) {
        cout << "Set is empty" << endl;
    } else {
        cout << "Set is not empty" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // CLEAR
    // ------------------------------------------------------------
    // Removes all elements from the set.

    numbers.clear();

    if (numbers.empty()) {
        cout << "Set is empty after clear()" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // SWAP
    // ------------------------------------------------------------
    // Exchanges the contents of two sets.

    set<int> set1 = {10, 20, 30};
    set<int> set2 = {100, 200, 300};

    set1.swap(set2);

    cout << "Set 1 after swap: ";

    for (int x : set1) {
        cout << x << " ";
    }

    cout << endl;

    cout << "Set 2 after swap: ";

    for (int x : set2) {
        cout << x << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // ------------------------------------------------------------
    // DESCENDING ORDERED SET
    // ------------------------------------------------------------
    // greater<int> stores elements in descending order.

    set<int, greater<int>> descendingSet;

    descendingSet.insert(10);
    descendingSet.insert(50);
    descendingSet.insert(20);
    descendingSet.insert(40);
    descendingSet.insert(30);

    cout << "Descending Set: ";

    for (int x : descendingSet) {
        cout << x << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    return 0;
}