#include <iostream>
#include <set>

using namespace std;

int main(int argc, char const *argv[]) {

    int nDot = 50;

    // ============================================================
    // ORDERED SET
    // ============================================================
    // set stores UNIQUE elements.
    // Elements are automatically stored in SORTED ORDER.
    // Default sorting order is ascending.
    //
    // Example:
    // Input:  50 20 40 10 30
    // Output: 10 20 30 40 50
    //
    // Internally, std::set is generally implemented using
    // a Red-Black Tree.
    //
    // Insert / Search / Erase: O(log n)
    // ============================================================


    set<int> numbers;


    // ============================================================
    // 1. INSERT
    // ============================================================

    numbers.insert(50);
    numbers.insert(20);
    numbers.insert(40);
    numbers.insert(10);
    numbers.insert(30);

    // Duplicate values are NOT inserted.
    numbers.insert(30);

    cout << "Set Elements: ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;
    cout << "Set Size: " << numbers.size() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 2. BEGIN()
    // ============================================================
    // Returns an iterator pointing to the FIRST element.
    // Since set is ordered, this is the smallest element.

    cout << "First Element: " << *numbers.begin() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 3. END()
    // ============================================================
    // end() points just AFTER the last element.
    // It should NOT be dereferenced directly.

    auto it = numbers.end();

    // To access the last element:
    it--;

    cout << "Last Element: " << *it << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 4. SIZE()
    // ============================================================

    cout << "Set Size: " << numbers.size() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 5. EMPTY()
    // ============================================================

    if (numbers.empty()) {
        cout << "Set is empty" << endl;
    } else {
        cout << "Set is not empty" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 6. FIND()
    // ============================================================
    // Searches for a specific value.
    // If found, returns iterator to that element.
    // Otherwise returns end().

    it = numbers.find(30);

    if (it != numbers.end()) {
        cout << "Data Found: " << *it << endl;
    } else {
        cout << "Data Not Found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 7. COUNT()
    // ============================================================
    // Returns 1 if the element exists.
    // Returns 0 if the element does not exist.
    //
    // A set cannot contain duplicate values.

    if (numbers.count(40) == 1) {
        cout << "40 is present" << endl;
    } else {
        cout << "40 is not present" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 8. ERASE BY VALUE
    // ============================================================
    // Removes the specified element.

    numbers.erase(20);

    cout << "After erase(20): ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 9. ERASE BY ITERATOR
    // ============================================================
    // First find the element, then erase it using iterator.

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


    // ============================================================
    // 10. INSERT WITH HINT
    // ============================================================
    // hint gives the set a position where the new element
    // may be inserted.

    numbers.insert(numbers.begin(), 25);

    cout << "After insert with hint: ";

    for (int x : numbers) {
        cout << x << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 11. LOWER_BOUND()
    // ============================================================
    // Returns iterator to the FIRST element that is
    // greater than or equal to the given value.

    it = numbers.lower_bound(25);

    if (it != numbers.end()) {
        cout << "Lower Bound of 25: " << *it << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 12. UPPER_BOUND()
    // ============================================================
    // Returns iterator to the FIRST element that is
    // strictly greater than the given value.

    it = numbers.upper_bound(25);

    if (it != numbers.end()) {
        cout << "Upper Bound of 25: " << *it << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 13. EQUAL_RANGE()
    // ============================================================
    // Returns a pair:
    //
    // first  -> lower_bound()
    // second -> upper_bound()
    //
    // For set, because duplicate values are not allowed,
    // the range will contain at most one element.

    auto range = numbers.equal_range(25);

    if (range.first != numbers.end()) {
        cout << "Equal Range First: "
             << *range.first << endl;
    }

    if (range.second != numbers.end()) {
        cout << "Equal Range Second: "
             << *range.second << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 14. ITERATE USING ITERATOR
    // ============================================================

    cout << "Using Iterator: ";

    for (set<int>::iterator i = numbers.begin();
         i != numbers.end();
         i++) {

        cout << *i << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 15. REVERSE ITERATION
    // ============================================================
    // rbegin() -> last element
    // rend()   -> before first element

    cout << "Reverse Order: ";

    for (auto i = numbers.rbegin();
         i != numbers.rend();
         i++) {

        cout << *i << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 16. CRBEGIN() AND CREND()
    // ============================================================
    // Constant reverse iterators.
    // Used when elements should be accessed in reverse order
    // without modifying them.

    cout << "Constant Reverse Order: ";

    for (auto i = numbers.crbegin();
         i != numbers.crend();
         i++) {

        cout << *i << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 17. CLEAR()
    // ============================================================
    // Removes ALL elements from the set.

    numbers.clear();

    cout << "After clear(), Set Size: "
         << numbers.size() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 18. SWAP()
    // ============================================================
    // Exchanges the contents of two sets.

    set<int> set1 = {10, 20, 30};
    set<int> set2 = {100, 200, 300};

    cout << "Before Swap:" << endl;

    cout << "Set 1: ";
    for (int x : set1) {
        cout << x << " ";
    }

    cout << endl;

    cout << "Set 2: ";
    for (int x : set2) {
        cout << x << " ";
    }

    cout << endl;

    set1.swap(set2);

    cout << endl << "After Swap:" << endl;

    cout << "Set 1: ";
    for (int x : set1) {
        cout << x << " ";
    }

    cout << endl;

    cout << "Set 2: ";
    for (int x : set2) {
        cout << x << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 19. MAXIMUM / MINIMUM ELEMENT
    // ============================================================
    // Because set is sorted:
    //
    // *begin()       -> Minimum
    // *rbegin()      -> Maximum

    cout << "Minimum: " << *set1.begin() << endl;
    cout << "Maximum: " << *set1.rbegin() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // 20. SET WITH DESCENDING ORDER
    // ============================================================
    // greater<int> changes the sorting order from ascending
    // to descending.

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


    // ============================================================
    // 21. SET OF STRINGS
    // ============================================================
    // Strings are also automatically sorted.

    set<string> countries;

    countries.insert("Bangladesh");
    countries.insert("India");
    countries.insert("Pakistan");
    countries.insert("Nepal");

    cout << "Countries: ";

    for (string country : countries) {
        cout << country << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    return 0;
}