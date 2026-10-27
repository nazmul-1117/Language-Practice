#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

void multiply10X(int x) {
    cout << x / 10 << " ";
}

bool evenFilter(int x) {
    return x % 2 == 0;
}

void printArray(const vector<int>& v) {
    int nDot = 60;

    for (int d : v) {
        cout << d << " ";
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;
}

void printIteratorArray(
    vector<int>::const_iterator st,
    vector<int>::const_iterator en
) {
    int nDot = 60;

    while (st != en) {
        cout << *st << " ";
        ++st;
    }

    cout << endl;
    cout << string(nDot, '-') << endl << endl;
}

int main() {

    cout << "C++ Standard: " << __cplusplus << endl << endl;

    int nDot = 60;

    vector<int> v = {
        31, 20, 11, 11, 70, 60
    };

    // Original Data
    cout << "Original Data:\t";
    printArray(v);


    // Function 1: for_each
    cout << "After For Each:\t";

    for_each(v.begin(), v.end(), multiply10X);

    cout << endl;
    cout << string(nDot, '-') << endl << endl;


    // Function 2: find
    cout << "find:\t\t";

    int target = 70;

    auto it = find(v.begin(), v.end(), target);

    if (it != v.end()) {
        cout << *it << endl;
    } else {
        cout << "Not Found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // Function 2: find_if
    cout << "find_if:\t";

    it = find_if(v.begin(), v.end(), evenFilter);

    if (it != v.end()) {
        cout << *it << endl;
    } else {
        cout << "Not Found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // Function 3: count
    target = 10;

    int ans = count(v.begin(), v.end(), target);

    cout << "Count:\t\t" << ans << endl;
    cout << string(nDot, '-') << endl << endl;


    // Function 4: sort
    cout << "After sorting:\t";

    sort(v.begin(), v.end());

    printArray(v);


    // Function 5: reverse
    cout << "After reverse:\t";

    reverse(v.begin(), v.end());

    printArray(v);


    // Function 6: right rotate by 2
    cout << "After right rotate of 2:\t";

    rotate(v.begin(), v.end() - 2, v.end());

    printArray(v);


    // Function 7: unique
    cout << "After unique:\t";

    auto uniqueEnd = unique(v.begin(), v.end());

    // Logical unique range
    printIteratorArray(v.begin(), uniqueEnd);

    // Actually remove duplicate elements
    v.erase(uniqueEnd, v.end());

    cout << "After erase:\t";
    printArray(v);

    // function 8 - partitioning
    cout << "After partition:\t";

    auto partitionItr = partition(v.begin(), v.end(), evenFilter);
    
    // Logical partition range
    printIteratorArray(v.begin(), v.end());



    return 0;
}