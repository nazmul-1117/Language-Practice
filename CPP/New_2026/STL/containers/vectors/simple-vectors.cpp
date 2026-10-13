#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]) {

    // ============================================================
    // Basic Configuration
    // ============================================================

    // Number of '-' characters used as a separator
    int nDot = 50;


    // ============================================================
    // Vector Initialization
    // ============================================================

    // Initialize an empty vector
    vector<int> v;

    // vector<int> v(5);       // Create a vector with size 5
    // vector<int> v(5, -1);  // Create a vector with size 5
                                // and initialize every element with -1


    // ============================================================
    // Adding and Removing Data from Vector
    // ============================================================

    // Add elements to the end of the vector
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    v.push_back(40);
    v.push_back(50);
    v.push_back(60);

    // Remove the last element from the vector
    // 60 will be removed
    v.pop_back();


    // ============================================================
    // Vector Size
    // ============================================================

    // Display the number of elements currently stored in the vector
    cout << "vector size: " << v.size() << endl;

    // Print a separator line
    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Printing Vector Data
    // ============================================================

    // ------------------------------------------------------------
    // 1. Using Normal For Loop
    // ------------------------------------------------------------

    cout << "vector data using for loop: \t\t";

    // Access vector elements using their index
    for(int i=0; i<v.size(); i++){

        // v[i] accesses the element at index i
        // v.at(i) can also be used instead of v[i]
        cout << v[i] << " ";
    }

    cout << endl;


    // ------------------------------------------------------------
    // 2. Using Range-Based For Each Loop
    // ------------------------------------------------------------

    cout << "vector data using for each loop: \t";

    // Automatically visits every element of the vector
    for(int i: v){

        cout << i << " ";
    }

    cout << endl;


    // ------------------------------------------------------------
    // 3. Using Iterators
    // ------------------------------------------------------------

    cout << "vector data using iterators: \t\t";

    // Declare a vector iterator
    vector<int>::iterator it;

    // begin() returns an iterator pointing to the first element
    it = v.begin();

    // Continue until the iterator reaches the end
    while(it != v.end()){

        // *it gives the value pointed to by the iterator
        cout << *it << " ";

        // Move iterator to the next element
        it++;
    }

    cout << endl;

    // Print separator
    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Front and Back Elements
    // ============================================================

    cout << "Front and Back: ";

    // front() returns the first element
    // back() returns the last element
    cout << v.front() << ", " << v.back() << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Empty Check, Size, Capacity and Maximum Size
    // ============================================================

    cout << "Empty check and size, capacity\n";

    // empty() returns true if the vector contains no elements
    // Here we check whether the vector is NOT empty
    if (v.empty() == false){

        cout << "Vector is not empty" << endl;

        // Number of elements currently stored
        cout << "Vector size: " << v.size() << endl;

        // Amount of storage currently allocated
        cout << "Capacity: " << v.capacity() << endl;

        // Maximum number of elements the vector can theoretically hold
        cout << "Max-Size: " << v.max_size() << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Clear and Erase
    // ============================================================

    cout << "Clear and Erase";

    // Create and initialize a second vector
    vector<int> v2 = {70, 80, 90, 100, 110};

    cout << "second vector data before erase: \t";

    // Display all elements of v2
    for(int i: v2){

        cout << i << " ";
    }

    cout << endl;


    // Set iterator to the beginning of v2
    it = v2.begin();

    // Erase elements from it+1 up to, but not including, it+3
    // Therefore, 80 and 90 will be removed
    v2.erase(it+1, it+3);

    cout << "second vector data after erase: \t\t\t";

    // Display vector after erase
    for(int i: v2){

        cout << i << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Clear Vector
    // ============================================================

    // Remove all elements from v2
    v2.clear();

    cout << "second vector data after clear: \t\t\t\n";

    // Check whether v2 is empty
    if (v2.empty() == true){

        cout << "Vector is empty" << endl;

        // Size becomes 0 after clear()
        cout << "Vector size: " << v2.size() << endl;

        // clear() removes elements but usually does not reduce capacity
        cout << "Capacity: " << v2.capacity() << endl;

        // Maximum possible size of the vector
        cout << "Max-Size: " << v2.max_size() << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Vector Insert and Swap
    // ============================================================

    cout << "vector: insert and swap\n";

    // Create and initialize a third vector
    vector<int> v3 = {900, 1000, 2000, 201, 730};

    cout << "Before Insert: \t";

    // Display v3 before insertion
    for(int i: v3){

        cout << i << " ";
    }

    cout << endl;


    // Set iterator to the beginning of v3
    it = v3.begin();

    // Insert -1 at position it+1
    // The existing elements are shifted to the right
    v3.insert(it+1, -1);

    cout << "After Insert: \t";

    // Display v3 after insertion
    for(int i: v3){

        cout << i << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Swapping Two Vectors
    // ============================================================

    // Exchange all elements of v and v3
    v.swap(v3);

    cout << "After Swap v: \t\t";

    // Display vector v after swap
    for(int i: v){

        cout << i << " ";
    }

    cout << endl;


    cout << "After Swap v3: \t\t";

    // Display vector v3 after swap
    for(const int& i: v3){

        cout << i << " ";
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}