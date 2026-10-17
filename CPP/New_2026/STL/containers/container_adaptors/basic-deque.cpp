#include <iostream>
#include <deque>

using namespace std;

int main(int argc, char const *argv[]){

    // ============================================================
    // Basic Configuration
    // ============================================================

    // Number of '-' characters used as a separator
    int nDot = 50;


    // ============================================================
    // Deque Initialization
    // ============================================================

    // Create an empty deque of integers
    deque<int> dq;


    // ============================================================
    // Insert Elements into Deque
    // ============================================================

    // push_back() adds an element to the BACK
    dq.push_back(10);
    dq.push_back(20);
    dq.push_back(30);

    // push_front() adds an element to the FRONT
    dq.push_front(5);
    dq.push_front(1);


    // ============================================================
    // Print Deque
    // ============================================================

    cout << "Deque data: ";

    for(int i: dq){

        cout << i << " ";
    }

    cout << endl;


    // ============================================================
    // Size of Deque
    // ============================================================

    // size() returns the number of elements
    cout << "Size: " << dq.size() << endl;


    // ============================================================
    // Front and Back
    // ============================================================

    // front() returns the first element
    cout << "Front: " << dq.front() << endl;

    // back() returns the last element
    cout << "Back: " << dq.back() << endl;


    // ============================================================
    // Remove Elements from Deque
    // ============================================================

    // pop_front() removes the first element
    dq.pop_front();

    // pop_back() removes the last element
    dq.pop_back();


    cout << "After pop_front() and pop_back(): ";

    for(int i: dq){

        cout << i << " ";
    }

    cout << endl;


    // ============================================================
    // Check Whether Deque is Empty
    // ============================================================

    cout << "Checking Empty or not" << endl;

    if(dq.empty() == false){

        cout << "Deque is not Empty" << endl;

        cout << "Size: " << dq.size() << endl;

        cout << "Front: " << dq.front() << endl;

        cout << "Back: " << dq.back() << endl;
    }

    else{

        cout << "Deque is Empty" << endl;
    }


    // ============================================================
    // Access Elements Using Index
    // ============================================================

    cout << "\nAccess elements using index: ";

    for(int i = 0; i < dq.size(); i++){

        cout << dq[i] << " ";
    }

    cout << endl;


    // ============================================================
    // Access Elements Using at()
    // ============================================================

    // at() provides bounds-checked access
    cout << "First element using at(): "
         << dq.at(0) << endl;


    // ============================================================
    // Insert Element at Specific Position
    // ============================================================

    // Insert 100 at index 1
    dq.insert(dq.begin() + 1, 100);

    cout << "After insert: ";

    for(int i: dq){

        cout << i << " ";
    }

    cout << endl;


    // ============================================================
    // Erase Element
    // ============================================================

    // Erase the element at index 1
    dq.erase(dq.begin() + 1);

    cout << "After erase: ";

    for(int i: dq){

        cout << i << " ";
    }

    cout << endl;


    // ============================================================
    // Create Second Deque
    // ============================================================

    deque<int> dq2;

    dq2.push_back(100);
    dq2.push_back(200);
    dq2.push_back(300);
    dq2.push_back(400);
    dq2.push_back(500);


    // ============================================================
    // Swap Two Deques
    // ============================================================

    // swap() exchanges the complete contents
    // of dq and dq2
    dq.swap(dq2);


    // ============================================================
    // Check Deques After Swap
    // ============================================================

    cout << "Print `dq`, front: "
         << dq.front()
         << ", back: "
         << dq.back() << endl;

    cout << "Print `dq2`, front: "
         << dq2.front()
         << ", back: "
         << dq2.back() << endl;


    // ============================================================
    // Clear Deque
    // ============================================================

    // clear() removes all elements
    dq2.clear();

    cout << "After clear(), dq2 size: "
         << dq2.size() << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}