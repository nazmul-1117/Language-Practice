#include<iostream>
#include<list>

using namespace std;

int main(int argc, char const *argv[]){

    // Number of '-' characters used as a separator
    int nDot = 50;
    
    list<int> myList;

    myList.push_back(0);

    myList.push_back(10);
    myList.push_back(20);
    myList.push_back(30);
    myList.push_back(40);

    myList.push_front(-10);
    myList.push_front(-20);
    myList.push_front(-30);
    myList.push_front(-40);

    myList.pop_front();
    myList.pop_back();

    // Display the number of elements currently stored in the vector
    cout << "My List size: " << myList.size() << endl;

    // Print a separator line
    cout << string(nDot, '-') << endl << endl;

    cout << "myList data: \t\t";
    for (int e: myList){
        cout << e << " ";
    }
    cout << endl;


    cout << "List data using iterators: \t\t";

    // Declare a vector iterator
    list<int>::iterator it;

    // begin() returns an iterator pointing to the first element
    it = myList.begin();

    // Continue until the iterator reaches the end
    while(it != myList.end()){

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
    cout << myList.front() << ", " << myList.back() << endl;

    cout << string(nDot, '-') << endl << endl;


    // Remove data
    myList.remove(10);
    cout << "myList data after remove: \t\t";
    for (int e: myList){
        cout << e << " ";
    }
    cout << endl;


    return 0;
}
