#include <iostream>
#include <queue>

using namespace std;

int main(int argc, char const *argv[]){

    // ============================================================
    // Basic Configuration
    // ============================================================

    // Number of '-' characters used as a separator
    int nDot = 50;


    // ============================================================
    // Queue Initialization
    // ============================================================

    // Create an empty queue of integers
    queue<int> q;


    // ============================================================
    // Push Elements into Queue
    // ============================================================

    // push() adds an element to the BACK of the queue
    q.push(10);
    q.push(20);
    q.push(30);
    q.push(40);
    q.push(50);
    q.push(60);


    // ============================================================
    // Remove Front Element
    // ============================================================

    // pop() removes the element from the FRONT of the queue
    // Here, 10 will be removed
    q.pop();


    // ============================================================
    // Check Whether Queue is Empty
    // ============================================================

    cout << "Checking Empty or not" << endl;

    // empty() returns true if the queue contains no elements
    if (q.empty() == false){

        // size() returns the number of elements
        cout << "Size: " << q.size() << endl;

        // front() returns the first element
        // without removing it
        cout << "Front: " << q.front() << endl;

        // back() returns the last element
        // without removing it
        cout << "Back: " << q.back() << endl;
    }

    else{

        cout << "Queue is Empty";
    }


    // ============================================================
    // Create Second Queue
    // ============================================================

    queue<int> q2;

    // Add elements to the second queue
    q2.push(100);
    q2.push(200);
    q2.push(300);
    q2.push(400);
    q2.push(500);


    // ============================================================
    // Swap Two Queues
    // ============================================================

    // swap() exchanges the complete contents of q and q2
    q.swap(q2);


    // ============================================================
    // Check Queue After Swap
    // ============================================================

    // Before swap:
    // q  = 20, 30, 40, 50, 60
    // q2 = 100, 200, 300, 400, 500
    //
    // After swap:
    // q  = 100, 200, 300, 400, 500
    // q2 = 20, 30, 40, 50, 60

    cout << "Print `q`, front: " << q.front()
         << ", back: " << q.back() << endl;

    cout << "Print `q2`, front: " << q2.front()
         << ", back: " << q2.back() << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}