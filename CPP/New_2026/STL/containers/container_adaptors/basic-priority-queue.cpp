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
    // Priority Queue Initialization
    // ============================================================

    // Create an empty priority queue of integers
    // By default, priority_queue is a MAX-HEAP
    // The largest element always stays at the TOP
    priority_queue<int> pq;


    // ============================================================
    // Push Elements into Priority Queue
    // ============================================================

    pq.push(10);
    pq.push(50);
    pq.push(20);
    pq.push(40);
    pq.push(30);


    // ============================================================
    // Check Priority Queue
    // ============================================================

    cout << "Checking Empty or not" << endl;

    // empty() returns true if the priority queue contains no elements
    if(pq.empty() == false){

        cout << "Priority Queue is not Empty" << endl;

        // size() returns the number of elements
        cout << "Size: " << pq.size() << endl;

        // top() returns the highest-priority element
        // For default priority_queue, the largest value is the top
        cout << "Top: " << pq.top() << endl;
    }

    else{

        cout << "Priority Queue is Empty" << endl;
    }


    // ============================================================
    // Access Top Element
    // ============================================================

    cout << "\nTop Element: " << pq.top() << endl;


    // ============================================================
    // Remove Top Element
    // ============================================================

    // pop() removes the element with the highest priority
    // Here, 50 will be removed
    pq.pop();

    cout << "After pop(), Top: " << pq.top() << endl;


    // ============================================================
    // Print Priority Queue
    // ============================================================

    cout << "\nPriority Queue Elements: ";

    // A priority_queue does not support normal iteration
    // We use a temporary copy to print all elements
    priority_queue<int> temp = pq;

    while(!temp.empty()){

        cout << temp.top() << " ";

        temp.pop();
    }

    cout << endl;


    // ============================================================
    // Create Second Priority Queue
    // ============================================================

    priority_queue<int> pq2;

    pq2.push(100);
    pq2.push(500);
    pq2.push(300);
    pq2.push(200);
    pq2.push(400);


    // ============================================================
    // Swap Two Priority Queues
    // ============================================================

    // swap() exchanges the complete contents
    // of pq and pq2
    pq.swap(pq2);


    // ============================================================
    // Check Priority Queues After Swap
    // ============================================================

    cout << "Print `pq`, top: "
         << pq.top() << endl;

    cout << "Print `pq2`, top: "
         << pq2.top() << endl;


    // ============================================================
    // Clear Priority Queue
    // ============================================================

    // priority_queue does not have a clear() function.
    // We can remove all elements using pop().

    while(!pq2.empty()){

        pq2.pop();
    }

    cout << "After clearing pq2, size: "
         << pq2.size() << endl;


    // ============================================================
    // Min Priority Queue
    // ============================================================

    // By default:
    // priority_queue<int> = MAX-HEAP
    //
    // To create a MIN-HEAP:
    // use greater<int>

    priority_queue<int, vector<int>, greater<int>> minPQ;

    minPQ.push(50);
    minPQ.push(10);
    minPQ.push(40);
    minPQ.push(20);
    minPQ.push(30);


    cout << "\nMin Priority Queue Top: "
         << minPQ.top() << endl;


    // ============================================================
    // Print Min Priority Queue
    // ============================================================

    cout << "Min Priority Queue Elements: ";

    while(!minPQ.empty()){

        cout << minPQ.top() << " ";

        minPQ.pop();
    }

    cout << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}