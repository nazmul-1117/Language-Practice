#include <iostream>
#include <stack>

using namespace std;

int main(int argc, char const *argv[]){

    // ============================================================
    // Basic Configuration
    // ============================================================

    // Number of '-' characters used as a separator
    int nDot = 50;


    // ============================================================
    // Stack Initialization
    // ============================================================

    // Create an empty stack of integers
    stack<int> st;


    // ============================================================
    // Push Elements into Stack
    // ============================================================

    // push() adds an element to the top of the stack
    st.push(10);
    st.push(20);
    st.push(30);
    st.push(40);
    st.push(50);
    st.push(60);


    // ============================================================
    // Remove Top Element
    // ============================================================

    // pop() removes the top element
    // Here, 60 will be removed
    st.pop();


    // ============================================================
    // Check Whether Stack is Empty
    // ============================================================

    cout << "Checking Empty or not" << endl;

    // empty() returns true if the stack contains no elements
    if (st.empty() == false){

        cout << "Stack is not Empty";

        // size() returns the number of elements
        cout << "Size: " << st.size() << endl;

        // top() returns the element at the top
        // without removing it
        cout << "Top: " << st.top() << endl;
    }

    else{

        cout << "Stack is Empty";
    }


    // ============================================================
    // Create Second Stack
    // ============================================================

    stack<int> st2;

    // Add elements to the second stack
    st2.push(100);
    st2.push(200);
    st2.push(300);
    st2.push(400);
    st2.push(500);


    // ============================================================
    // Swap Two Stacks
    // ============================================================

    // swap() exchanges the complete contents of st and st2
    st.swap(st2);


    // ============================================================
    // Check Stack After Swap
    // ============================================================

    // Before swap:
    // st  = 10, 20, 30, 40, 50
    // st2 = 100, 200, 300, 400, 500
    //
    // After swap:
    // st  = 100, 200, 300, 400, 500
    // st2 = 10, 20, 30, 40, 50

    cout << "Print `st`, top: " << st.top() << endl;

    cout << "Print `st2`, top: " << st2.top() << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}