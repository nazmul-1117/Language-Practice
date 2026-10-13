#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]){

    // ============================================================
    // 2D Vector Basics
    // ============================================================

    vector<int> x; // 1-D array

    // 2D vector with 4 rows only
    vector<vector<int>> y(4);

    // 2D vector with 4 rows and 3 columns
    // All elements are initialized with 0
    vector<vector<int>> v(4, vector<int>(3, 0));


    // ============================================================
    // Find Row and Column Size
    // ============================================================

    int rowNumber = v.size();
    int colNumber = v[0].size();


    // ============================================================
    // Print 2D Array
    // ============================================================

    cout << "Print mXn Matrix: \n";

    for (auto e: v){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }

    cout << endl;


    // ============================================================
    // Initialize 2D Vector with Different Column Sizes
    // ============================================================

    vector<vector<int>> vx(4); // how many rows only

    vx[0] = vector<int>(4, -1);
    vx[1] = vector<int>(2, -2);
    vx[2] = vector<int>(4, -3);
    vx[3] = vector<int>(3, -4);


    // Print manually initialized 2D vector
    cout << "Print New manual Matrix: \n";

    for (auto e: vx) {

        for(int i: e) {

            cout << i << " ";
        }

        cout << endl;
    }

    cout << endl;


    cout << "Total Row: " << rowNumber
         << ", Total Column: " << colNumber << endl;


    // ============================================================
    // Access Individual Elements
    // ============================================================

    cout << "\nAccess Individual Elements:\n";

    // v[row][column]
    cout << "v[0][0] = " << v[0][0] << endl;
    cout << "v[1][2] = " << v[1][2] << endl;


    // ============================================================
    // Update Individual Elements
    // ============================================================

    cout << "\nUpdate Elements:\n";

    v[0][0] = 10;
    v[0][1] = 20;
    v[0][2] = 30;

    v[1][0] = 40;
    v[1][1] = 50;
    v[1][2] = 60;

    cout << "Updated Matrix:\n";

    for(auto e: v){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }


    // ============================================================
    // Add a New Row
    // ============================================================

    cout << "\nAfter Adding New Row:\n";

    // Add a row containing 3 elements
    v.push_back(vector<int>{70, 80, 90});

    for(auto e: v){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }


    // ============================================================
    // Add a New Column
    // ============================================================

    cout << "\nAfter Adding New Column:\n";

    // Add one element to every row
    for(auto &e: v){

        e.push_back(100);
    }

    for(auto e: v){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }


    // ============================================================
    // Remove Last Row
    // ============================================================

    cout << "\nAfter Removing Last Row:\n";

    v.pop_back();

    for(auto e: v){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }


    // ============================================================
    // Remove Last Column
    // ============================================================

    cout << "\nAfter Removing Last Column:\n";

    for(auto &e: v){

        e.pop_back();
    }

    for(auto e: v){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }


    // ============================================================
    // Using Normal For Loop with Index
    // ============================================================

    cout << "\nUsing Normal For Loop:\n";

    for(int i = 0; i < v.size(); i++){

        for(int j = 0; j < v[i].size(); j++){

            cout << v[i][j] << " ";
        }

        cout << endl;
    }


    // ============================================================
    // Using at() Function
    // ============================================================

    cout << "\nUsing at() Function:\n";

    // at(row).at(column)
    cout << "v.at(0).at(0) = " << v.at(0).at(0) << endl;


    // ============================================================
    // Find Number of Rows
    // ============================================================

    cout << "\nNumber of Rows: " << v.size() << endl;


    // ============================================================
    // Find Number of Columns
    // ============================================================

    cout << "Number of Columns in First Row: "
         << v[0].size() << endl;


    // ============================================================
    // Check Whether 2D Vector is Empty
    // ============================================================

    cout << "\nEmpty Check:\n";

    if(v.empty()){

        cout << "Vector is empty" << endl;

    }else{

        cout << "Vector is not empty" << endl;
    }


    // ============================================================
    // Clear Entire 2D Vector
    // ============================================================

    vector<vector<int>> temp = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    cout << "\nBefore Clear:\n";

    for(auto e: temp){

        for(int i: e){

            cout << i << " ";
        }

        cout << endl;
    }

    // Remove all rows
    temp.clear();

    cout << "\nAfter Clear:\n";

    if(temp.empty()){

        cout << "2D Vector is empty" << endl;
    }


    return 0;
}