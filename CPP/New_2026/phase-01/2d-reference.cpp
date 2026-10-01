
#include<iostream>
using namespace std;

void displayOneD(int* arr, int& n);
void displayTwoD(int** arr, int& n, int& m);

void oneDReference(){

    cout << "One-D reference\n-----------------------------------------------------\n";
    int n = 5; 
    int* ptr1 = new int[5];

    *ptr1 = 101;
    *(ptr1+1) = 201;
    *(ptr1+2) = 301;
    *(ptr1+3) = 401;
    *(ptr1+4) = 501;

    displayOneD(ptr1, n);
    
    delete[] ptr1;
    std :: cout << "delete pointer successfully\n";
}

void twoDReference(){

    cout << "Two-D reference\n-----------------------------------------------------\n";

    int n=5, m=7;
    
    int** ptr = new int*[n];

    for(int i=0; i<n; i++){
        ptr[i] = new int[m];
    }

    for (int i=0; i<n; ++i){
        for (int j=0; j<m; ++j){
            ptr[i][j] = j;
        }
    }

    displayTwoD(ptr, n, m);

    // free memory allocation
    for (int i=0; i<n; i++){
        delete[] ptr[i];
    }

    delete[] ptr;

    cout<< "memory free done\n";

}

void displayOneD(int* arr, int& n){
    
    for (size_t i = 0; i < n; i++) {
        std :: cout << arr[i] << std :: endl;
    }
}

void displayTwoD(int** arr, int& n, int& m){
    for (int i=0; i<n; ++i){
        for (int j=0; j<m; ++j){
            cout << arr[i][j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char const *argv[]) {

    oneDReference();
    twoDReference();

    return 0;
}
