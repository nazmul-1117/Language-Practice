#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]){

    cout << "C++ Standard: " << __cplusplus << endl << endl;

    int nDot = 60;

    vector<int> v = {
        10, 15, 20
    };

    vector<int> w = {
        10, 20, 30
    };

    // function 1 -accumulate
    int sum = accumulate(v.begin(), v.end(), 0);
    cout << "Accumulate Sum: " << sum << endl;
    
    // function 2: inner_product
    sum = inner_product(v.begin(), v.end(), w.begin(), 0);
    cout << "Inner Product Sum: " << sum << endl;

    // function 3: partial sum
    vector<int> x = {
       10, 20, 30, 40, 50
    };
    vector<int> result(x.size());

    cout << "Original Data ";
    for (int i: x){
        cout << i << " ";
    }
    cout <<endl;

    partial_sum(x.begin(), x.end(), result.begin());
    cout << "Print partial sum ";
    for (int i: result){
        cout << i << " ";
    }
    cout << endl;


    vector<int> y(x.size());
    iota(y.begin(), y.end(), 101);
    cout << "Data of y: ";
    for (int i: y){
        cout << i << " ";
    }
    cout <<endl;





    return 0;
}
