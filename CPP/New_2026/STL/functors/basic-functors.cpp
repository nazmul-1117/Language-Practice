#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>

using namespace std;

class isEven{
public:
    bool operator() (int a){
        return (a%2 == 0);
    }
};

class greterFind{
public:
    bool operator() (int a, int b){
        return a<b;
    }
};


int main(int argc, char const *argv[]){
    isEven even;

    if (even(11))
        cout << "Even number" << endl;
    else
        cout << "Odd number" << endl;

    greterFind greater;
    if( greater(15, 11) )
        cout << "b is greater" << endl;
    else
        cout << "a is greater" << endl;

    return 0;
}
