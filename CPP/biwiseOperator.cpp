#include <iostream>
using namespace std;

void bitwiseOperator(int &a, int &b){
    cout<< " a = "<<a<<endl;
    cout<< " b = "<<b<<endl;

    int x = a | b;
    cout<< " a|b = " <<x<<endl;

    x = a & b;
    cout<< " a&b = " <<x<<endl;

    cout<< " ~a = " <<~a<<endl;

    cout<< " -a = " <<~a+1<<endl;
    cout<< " a<<2 = " <<(a<<2)<<endl;
    cout<< " a>>1 = " <<(a>>1)<<endl;

    cout<<endl;
}

bool isEven(int &a){

    if(a&1){
        return false;
    }
    return true;
}

int main(){
    int a = 100;
    int b = 3;
    bitwiseOperator(a, b);




     int n = 50;
     if(isEven(n)){
        cout<<" Even Number."<<endl;
     }
     else{
        cout<<" Odd Number."<<endl;
     }



}
