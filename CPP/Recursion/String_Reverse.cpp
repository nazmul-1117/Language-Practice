#include <iostream>
using namespace std;

void displayData(char* ch, int n){

    for(int i=0; i <= n; i++){
        cout<<*(i+ch);
    }
    cout<<endl;
}

void stringReverse(char* ch, int b, int e){
    if(b > e){
        return ;
    }
    swap(ch[b], ch[e]);
    b++; e--;
    stringReverse(ch, b, e);
}

void stringReverse_Loop(char* ch, int n){
    int b = 0, e = n-1;

    while(b < e){
        swap(ch[b], ch[e]);
        b++;
        e--;
    }
}

int main(){
    int n=26, b=0;
    char ch[n] = {"abcdefghijklmnopqrstuvwxyz"};

    n=25;
    char ch2[n] = {"Nazmul"};
    stringReverse(ch, b, n);
    displayData(ch, n);
}
