#include <iostream>
#include <string>
using namespace std;

bool isPallindrom(string& name, int b, int l){
    if(b > l){
        return true;
    }

    if(name[b] != name[l]){
        return false;
    }

    return isPallindrom(name, b+1, l-1);
}


int main(){
    int n=26, b=0;
    string name = "abccba";

    int l = name.length()-1;
    bool ans = isPallindrom(name, b, l);

    if(ans){
        cout<<"Pallindrom"<<endl;
    }
    else{
        cout<<"Not Pallindrom"<<endl;
    }
}
