#include <iostream>
#include<string>
using namespace std;

int main(){
    long long digit=0;
    int a=0;
    string str = "NULL";

    cout<<"Enter a Number as String: ";
    cin>>str;

    for(int i=str.length()-1, j=1; i >=0 ; i--, j*=10){
            if(('0' <= str[i]) && ('9' >= str[i])){
                a = str[i]-48;
                digit += (a*j);
            }
    }
    cout<<"Your Number is as an Integers: ";
    cout<<digit<<endl;
}
