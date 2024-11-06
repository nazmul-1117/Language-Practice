#include <iostream>
#include <array>

using namespace std;

void Array(){
    array<int, 7> arr = {8, 1, 5, 7, 2, 0, 9};

    for(int i=0; i<arr.size(); i++){
        cout<<arr.at(i)<<" ";
    }
    cout<<endl;

    cout<<"Size: "<<arr.size()<<endl;
    cout<<"First: "<<arr.front()<<endl;
    cout<<"Last: "<<arr.back()<<endl;
    cout<<"Is Empty? "<<arr.empty()<<endl;

}

int main(){

    Array();

}
