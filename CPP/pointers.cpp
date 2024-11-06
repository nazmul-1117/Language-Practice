#include <iostream>
using namespace std;

int main(){
    int a , b, c, d, e;
    {
        a = 5;
        b = 20;
        c = 123;
    }



    int *ptra, *ptrb, *ptrc;
    ptra = ptrb = ptrc = NULL;

    ptra = &a;
    cout<<"(Adress of Ptr) "<<&ptra<<endl;
    cout<<"(data of Ptr) "<<ptra<<endl;
    cout<<"(Derefernce of Ptr) "<<*ptra<<endl<<endl;

    cout<<"(Value of B) "<<b<<endl;
    cout<<"(Adress of B) "<<&b<<endl;

    cout<<"(adress of ptr + 2 of Ptra) "<<(&ptra+2)<<endl;
    cout<<"(ptr + 2 of Ptra) "<<*(&ptra+2)<<endl<<endl<<endl;


    int arr[a] = {10, 25, 12, 32, 89};
    cout<<"(Array Part) "<<endl;
    for(int i=0; i<a; i++){
        cout<<i[arr]<<" ";
    }
    cout<<endl;
    cout<<"(Value of *arr): "<<*arr<<endl;
    cout<<"(Value of *(arr+1)): "<<*(arr+1)<<endl;

    for(int i=0; i<a; i++){
        cout<<*(i+arr)<<" ";
    }

}
