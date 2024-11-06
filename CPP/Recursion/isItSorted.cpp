#include <iostream>
using namespace std;

int sumOfArraay(int *arr, int n){
    if(n <=0 ){
        return 0;
    }
    return arr[0] + sumOfArraay(arr+1, n-1);
}

bool isItSorted(int *arr, int n){
    if(n==0 || n==1){
        return true;
    }

    else if(arr[0] > arr[1]){
        return false;
    }

    return isItSorted(arr+1, n-1);
}

int main(){

    int n=4;
    ;
    int arr[n] = { 2, 1, 3, 4};;
    if(isItSorted(arr, n)){
        cout<<"Array Sorted"<<endl;
    }
    else{
        cout<<"Not Sorted"<<endl;
    }

    int sum = sumOfArraay(arr, n);
    cout<<"Sum = "<<sum<<endl;
}
