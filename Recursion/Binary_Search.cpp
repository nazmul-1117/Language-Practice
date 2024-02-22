#include <iostream>
using namespace std;

void displayData(int *arr, int n, int b=0){\
    for(b; b < n; b++){
        cout<<*(b+arr)<<" ";
    }
    cout<<endl;
}

void binarySearch(int *arr, int &n, int &sItem){
    int first, last, mid;
    first = 0, last = n-1;

    while(first <= last){
        mid = (first+last)/2;

        if(arr[mid] == sItem){
            cout<<sItem<<" Found"<<endl;
            return ;
        }

        if(arr[mid] > sItem){
            last = mid - 1;
        }
        else{
            first = mid + 1;
        }
    }

    cout<<sItem<<" Not Found"<<endl;
}

bool Binary_Search(int *arr, int b, int n, int sItem){

    int right, left, mid;
    right = b, left = n;

    mid = (right+left)/2;

    cout<<"Your Array Data: ";
    displayData(arr, left, right);
    cout<<"Mid = "<<mid[arr]<<endl;



    if(right > left){
        return false;
    }

    if(arr[mid] == sItem){
        return true;
    }

    if(arr[mid] > sItem){
        left = mid-1;
    }
    else{
        right = mid+1;
    }

    bool ans = Binary_Search(arr, right, left, sItem);
    return ans;
}


int main(){

    int n=6, b=0;
    int arr[n] = {2, 4, 6, 10, 14, 16};
    int sItem = 16;

    //binarySearch(arr, n, sItem);

    //cout<<"Your Array Data: ";
    //displayData(arr, n);

    bool ans = Binary_Search(arr, b, n, sItem);

    if(ans){
        cout<<sItem<<" Data Found"<<endl;
    }
    else{
        cout<<sItem<<" Not Data Found"<<endl;
    }

}
