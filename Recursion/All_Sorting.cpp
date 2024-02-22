#include <bits/stdc++.h>
using namespace std;

void displayData(int* &arr, int &n){
    cout<<" ";
        for(int i=0; i<n; i++){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

void bubbleSort(int* &arr, int n){

    if(n < 0){
        return ;
    }

    for(int i=0; i<n-1; i++){
        if(arr[i] > arr[i+1]){
            swap(arr[i], arr[i+1]);
        }
    }
    bubbleSort(arr, n-1);
}

void selectionSort(int* &arr, int &n){
    if(n < 0){
        return ;
    }

}

void mergeSort(){

}

int partision(int* &arr, int st, int en) {

    int pivot = arr[st];
    int cnt=0;

    for(int i=st+1; i<=en; i++){
        if(pivot >= arr[i]){
            cnt++;
        }
    }

    int pIndex = st+cnt;
    printf("pivot = %d\n", pIndex);
    swap(arr[st], arr[pIndex]);

    int i=st, j=en;
    while(i < pIndex && j > pIndex){

        while(arr[i] <= pivot && i<pIndex){
            i++;
        }

        while(arr[j] > pivot && j>pIndex){
            j--;
        }

        if(i < pIndex && j > pIndex){
            swap(arr[i], arr[j]);
            i++; j--;
        }
    }

    return pIndex;
}

void quickSort(int* &arr, int st, int en){
    if(st >= en){
        return ;
    }

    int p = partision(arr, st, en);

    quickSort(arr, st, p-1);

    //for right partision;
    quickSort(arr, p+1, en);
}

int main(){

    system("Color E");
    //system("Color E4");

    srand(time(0));
    int n=10;
    int *arr = new int[n];
    for(int i=0; i<n; i++){
        arr[i] = rand()%100;
    }

    displayData(arr, n);
    system("Color 0A");
    //bubbleSort(arr, n);
    //cout<<"After Bubble Sort: ";
    //displayData(arr, n);
    cout<<endl;
    int st=0;

    quickSort(arr, st,  n-1);
    cout<<"After Quick Sort: ";
    displayData(arr, n);


    delete[] arr;

}
