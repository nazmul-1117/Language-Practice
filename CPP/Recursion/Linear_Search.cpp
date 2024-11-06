#include <iostream>
#include <ctime>
#include <conio.h>
using namespace std;

void displayData(int *arr, int n){
    for(int i=0; i<n; i++){
        cout<<i[arr]<<" ";
    }
    cout<<endl;
}

void setData(int *arr, int n){
    srand(time(0));
    for(int i=0; i<n; i++){
        *(arr+i) = rand()%100;
    }
    cout<<endl;
}

bool linearSearch(int *arr, int n, int item){
    if(n == 0){
        return false;
    }

    if(arr[0] == item){
        return true;
    }

   bool ans = linearSearch(arr+1, n-1, item);
   return ans;
}

int main(){
        int n=10;
        int arr[n];
        int item;

        while(true){
                system("cls");
                setData(arr, n);
                cout<<"Your Array: ";
                displayData(arr, n);

                cout<<"Linear Search Item: ";
                cin>>item;

                if(linearSearch(arr, n, item))
                {
                    cout<<item<<" Found"<<endl;
                }
                else{
                    cout<<item<<" Not Found"<<endl;
                }

                getch();
        }
}
