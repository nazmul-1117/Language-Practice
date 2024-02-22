#include <iostream>
using namespace std;

void isPrime(){
    int n=50000, c;
    bool isPrime = true;

   for(int j=2; j<=n; j++){
        for(int i=2; i*i<=j; i++){
        isPrime = true;
            if(j%i == 0){
                isPrime = false;
                break;
            }
        }

        if(isPrime){
            cout<<j<<" ";
                    c++;
        }
    }
    cout<<endl<<endl;
    cout<<"Counter = "<<c<<endl;
}

void isPrime2(){
    int n=50000, c;
    bool isPrime = true;

   for(int j=2; j<=n; j++){

        if(j&1){

            for(int i=2; i*i<=j; i++){
            isPrime = true;
                if(j%i == 0){
                    isPrime = false;
                    break;
                }
            }

        if(isPrime){
            cout<<j<<" ";
            c++;
        }
    }
   }
    cout<<endl<<endl;
    cout<<"Counter = "<<c<<endl;
}


void Sieve_of_Eratosthenes(){
    int n=500;
    bool *arr = new bool[n+1];

    for(int i=2; i<=n; i++){
        arr[i] = true;
    }
    arr[0] = arr[1] = false;

    for(int i=2; i<=n; i++){

        if(arr[i]){

            for(int j=i*2; j<n; j += i){
             arr[j] = false;
            }
        }

    }

    for(int i=0; i<n; i++){

        if(arr[i]){
            cout<<i<<" ";
        }
    }

    cout<<endl<<endl;

    delete[] arr;
}

int main(){

    //isPrime();
    //isPrime2();
    Sieve_of_Eratosthenes();
}
