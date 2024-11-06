#include <iostream>
//#define lld long long int
#define lld long double
using namespace std;

lld Power(int a, int b){
   if(b == 0){
    return 1;
   }

   lld ans = a * Power(a, b-1);
   return ans;
}

int main(){
    int a, b;
    a = 2, b = 5000;
    lld ans = Power(a, b);
    cout<<ans;
}

