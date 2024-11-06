#include <bits/stdc++.h>
#define PI 3.14

using namespace std;

inline int getSum(int a, int b=0){
	return a+b;
}

inline int getMax(int& a, int& b){
	return (a>b) ? a : b;
}

int main(){

	// PI = 3.14; error code, cannot be change

	int a = 5;
	int b = 10;

	int ans;

	ans = getMax(a, b);
	cout << "Max data: " << ans << endl;

	ans = getSum(a);
	cout << "Sum: " << ans << endl;
  
}