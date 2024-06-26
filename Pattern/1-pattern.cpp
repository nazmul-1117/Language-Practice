#include <bits/stdc++.h>
using namespace std;

void forLoop(const int& n){
	for (int i = 1; i <= n; ++i){
		for (int j = 1; j <= n; ++j){
			cout << j << " ";
			// cout << n-j+1  << " ";
		}
		cout << endl;
	}
}

void whileLoop(const int& n){
	int i=1;
	while(i <= n){
		int j=1;
		while(j <= n){
			cout << j << " ";
			j++;
		}
		cout << endl;
		i++;
	}
}

void recursionLoop(const int& n, int j){
	if (j > n){
		return;
	}
	for(int i=1; i <= 3; i++){
		cout << i << " ";
	}
	cout << endl;
	recursionLoop(n, j+1);
}

int32_t main(){

	int n = 3;
	forLoop(n);
	// whileLoop(n);
	// recursionLoop(n, 1);
}