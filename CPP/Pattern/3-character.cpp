#include <bits/stdc++.h>
using namespace std;

void forLoop(const int& n){
	for (int i = 1; i <= n; ++i){
		for (int j = 1; j <= n; ++j){
			// cout << (char)(i+64) << " ";
			cout << (char)(j+64) << " ";
		}
		cout << endl;
	}
}

void whileLoop(const int& n){
	int i=1;
	while(i <= n){
		int j=1;
		while(j <= n){

			// cout << (char)(i+64) << " ";
			cout << (char)(j+64) << " ";

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
	for(int i=1; i <= j; i++){
		
			// cout << (char)(i+64) << " ";
			cout << (char)(j+64) << " ";
			
	}
	cout << endl;
	recursionLoop(n, j+1);
}

int32_t main(){

	int n = 4;
	// forLoop(n);
	// whileLoop(n);
	recursionLoop(n, 1);
}