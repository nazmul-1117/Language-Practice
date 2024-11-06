#include <bits/stdc++.h>
using namespace std;

int32_t main(){

	int n;
	cin >> n;

	int *arr = new int(n);

	// input array
	for (int i = 0; i < n; ++i){
		cin >> i[arr];
	}

	// output array
	cout << "Array's Data: ";
	for (int i = 0; i < n; ++i){
		cout << *(arr+i) << " ";
	}
	cout << endl;


	// find the max data from an array
	int maxData = INT_MIN;
	for (int i = 0; i < n; ++i){
		maxData = max(maxData, arr[i]);
	}
	cout << "Maximum Data: " << maxData << endl;

	// find the min data from an array
	int minData = INT_MAX;
	for (int i = 0; i < n; ++i){
		minData = min(minData, arr[i]);
	}
	cout << "Minimum Data: " << minData << endl;


	delete[] arr;
}