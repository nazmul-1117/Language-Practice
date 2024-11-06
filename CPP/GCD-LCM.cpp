#include <bits/stdc++.h>
#define lld long long int

using namespace std;

lld gcd(int a, int b){
	if (b == 0){
		return a;
	}
	return gcd(b, a%b);
}

lld lcm(int a, int b){
	return (a*b)/gcd(a, b);
}

int32_t main(){
    ios_base :: sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int a, b;
    cin >> a >> b;

    cout << "GCD: " << gcd(a, b) << endl;
    cout << "LCM: " << lcm(a, b) << endl;

  
}