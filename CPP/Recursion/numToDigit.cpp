#include <iostream>
#define lld long long unsigned

using namespace std;

void digitToName(string *name, lld number){

        if(number == 0){
            return ;
        }

        lld lastDigit = number%10;
        number /= 10;

        digitToName(name, number);

        cout<<name[lastDigit];
}

int main(){
    string name[10] = {"Zero ", "One ", "Two ", "Three ", "Four ",
                                    "Five ", "Six ", "Seven ", "Eight ", "Nine " };

    lld number = 689;
    cout<<"Enter Your Number: ";
    cin>>number;
    cout<<"Your Digit: "<<number<<endl;

    cout<<"Number is English Name: ";
    digitToName(name, number);
    cout<<endl;

}
