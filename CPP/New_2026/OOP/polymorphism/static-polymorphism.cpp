#include<iostream>
using namespace std;

class A{
    private:
        int realNumber;
        int* imaginaryNumber;

    public:
        A(int a, int y)
            : realNumber(a), imaginaryNumber(new int(y)){

            }
         // method overload
         int abc(int x){
            cout << "One parameter 'int x'" << endl;

            return 1;
         }

         int abc(int x, int y){
            cout << "One parameter 'int x, int y'" << endl;

            return 1;
         }

         void abc(double x){
            cout << "One parameter 'int x'" << endl;
         }


         // operator overload
         A operator+ (A& b){
            int realNum = this -> realNumber + b.realNumber;
            int imagNum = *(this -> imaginaryNumber) + *(b.imaginaryNumber);

            return A(realNum, imagNum);
         }

         void print(){
            cout << "Complex Number: " << realNumber << " + i" << *imaginaryNumber << endl << endl;
         }
};


int main(int argc, char const *argv[])
{
    A obj(2, 5);
    A obj2(5, 7);

    A obj3 = obj+obj2;

    obj.print();
    obj2.print();
    obj3.print();

    return 0;
}
