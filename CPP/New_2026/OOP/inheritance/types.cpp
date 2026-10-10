#include<iostream>
using namespace std;

// Single Class
class A{

    private:
        int a;

    public:
        A(): a(0){}
        A(int a): a(a) {}

        void setA(int a){
            this -> a = a;
        }

        int getA() const {
            return a;
        }

        void print() const {
            cout << "I am from class A"<<endl;
            printf("The value of 'a': %d\n\n", a);
        }

        void xd() const {
            cout << "I am " << __FUNCTION__ <<" From class A" << endl;
        }
};

class A2{

    private:
        int a;

    public:
        A2(): a(0){}
        A2(int a): a(a) {}

        void setA2(int a){
            this -> a = a;
        }

        int getA2() const {
            return a;
        }

        void print() const {
            cout << "I am from class A2"<<endl;
            printf("The value of 'a2': %d\n\n", a);
        }

        void xd() const {
            cout << "I am " << __FUNCTION__ <<" From class A2" << endl;
        }
};


// multi-level class type
class B: public A{
    private:
        int b;

    public:
        B(): b(0), A(0) {}
        B(int x, int y=0): b(x), A(y) {}

        void setB(int x){
            this -> b = x;
        }

        int getB() const {
            return b;
        }

        void print() const {
            cout << "I am from class B"<<endl;
            printf("The value of 'a'=%d and 'b'=%d\n\n", getA(), b);
        }
};


// multiple class typef
class C: public A, public A2{
    private:
        int c;

    public:
        C(): c(0), A(0), A2(0) {}
        C(int x, int y=0, int z=0): c(x), A(y), A2(z) {}

        void setC(int x){
            this -> c = x;
        }

        int getC() const {
            return c;
        }

        void print() const {
            cout << "I am from class C"<<endl;
            printf("The value of 'a'=%d and 'a2'=%d and 'c'=%d\n\n", getA(), getA2(), c);
        }
};


// hierarchical class type
// hybrid class type



int main(int argc, char const *argv[]){

    int xa=10, xb=20, xc=30, xd=40, xe=50;
    
    // single class type
    A a(xa);
    a.print();

    // multi-level class type
    B b(xb, xa);
    b.print();

    // multiple class typef
    C c(xb, xa, xc);
    c.A::xd();
    c.A2::xd();
    c.print();
    
    // hierarchical class type
    // hybrid class type

    return 0;
}
