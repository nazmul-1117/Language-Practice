#include <iostream>
using namespace std;

class Payment {

public:
    virtual void pay(double amount) = 0;
};

class Bkash : public Payment {

public:
    void pay(double amount) override {
        cout << "Paid " << amount << " using Bkash" << endl;
    }
};

class Card : public Payment {

public:
    void pay(double amount) override {
        cout << "Paid " << amount << " using Card" << endl;
    }
};

int main() {

    Payment* p1 = new Bkash();
    Payment* p2 = new Card();

    p1->pay(500);
    p2->pay(1000);

    delete p1;
    delete p2;
}