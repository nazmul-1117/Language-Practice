#include <iostream>
using namespace std;

// Level 1: Base Class (Grandparent)
class Grandfather {
private:
    int x;
    int *y;

public:

    Grandfather(int x, int y)
        :x(x), y(new int(y)) {}

    void pullGrandpaWeeds() const {
        cout << "Grandfather's gardening skills inherited!" << endl;
    }

    void setX(int x){
        this -> x = x;
    }

    int getX() const {
        return x;
    }

    void setY(int y){
        *(this -> y) = y;
    }

    int getY() const {
        return *y;
    }
};


int main() {

    // type1: When Pointer const: const Grandfather *obj -> then value is const 
    // the object has type qualifiers that are not compatible with the member function "Grandfather::setX" const-type.cpp(41, 5): object type is: const Grandfather with  `const Grandfather *obj = new Grandfather(10, 15);`


    // Type: 2 -> const pointer address, non const data
    // Error:  expression must be a modifiable lvalue
    // Grandfather* const obj = new Grandfather(10, 15);
    // Grandfather* obj2 = new Grandfather(1, 5);
    // obj = obj2;


    // Type: 3 -> const pointer const data
    const Grandfather* const obj = new Grandfather(10, 15);
    
    // Error 1: expression must be a modifiable lvalue
    // Grandfather* obj2 = new Grandfather(1, 5);
    // obj = obj2;

    // Error 2: the object has type qualifiers that are not compatible with the member function "Grandfather::setY" const-type.cpp(59, 5): object type is: const Grandfather
    // obj->setX(99);
    // obj->setY(99);
    
    
    cout << "Data: ";
    cout << obj->getX() << ", " << obj->getY() << endl;

    return 0;
}