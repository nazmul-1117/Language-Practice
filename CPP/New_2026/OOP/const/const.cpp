#include <iostream>
using namespace std;

// Level 1: Base Class (Grandparent)
class Grandfather {
private:
    int x;

public:
    void pullGrandpaWeeds() const {
        cout << "Grandfather's gardening skills inherited!" << endl;
    }

    void setX(int x){
        this -> x = x;
    }
};

// Level 2: Derived Class (Parent) inherits from Level 1
class Father : public Grandfather {
private:
    int x;

public:
    void driveFatherCar() const {
        cout << "Father's driving skills inherited!" << endl;
    }

    void setX(int x){
        this -> x = x;
    }
};

// Level 3: Further Derived Class (Child) inherits from Level 2
class Child : public Father {
private:
    int x;

public:
    void playVideoGames() const {
        cout << "Child is playing video games." << endl;
    }

    void setX(int x){
        this -> x = x;
    }
};


void printAll(const Child& obj){
    
    // The child object can access methods from all 3 levels
    obj.playVideoGames();    // Local method (Level 3)
    obj.driveFatherCar();    // Inherited from Father (Level 2)
    obj.pullGrandpaWeeds();  // Inherited from Grandfather (Level 1)

    // Error: the object has type qualifiers that are not compatible with the member function "Child::setX" const.cpp(52, 5): object type is: const Child
    // obj.setX(10); 
}

int main() {

    // Instantiate the 3rd-level class
    Child obj;

    printAll(obj);

    return 0;
}
