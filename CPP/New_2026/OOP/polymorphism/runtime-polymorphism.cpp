#include <iostream>

class Animal {
public:
    // Virtual function enables runtime polymorphism
    virtual void makeSound() {
        std::cout << "The animal makes a generic sound." << std::endl;
    }
    
    virtual ~Animal() = default; // Good practice: virtual destructor
};

class Dog : public Animal {
public:
    // Overriding the base class function
    void makeSound() override {
        std::cout << "The dog barks: Woof! Woof!" << std::endl;
    }
};

int main() {
    // 1. Polymorphic behavior using a Base Class Pointer
    Animal* myAnimal = new Dog();
    
    // Calls Dog's version because makeSound() is virtual
    myAnimal->makeSound(); 

    // 2. Accessing the overridden base function explicitly
    // You can still call the base implementation using the scope resolution operator (::)
    myAnimal->Animal::makeSound();

    delete myAnimal;
    return 0;
}
