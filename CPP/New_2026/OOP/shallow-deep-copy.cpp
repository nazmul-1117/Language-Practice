#include <iostream>
using namespace std;


// ======================================================
// SHALLOW COPY
// ======================================================

class Shallow {

private:
    int* data;
    int level;

public:

    // Constructor
    Shallow(int data, int level)
        : data(new int(data)), level(level) {
    }

    // Destructor
    ~Shallow() {
        delete data;
    }

    // Setter
    void setData(int value) {
        *data = value;
    }

    // Getter
    int getData() const {
        return *data;
    }

    // Setter
    void setLevel(int value) {
        level = value;
    }

    // Getter
    int getLevel() const {
        return level;
    }

    // Print
    void printData() const {
        cout << "Data: " << *data
             << " Level: " << level
             << endl;
    }
};


// ======================================================
// DEEP COPY
// ======================================================

class Deep {

private:
    int* data;
    int level;

public:

    // --------------------------------------------------
    // Constructor
    // --------------------------------------------------

    Deep(int data, int level)
        : data(new int(data)), level(level) {
    }


    // --------------------------------------------------
    // Copy Constructor
    // Deep Copy
    // --------------------------------------------------

    Deep(const Deep& source) {

        // Create NEW memory and copy the value
        data = new int(*source.data);

        // Copy normal variable
        level = source.level;
    }


    // --------------------------------------------------
    // Copy Assignment Operator
    // Deep Copy
    // --------------------------------------------------

    Deep& operator=(const Deep& source) {

        // Self-assignment check
        if (this == &source) {
            return *this;
        }

        // Delete old memory
        delete data;

        // Allocate NEW memory and copy the value
        data = new int(*source.data);

        // Copy level
        level = source.level;

        // Return current object
        return *this;
    }


    // --------------------------------------------------
    // Destructor
    // --------------------------------------------------

    ~Deep() {
        delete data;
    }


    // --------------------------------------------------
    // Setter
    // --------------------------------------------------

    void setData(int value) {
        *data = value;
    }


    // --------------------------------------------------
    // Getter
    // --------------------------------------------------

    int getData() const {
        return *data;
    }


    // --------------------------------------------------
    // Setter
    // --------------------------------------------------

    void setLevel(int value) {
        level = value;
    }


    // --------------------------------------------------
    // Getter
    // --------------------------------------------------

    int getLevel() const {
        return level;
    }


    // --------------------------------------------------
    // Print
    // --------------------------------------------------

    void printData() const {
        cout << "Data: " << *data
             << " Level: " << level
             << endl;
    }
};


// ======================================================
// MAIN
// ======================================================

int main() {

    // ==================================================
    // SHALLOW COPY EXAMPLE
    // ==================================================

    cout << "===== SHALLOW COPY =====" << endl;

    Shallow obj1(42, 77);

    // Default copy constructor performs SHALLOW COPY
    Shallow obj2 = obj1;

    cout << "Before changing obj1:" << endl;

    cout << "obj1: ";
    obj1.printData();

    cout << "obj2: ";
    obj2.printData();


    // Both objects point to the SAME memory
    obj1.setData(10);

    cout << "\nAfter changing obj1 data to 10:" << endl;

    cout << "obj1: ";
    obj1.printData();

    cout << "obj2: ";
    obj2.printData();


    /*
        IMPORTANT:

        obj1.data and obj2.data point to the same
        memory location.

        Therefore:

        obj1.setData(10);

        also changes obj2's data.

        Also, because both destructors try to delete
        the same memory, this design causes
        undefined behavior / possible double-delete.

        This is why shallow copy is dangerous when
        a class owns dynamically allocated memory.
    */


    // ==================================================
    // DEEP COPY EXAMPLE
    // ==================================================

    cout << "\n\n===== DEEP COPY =====" << endl;

    Deep deepObj1(42, 77);

    // Custom copy constructor is called
    Deep deepObj2 = deepObj1;


    cout << "Before changing deepObj2:" << endl;

    cout << "deepObj1: ";
    deepObj1.printData();

    cout << "deepObj2: ";
    deepObj2.printData();


    // Change deepObj2
    deepObj2.setData(99);


    cout << "\nAfter changing deepObj2 data to 99:" << endl;

    cout << "deepObj1: ";
    deepObj1.printData();

    cout << "deepObj2: ";
    deepObj2.printData();


    // ==================================================
    // COPY ASSIGNMENT OPERATOR EXAMPLE
    // ==================================================

    cout << "\n\n===== COPY ASSIGNMENT =====" << endl;

    Deep deepObj3(100, 200);

    cout << "Before assignment:" << endl;

    cout << "deepObj3: ";
    deepObj3.printData();


    // Copy assignment operator is called
    deepObj3 = deepObj1;


    cout << "\nAfter deepObj3 = deepObj1:" << endl;

    cout << "deepObj1: ";
    deepObj1.printData();

    cout << "deepObj3: ";
    deepObj3.printData();


    // ==================================================
    // SELF ASSIGNMENT
    // ==================================================

    cout << "\n\n===== SELF ASSIGNMENT =====" << endl;

    deepObj1 = deepObj1;

    cout << "deepObj1: ";
    deepObj1.printData();


    return 0;
}