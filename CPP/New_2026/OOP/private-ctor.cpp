#include <iostream>
using namespace std;

class Student {
private:
    string name;
    int age;

    // Private Constructor
    Student(string n, int a) {
        name = n;
        age = a;
    }

    // Student createStudent(string name, int age);

public:
    // Public static function to create an object
    // static Student createStudent(string n, int a) {
    //     return Student(n, a);
    // }

    // Display student information
    void display() {
        cout << "Name: " << name << endl;
        cout << "Age: " << age << endl;
    }

    // Friend keyword
    friend class GUBStudent;
    friend Student& createStudent(string name, int age);
};

class GUBStudent{
private:
    Student *student;
public:
    GUBStudent()
        : student(new Student("Md. Nazmul Hosssain", 25)) {

        };

    ~GUBStudent(){
        delete student;
    }

    void printGubStudent(){
        cout << "Name: " << student -> name << endl
        
             << "Age: " << student->age << endl;
    }
};

Student& createStudent(string name, int age){
    Student *student = new Student(name, age);

    return *student;
}

int main() {

    // Student s1("Rahim", 22);  
    // ❌ Error: Constructor is private

    // Object is created through public static function
    // Student s1 = Student::createStudent("Rahim", 22);

    GUBStudent *nazmul = new GUBStudent();
    nazmul->printGubStudent();

    Student shohag = createStudent("Shohag", 26);
    shohag.display();


    return 0;
}