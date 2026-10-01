#include <iostream>
#include <string>

namespace University {

    struct Student {
        int id;
        std::string name;
        int age;
        float cgpa;
    };

}

void addStudent(University::Student& student);
void showAllStudent(University::Student& student);
void displayMainMenu();
bool mainMenu(University::Student& student);


void addStudent(University::Student& student) {

    std::cout << "Enter ID: ";
    std::cin >> student.id;

    std::cout << "Enter Name: ";
    std::getline(std::cin >> std::ws, student.name);

    std::cout << "Enter Age: ";
    std::cin >> student.age;

    std::cout << "Enter CGPA: ";
    std::cin >> student.cgpa;

    std::cout << "\nStudent added successfully!\n";
}


void showAllStudent(University::Student& student) {

    std::cout << "\n========== Student Information ==========\n";

    std::cout << "ID: " << student.id << std::endl;
    std::cout << "Name: " << student.name << std::endl;
    std::cout << "Age: " << student.age << std::endl;
    std::cout << "CGPA: " << student.cgpa << std::endl;

    std::cout << "=========================================\n";
}


void displayMainMenu() {

    std::cout << "\n========================================\n";
    std::cout << "\tSTUDENT MANAGEMENT SYSTEM\n";
    std::cout << "========================================\n";

    std::cout << "1. Add Student\n";
    std::cout << "2. Show All Students\n";
    std::cout << "3. Search Student\n";
    std::cout << "4. Update Student\n";
    std::cout << "5. Delete Student\n";
    std::cout << "6. Show Statistics\n";
    std::cout << "7. Exit\n";

    std::cout << "========================================\n";
}


bool mainMenu(University::Student& student) {

    int choice;

    displayMainMenu();

    std::cout << "Enter Your Choice: ";
    std::cin >> choice;

    switch (choice) {

        case 1:
            addStudent(student);
            break;

        case 2:
            showAllStudent(student);
            break;

        case 7:
            std::cout << "Exiting program...\n";
            return false;

        default:
            std::cout << "Invalid choice!\n";
            break;
    }

    return true;
}


int main() {

    University::Student student;

    while (mainMenu(student)) {
        // Keep showing menu
    }

    return 0;
}