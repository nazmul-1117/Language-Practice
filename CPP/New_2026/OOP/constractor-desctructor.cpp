
#include<iostream>
using namespace std;


class Hero{

    private:
        int health;
        char level;
        inline static int timeToPlay = 100;
    
    public:

        // constructor
        Hero(){
            cout << "Default Constructor is Created...\n";
        }

        Hero(int& health, char& level){
            cout << "Parameterized Constructor is Created...\n";
            cout << "this (address) -> " << this << endl;
            

            this -> health = health;
            this -> level = level;
        }

        ~Hero(){
            cout << "Default Destructor called...\n";
        }



        static int getRemineTime(){
            return timeToPlay;
        }



        void setHealth(int& h){
            health = h;
        }

        void setLevel(char& l){
            level = l;
        }

        int getHealth(){
            return health;
        }

        char getLevel(){
            return level;
        }
};



int main() {

    int health = 100;
    char level = 'A';

    // Default constructor
    Hero hero;

    cout << "=====================================================================================\n";
    cout << "Default Constructor" << endl;
    cout << "=====================================================================================\n";

    cout << "Default Health: " << hero.getHealth() << endl;

    cout << "-------------------------------------------------------------------------------------\n";

    // Parameterized constructor
    cout << "=====================================================================================\n";
    cout << "Parameterized Constructor" << endl;
    cout << "=====================================================================================\n";

    Hero* hero2 = new Hero(health, level);

    cout << "Default Object Address -> " << &hero << endl;
    cout << "Parameterized Object Address -> " << hero2 << endl;

    cout << "Parameterized Health: " << hero2->getHealth() << endl;

    cout << "-------------------------------------------------------------------------------------\n";

    Hero hero3(hero);
    
    cout << "Hero 3 Default Health: " << hero3.getHealth() << endl;
    cout << "Static Keywprd: " << Hero :: getRemineTime() << endl;

    delete hero2;

    return 0;
}
