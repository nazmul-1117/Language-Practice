
#include<iostream>
using namespace std;


class Hero{

    private:
        const int health;
        const char level;
        int damage;
    
    public:

        // constructor
        Hero()
            : health(-1), level('Z'), damage(101)
        {
            cout << "Hero Default Constructor is Created...\n";
        }


        // initialization list
        Hero(int& health, char& level, int& damage)
            : health(health), level(level), damage(damage)
        {
            cout << "Hero Parameterized Constructor is Created...\n";
        }


        int getHealth(){
            return health;
        }

        char getLevel(){
            return level;
        }

        int getDamage(){
            return damage;
        }

};

class Travor{
    private:
        Hero hero;

    public:
        Travor(int& health, char& level, int& damage)
            : hero(health, level, damage)
        {
            cout << "Travor Parameterized Constructor is Created...\n";
        }

        Hero getHero(){
            return hero;
        }
};



int main() {

    int health = 100;
    int damage = 0;
    char level = 'A';

    // Default constructor
    Hero hero(health, level, damage);

    cout << "=====================================================================================\n";
    cout << "Parameterized Constructor" << endl;
    cout << "=====================================================================================\n";

    cout << "Health " << hero.getHealth() << endl;
    cout << "Level " << hero.getLevel() << endl;
    cout << "Damage " << hero.getDamage() << endl;

    cout << "-------------------------------------------------------------------------------------\n";

    Hero hero2;
    cout << "\n\n=====================================================================================\n";
    cout << "Default Constructor" << endl;
    cout << "=====================================================================================\n";

    cout << "Health " << hero2.getHealth() << endl;
    cout << "Level " << hero2.getLevel() << endl;
    cout << "Damage " << hero2.getDamage() << endl;
    cout << "-------------------------------------------------------------------------------------\n";

    health=70, level='C', damage=30;
    Travor travor(health, level, damage);
    cout << "\n\n=====================================================================================\n";
    cout << "Travor Constructor" << endl;
    cout << "=====================================================================================\n";

    cout << "Travor-Health " << travor.getHero().getHealth() << endl;
    cout << "Travor-Level " << travor.getHero().getLevel() << endl;
    cout << "Travor-Damage " << travor.getHero().getDamage() << endl;
    cout << "-------------------------------------------------------------------------------------\n";


    return 0;
}
