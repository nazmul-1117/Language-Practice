
#include<iostream>
using namespace std;


class Hero{

};

class Hero2{
    int health;
    char level;

};

class Hero3 {
    // access modifier
    private:
        int health;
        char level;
    
    public:
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


int main(int argc, char const *argv[]) {

    Hero hero1;
    Hero2 hero2;
    
    cout << "=====================================================================================\n";
    cout << "Print Class Memory allocation Size" << endl;
    cout << "=====================================================================================\n";
    cout << "Empty-Hero1-Size " << sizeof(hero1) << endl;
    cout << "Empty-Hero2-Size " << sizeof(hero2) << endl;
    cout << "-------------------------------------------------------------------------------------\n";

    cout << "=====================================================================================\n";
    cout << "Print Class Access Modifier" << endl;
    cout << "=====================================================================================\n";
    
    Hero3 hero3;
    int health = 100;
    char level = 'A';

    hero3.setHealth(health);
    hero3.setLevel(level);

    cout << "Hero3 Health " << hero3.getHealth() << endl;
    cout << "Hero3 Level " << hero3.getLevel() << endl;

    cout << "-------------------------------------------------------------------------------------\n";



    cout << "=====================================================================================\n";
    cout << "Print Dynamic Memory Allocation and Deallocation" << endl;
    cout << "=====================================================================================\n";
    
    Hero3* hero4 = new Hero3;
    health = 90;
    level = 'B';

    (*hero4).setHealth(health);
    hero4 -> setLevel(level);

    cout << "Hero4 Health " << hero4 -> getHealth() << endl;
    cout << "Hero4 Level " << hero4 -> getLevel() << endl;

    cout << "-------------------------------------------------------------------------------------\n";

    return 0;
}
