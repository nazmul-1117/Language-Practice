#include<iostream>
using namespace std;

class Vehicle{

    // Encpasulation
    private:
        int speed;
        int* year;

    public:
        Vehicle(int s, int y)
            : speed(s), year(new int(y))
            {

            }

        ~Vehicle(){
            delete year;
        }

        void engineStart(){
            cout << "Vehicle Starting...: with parameters: "  << speed << endl;
        }
};

class Car: public Vehicle{

    private:
        int noOfDoor;
        int noOfSeat;

    public:
        Car(int d, int s, int sx, int yy)
            : noOfDoor(d), noOfSeat(s), Vehicle(sx, yy)
            {

            }

        void engineStart(){
            cout << "CAR Starting...: with parameters: " << noOfDoor << endl;
        }

};

int main(int argc, char const *argv[]) {
    
    Vehicle vehicle(1, 2);
    vehicle.engineStart();

    Car car(3, 4, 1, 2);
    car.engineStart();
    
    return 0;
}
