//distance between two point using class, friend, constructor;
//s = sqrt((x2-x1)^2 + (y2-y1)^2);

#include <bits/stdc++.h>
using namespace std;

class point{
  int x;
  int y;
  
  public:
    point(int a, int b);
    void displayData(void);
    friend void distanceData(point , point , double*);
};

void distanceData(point o11, point o21, double *s){
    
    double p;
    *s = sqrt(pow((float)(o21.x-o11.x), 2) + pow((float)(o21.y-o11.y), 2));
}

point :: point(int a, int b){   //constructor
    x = a;
    y = b;
}

void point :: displayData(void){
    printf("Point (%d, %d).\n", x, y);
}

int main()
{
    point o1(2, 3);
    point o2 = point(4, 6);
    
    o1.displayData();
    o2.displayData();
    
    double distance;
    distanceData(o1, o2, &distance);
    
    cout<<"Distance is: ";
    cout<<fixed<<setprecision(2)<<distance<<endl;
    
    return 0;
}
