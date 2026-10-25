#include <iostream>
#include <map>

using namespace std;

int main(int argc, char const *argv[]) {

    // ============================================================
    // Basic Configuration
    // ============================================================

    // Number of '-' characters used as a separator
    int nDot = 50;


    // ============================================================
    // Initialize Map
    // ============================================================

    // Create a map with:
    //
    // Key   -> string
    // Value -> string
    //
    // map automatically stores elements
    // in ascending order of their keys.
    map<string, string> countries;


    // ============================================================
    // Insert Values - 3 Ways
    // ============================================================

    // ------------------------------------------------------------
    // Way 1: Using [] operator
    // ------------------------------------------------------------

    // Insert key-value pairs
    countries["bd"] = "Bangladesh";
    countries["np"] = "Nepal";
    countries["ch"] = "United Kingdom";


    // ------------------------------------------------------------
    // Way 2: Using insert() and make_pair()
    // ------------------------------------------------------------

    // make_pair() creates a pair containing key and value
    countries.insert(make_pair("pk", "Pakistan"));


    // ------------------------------------------------------------
    // Way 3: Using pair
    // ------------------------------------------------------------

    // Create a pair of strings
    pair<string, string> p;

    // first  -> key
    // second -> value
    p.first = "in";
    p.second = "India";

    // Insert the pair into the map
    countries.insert(p);


    // ============================================================
    // Modify Value Using at()
    // ============================================================

    // at() accesses the value associated with a key
    //
    // "ch" originally contains "United Kingdom"
    // It will now contain "China"
    countries.at("ch") = "China";


    // ============================================================
    // Print Data
    // ============================================================

    // Declare an iterator for the map
    map<string, string>::iterator it;

    // begin() returns an iterator to the first element
    it = countries.begin();


    // Print the number of key-value pairs
    cout << "Map Size: " << countries.size() << endl;


    // Traverse through the complete map
    while (it != countries.end()) {

        // it->first  = Key
        // it->second = Value
        cout << it->first << " : " << it->second << "\n";

        // Move iterator to the next element
        it++;
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Find
    // ============================================================

    // find() searches for a specific key
    //
    // If the key exists:
    //     iterator points to the element
    //
    // If the key does not exist:
    //     iterator becomes countries.end()
    it = countries.find("bd");


    if (it != countries.end()) {

        cout << "Data found, "
             << it->first << " : "
             << it->second << endl;

    } else {

        cout << "Data not found\n";
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Count
    // ============================================================

    // count() checks whether a key exists.
    //
    // For map:
    //     1 -> key exists
    //     0 -> key does not exist
    //
    // A map cannot contain duplicate keys.
    if (countries.count("bd") == 1) {

        cout << "Key Found" << endl;

    } else {

        cout << "Key Not Found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Clear
    // ============================================================

    // clear() removes all key-value pairs
    countries.clear();


    // Check whether the map is empty
    if (countries.empty()) {

        cout << "Map is empty" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}