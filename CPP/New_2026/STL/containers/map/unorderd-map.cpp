#include <iostream>

// #include <map>

#include <unordered_map>

using namespace std;

int main(int argc, char const *argv[]) {

    // ============================================================
    // Basic Configuration
    // ============================================================

    // Number of '-' characters used as a separator
    int nDot = 50;


    // ============================================================
    // Initialize Unordered Map
    // ============================================================

    // Create an unordered_map with:
    //
    // Key   -> string
    // Value -> string
    //
    // Example:
    // "bd" -> "Bangladesh"
    //
    // unordered_map stores data as KEY-VALUE pairs.
    unordered_map<string, string> countries;


    // ============================================================
    // Insert Values - 3 Ways
    // ============================================================

    // ------------------------------------------------------------
    // Way 1: Using [] operator
    // ------------------------------------------------------------

    // Insert key-value pairs using [] operator
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

    // Create an empty pair
    pair<string, string> p;

    // Assign key to first
    p.first = "in";

    // Assign value to second
    p.second = "India";

    // Insert the pair into the unordered_map
    countries.insert(p);


    // ============================================================
    // Modify Value Using at()
    // ============================================================

    // at() accesses the value associated with a key
    //
    // Here:
    // "ch" originally contains "United Kingdom"
    //
    // It will be changed to "China"
    countries.at("ch") = "China";


    // ============================================================
    // Print Data
    // ============================================================

    // Declare an iterator for unordered_map
    unordered_map<string, string>::iterator it;

    // begin() returns an iterator to the first element
    //
    // IMPORTANT:
    // unordered_map does NOT maintain sorted order.
    // Therefore, the printing order is not guaranteed.
    it = countries.begin();


    // Print number of key-value pairs
    cout << "Map Size: " << countries.size() << endl;


    // Traverse the complete unordered_map
    while (it != countries.end()) {

        // it->first  = Key
        // it->second = Value
        cout << it->first << " : " << it->second << "\n";

        // Move iterator to next element
        it++;
    }

    cout << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Find
    // ============================================================

    // find() searches for a key
    //
    // If the key exists:
    //     iterator points to that element
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
    // For unordered_map:
    //     1 -> key exists
    //     0 -> key does not exist
    //
    // Each key can exist only once.
    if (countries.count("bd") == 1) {

        cout << "Key Found" << endl;

    } else {

        cout << "Key Not Found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Access Value Using [] Operator
    // ============================================================

    // [] can be used to access the value using a key
    cout << "Value of bd: "
         << countries["bd"] << endl;

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Check Another Key
    // ============================================================

    if (countries.find("us") != countries.end()) {

        cout << "US Found" << endl;

    } else {

        cout << "US Not Found" << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Erase
    // ============================================================

    // erase() removes a key-value pair using the key
    countries.erase("np");

    cout << "After erasing np:" << endl;

    for(auto element : countries) {

        cout << element.first
             << " : "
             << element.second << endl;
    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // Clear
    // ============================================================

    // clear() removes all key-value pairs
    countries.clear();


    // empty() checks whether the unordered_map is empty
    if (countries.empty()) {

        cout << "Map is empty" << endl;

    }

    cout << string(nDot, '-') << endl << endl;


    // ============================================================
    // End of Program
    // ============================================================

    return 0;
}