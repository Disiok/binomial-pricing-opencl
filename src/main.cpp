#include <iostream>
#include "option_spec.h"
#include "pricer.h"

int main() {
    std::cout << "[INFO] Starting tester main function." << std::endl;
    // Construct test option specification
    OptionSpec optionSpec = {-1, 100, 100, 1.0, 0.3, 0.02, 500, true};
    OptionPricer* pricer = new SerialPricer();
    double optionValue = pricer->price(optionSpec);
    std::cout << "The value of the option is: " << optionValue << std::endl;
    std::cout << "[INFO] Terminating tester main function." << std::endl;
}
