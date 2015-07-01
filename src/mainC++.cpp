#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "option_spec.h"

float priceLattice(OptionSpec& optionSpec) {
    // ------------------------Derived Parameters------------------------------
    float deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    float upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    float downFactor = 1 / upFactor;

    float discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    float upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    float downWeight = 1 - upWeight;

    // Calculate option value at expiry
    std::vector<float> valueAtExpiry(optionSpec.numSteps);
    for (int i = 0; i < optionSpec.numSteps; ++i) {
        float stockPriceAtExpiry = optionSpec.stockPrice * 
                                   pow(upFactor, 2 * i - optionSpec.numSteps);
        valueAtExpiry[i] = std::max(optionSpec.type * 
                                (stockPriceAtExpiry - optionSpec.strikePrice),
                                0.0f);
    }
    
    for (int i = optionSpec.numSteps - 1; i >= 0; --i) {
        for (int j = 0; j < i; j++) {
            valueAtExpiry[j] = (downWeight * valueAtExpiry[j] +
                                upWeight * valueAtExpiry[j + 1]) 
                                / discountFactor; 
        }    
    }

    return valueAtExpiry[0];
}
int main() {
    std::cout << "[INFO] Starting main function" << std::endl;
    OptionSpec optionSpec = {-1, 100, 100, 1.0, 0.3, 0.02, 5000};
    float price = priceLattice(optionSpec);
    std::cout << "Final price is: " << price << std::endl;
}


