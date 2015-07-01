#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include "option_spec.h"

double priceLattice(OptionSpec& optionSpec) {
    // ------------------------Derived Parameters------------------------------
    double deltaT = optionSpec.yearsToMaturity / optionSpec.numSteps;

    double upFactor = exp(optionSpec.volatility * sqrt(deltaT));
    double downFactor = 1.0 / upFactor;

    double discountFactor = exp(optionSpec.riskFreeRate * deltaT);

    double upWeight = (discountFactor - downFactor) / (upFactor - downFactor);
    double downWeight = 1.0 - upWeight;

    // Calculate option value at expiry
    std::vector<double> valueAtExpiry(optionSpec.numSteps + 1);
    for (int i = 0; i <= optionSpec.numSteps; ++i) {
        double stockPriceAtExpiry = optionSpec.stockPrice * 
                                   pow(upFactor, i) *
                                   pow(downFactor, optionSpec.numSteps - i);
        valueAtExpiry[i] = std::max(optionSpec.type * 
                                (stockPriceAtExpiry - optionSpec.strikePrice),
                                0.0);
    }
    
    for (int i = optionSpec.numSteps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; j++) {
            valueAtExpiry[j] = (downWeight * valueAtExpiry[j] +
                                upWeight * valueAtExpiry[j + 1]) 
                                / discountFactor; 
            if (optionSpec.isAmerican) {
                double stockPrice = optionSpec.stockPrice *
                                    pow(upFactor, j) *
                                    pow(downFactor, i - j);
                valueAtExpiry[j] = std::max(valueAtExpiry[j], std::max(0.0, 
                    optionSpec.type * (stockPrice - optionSpec.strikePrice)));
            }
        }    
    }

    return valueAtExpiry[0];
}
int main() {
    std::cout << "[INFO] Starting main function" << std::endl;
    OptionSpec optionSpec = {-1, 100, 100, 1.0, 0.3, 0.02, 500, true};
    double price = priceLattice(optionSpec);
    std::cout << "Final price is: " << price << std::endl;
}


