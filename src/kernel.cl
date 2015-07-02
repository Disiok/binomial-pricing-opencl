__kernel void
init(
     const float stockPrice,
     const float strikePrice,
     const float yearsToMaturity,
     const float volatility,
     const float riskFreeRate,
     const int numSteps,
     const int type,
     __global float* valueAtExpiry
     )
{
    size_t id = get_global_id(0);

    // -------------------------Derived Parameters-----------------------------
    float deltaT = yearsToMaturity / numSteps;
    float upFactor = exp(volatility * sqrt(deltaT));
    float downFactor = 1.0f / upFactor;

    // ---------------------Calculate option value at expiry-------------------
    float stockPriceAtExpiry = stockPrice * pow(upFactor, id) *
                                            pow(downFactor, numSteps - id);
    valueAtExpiry[id] = max(type * (stockPriceAtExpiry - strikePrice), 0.0f); 
}

__kernel void
iterate(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __constant const float* optionValueIn,
        __global float* optionValueOut
        )
{
    size_t id = get_global_id(0);
    optionValueOut[id] = (upWeight * optionValueIn[id] +
                         downWeight * optionValueIn[id + 1]) / discountFactor;
    
}
