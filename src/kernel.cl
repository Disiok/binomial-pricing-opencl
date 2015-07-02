__kernel void
init(
     const float stockPrice,
     const float strikePrice,
     const int numSteps,
     const int type,
     const float deltaT,
     const float upFactor,
     const float downFactor,
     __global float* valueAtExpiry
     )
{
    // ---------------------Calculate option value at expiry-------------------
    size_t id = get_global_id(0);
    float stockPriceAtExpiry = stockPrice * pow(upFactor, id) *
                                            pow(downFactor, numSteps - id);
    valueAtExpiry[id] = max(type * (stockPriceAtExpiry - strikePrice), 0.0f); 
}

__kernel void
iterate(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValue,
        __local float* tempOptionValue
        )
{
    size_t localId = get_local_id(0);
    size_t groupId = get_group_id(0);

    tempOptionValue[localId] = optionValue[groupId + localId];
    printf("Group %d, local %d, value %f \n", groupId, localId, optionValue[groupId + localId]);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 512 - 2 ; i >= 0; i --) {
        float value = tempOptionValue[localId];
        if (localId <= i) {
            value = (downWeight * tempOptionValue[localId] +
                    upWeight * tempOptionValue[localId + 1])
                    / discountFactor;
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        tempOptionValue[localId] = value;
    }
    if (localId == 0) {
        optionValue[groupId] = tempOptionValue[localId];
        printf("Last iteration with local %d, value %f \n", localId,optionValue[groupId]);
    }
}
