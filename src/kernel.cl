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
    // printf("[TRACE] valueAtExpiry[%d] = %f\n", id, valueAtExpiry[id]);
}

__kernel void
group(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        const int currentNumLattice,
        const int groupSize
        )
{
    int startIndex = get_global_id(0) * groupSize;
    int endIndex = min(startIndex + groupSize, currentNumLattice);

    for (int i = startIndex; i < endIndex; i++) {
        optionValueOut[i] = (downWeight * optionValueIn[i] + 
                               upWeight * optionValueIn[i + 1])
                               / discountFactor;
    }
}

__kernel  
void upTriangle(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValue,
        __local float* tempOptionValue,
        __global float* triangle
        )
{
    size_t localId = get_local_id(0);
    size_t groupSize = get_local_size(0);
    size_t groupId = get_group_id(0);
    size_t stepSize = groupSize - 1;

    tempOptionValue[localId] = optionValue[stepSize * groupId + localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = stepSize - 1 ; i >= 0; i --) {
        float value;
        if (localId <= i) {
            value = (downWeight * tempOptionValue[localId] +
                    upWeight * tempOptionValue[localId + 1])
                    / discountFactor;
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId <= i) {
            tempOptionValue[localId] = value;
        }
        // Decide whether to update vertex value: "&&localId != 0"
        if (localId == i) {
           optionValue[groupId * stepSize + localId] = value;
           // printf("Updating optionValue[%d] with value = %f\n", 
           //         groupId * stepSize + localId, value);
        }
        if (localId == 0) {
            triangle[groupId * stepSize + stepSize - i] = value; 
            // printf("Updating triange[%d] with value from localId = %d, i = %d, groupId = %d\n", 
            //         stepSize - i, localId, i, groupId);
        }
    }
}
__kernel  
void downTriangle(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValue,
        __local float* tempOptionValue,
        __global float* triangle
        )
{
    size_t localId = get_local_id(0);
    size_t groupSize = get_local_size(0);
    size_t groupId = get_group_id(0);
    size_t stepSize = groupSize - 1;

    for (int i = stepSize - 1 ; i >= 1; i --) {
        float value;
        float upValue;
        float downValue;

        if (localId == stepSize - 1) {
            upValue = triangle[groupId * groupSize + stepSize - i];
            // printf("Updating with triangle[%d],", stepSize - i);
        } else {
            upValue = tempOptionValue[localId + 1];
            // printf("Updating with temp[%d],", localId + 1);
        }

        if (localId == stepSize - i) {
            downValue = optionValue[stepSize * groupId + localId];
            // printf("optionValue[%d], ", localId);
        } else {
            downValue = tempOptionValue[localId];
            // printf("temp[%d], ", localId);
        }

        if (localId >= i && localId != stepSize) {
            value = (downWeight * downValue+
                    upWeight * upValue)
                    / discountFactor;
            // printf("to temp[%d]\n", localId);
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        if (localId >= i && localId != stepSize) {
            tempOptionValue[localId] = value;
        }
    }
    if (localId != 0 && localId != stepSize) {
        optionValue[groupId * stepSize + localId] = tempOptionValue[localId]; 
    }
}
