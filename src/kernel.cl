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

/*
__kernel void
iterate(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        __local float* tempOptionValue
        )
{
    size_t localId = get_local_id(0);
    size_t groupSize = get_local_size(0);
    size_t groupId = get_group_id(0);
    size_t stepSize = groupSize - 1;

    tempOptionValue[localId] = optionValueIn[groupId + localId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = stepSize - 1 ; i >= 0; i --) {
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
        optionValueOut[groupId] = tempOptionValue[0];
        // printf("[TRACE] groupId = %d, stepSize = %d, optionValueOut[%d] = %f\n", groupId, stepSize, groupId, optionValueOut[groupId]);
    }
}
*/

__kernel void
group(
        const float upWeight,
        const float downWeight,
        const float discountFactor,
        __global float* optionValueIn,
        __global float* optionValueOut,
        const int currentNumLattice,
        const int stepSize
        )
{
    size_t id = get_global_id(0) * stepSize;

    for (int i = 0; i < stepSize; i++) {
        if (id + i < currentNumLattice) {
            optionValueOut[id + i] = (downWeight * optionValueIn[id + i] + 
                                   upWeight * optionValueIn[id + i + 1])
                                   / discountFactor;
        }
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
        float value = tempOptionValue[localId];
        if (localId <= i) {
            value = (downWeight * tempOptionValue[localId] +
                    upWeight * tempOptionValue[localId + 1])
                    / discountFactor;
        } 
        barrier(CLK_LOCAL_MEM_FENCE);
        tempOptionValue[localId] = value;
        if (localId == i && localId != 0) {
           optionValue[groupId * stepSize + localId] = value;
        }
        if (localId == 0) {
            triangle[stepSize - i] = value; 
            // printf("Updating triange[%d] with value from localId = %d, i = %d, groupId = %d\n", stepSize - i, localId, i,groupId);
        }
    }
    
    printf("[TRACE] groupId = %d, stepSize = %d, localId = %d\n", groupId, stepSize, localId);
    if (localId == 0) {
        optionValue[groupId * stepSize] = tempOptionValue[localId];
        printf("[TRACE] groupId = %d, stepSize = %d, optionValue[%d] = %f\n", groupId, stepSize, groupId * stepSize, optionValue[groupId * stepSize]);
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
        float value = tempOptionValue[localId];
        float upValue;
        float downValue;

        if (localId == stepSize - 1) {
            upValue = triangle[stepSize - i];
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
        tempOptionValue[localId] = value;
    }
    if (localId != 0 && localId != stepSize) {
        optionValue[groupId * stepSize + localId] = tempOptionValue[localId]; 
    }
}
