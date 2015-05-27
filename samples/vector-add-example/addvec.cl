__kernel void 
simple_add(
           __constant const int* a, 
           __constant const int* b, 
           __global int* c) 
{
   c[get_global_id(0)] = a[get_global_id(0)] + b[get_global_id(0)];
}
