#include <arm_neon.h>
#include <cstdio>
#include <cstdint>

int main(){
    //Neon registers are 128 bit so if we want to make a variable we have to be it divisuble by 128
    /*Here were' going to make an int array that will be 16 bits
    we will also make sure of this by using the aligans(16) funciton which asks the complier to place them at an address that is 
    divisable by 16*/
    alignas(16) uint8_t a[16];
    alignas(16) uint8_t b[16];
    alignas(16) uint8_t c[16];

    //loading 16 byes at once is normal and natural for NEON 

    for(int i = 0; i < 16; i++){
        ///we're making an array with known values in this case 0-15 for a and 0 - 30 in 2's for b
        a[i] = (uint8_t)i;
        b[i] = (uint8_t)(2*i);

    }
    /*
    uint8x16_t is a NEON type that says a vector of 16 lanes with each lane being a unsigned 8 bit int
    this means that we're going to have a sort of array that will hold 8, 8 bit ints which are all going to be packed
    into one 128-bit regs like va = [a0, a1, ..... , a15]
    */
    uint8x16_t va = vld1q_u8(a);
    uint8x16_t vb = vld1q_u8(a);

    /*vadd = Vecotr add q means size 128 bit and u8 is operate on bytes
    so this does 16 additions in parrallel 
    for each lane k from (0 to 15)
    we do vc[k] = va[k] + vb[k]
    so the expected result lane by lane is going to be 
    c[i] = a[i] + b[i] = i * (2*i) = 3*i
    
    example: 
    i = 0 -> 0 + 0 = 0 
    i = 1 -> 1 + 2 = 3
    i = 2 -> 2 + 4 = 6 etc until we reach i = 16/k=15*/
    uint8x16_t vc = vaddq_u8(va, vb);

    /*vst1 -> vecotor store with the same q  to show that it will take 128 bytes, u8 the store bytes or the bytes of the variables we are going to be storing
    this writes all 16 lanes back into array c */
    vst1q_u8(c, vc);
    /* for loop to go though the 16 bytes and printes out the 
    c line or c array which is where we stored our added array from a + b*/
    for(int i = 0; i < 16; i++) std::printf("%d ", c[i]);
    std::printf("\n");

    /*Takeaway: instead of doing 16 seperate additions in a loop 
    with NEON we do one instruction that packs 16 values into a vecotor*/


    /*for a funciton like this is saying there are 3 interleaved structures per elemetnt 
    so we load the interleaved data where each element has 3 parts
    each part gets its own vector with each vecor beign 128 bits = 16 bytes
    it would return something like
    
    struct{
        uint8x16_t val[3];
    }
        
    this then becomes:

    val[0] = all B values 
    val[1] = all G values
    val[2] = all R values
    */
    vld3q_u8(---);

}