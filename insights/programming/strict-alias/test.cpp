#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

// ~/repos/mind-lab/insights/programming/strict-alias (main)gcc test.cpp -O2 && ./a.out 
// GREEN_LED, RED_LED: (0, 0) -> (1, 0)
// RED_LED, RED_LED: (0, 0) -> (1, 0)
// ~/repos/mind-lab/insights/programming/strict-alias (main)gcc test.cpp -O1 && ./a.out 
// GREEN_LED, RED_LED: (0, 0) -> (1, 0)
// RED_LED, RED_LED: (0, 0) -> (0, 0)

struct Status {
    unsigned left;
    unsigned right;
};

struct LED_CTRL_bits {
    uint32_t LED_ON  : 1;
    uint32_t LED_PWM : 3;
};

union LED_CTRL {
    struct LED_CTRL_bits bits;
    uint32_t value;
};

union LED2_CTRL {
    struct LED_CTRL_bits bits;
    uint32_t value;
};

typedef union LED_CTRL LED_CTRL_t;
typedef union LED2_CTRL LED2_CTRL_t;

template<typename T1, typename T2>struct Status Get(T1 *left, T2 *right)
{
    return { .left = left->bits.LED_ON, .right = right->bits.LED_ON };
}

template<typename T1, typename T2>
struct Status SetLClearR(T1 *left, T2 *right)
{
    left->bits.LED_ON = 1;
    left->bits.LED_PWM = 3;
    // usleep(1000);
    right->value = 0;
    return { .left = left->bits.LED_ON,
             .right = right->bits.LED_ON };
}


static uint8_t GREEN_LED[sizeof(LED_CTRL_t)];
static uint8_t RED_LED[sizeof(LED2_CTRL_t)];

int main()
{
    {
        printf("GREEN, RED: ");
        Status before = Get((LED_CTRL_t*)&GREEN_LED, (LED2_CTRL_t*)&RED_LED);
        Status after = SetLClearR((LED_CTRL_t*)&GREEN_LED, (LED2_CTRL_t*)&RED_LED);
        printf("(%d, %d) -> (%d, %d)\n", before.left, before.right, after.left, after.right);
    }
    {
        printf("RED, RED: ");
        Status before = Get((LED_CTRL_t*)&RED_LED, (LED2_CTRL_t*)&RED_LED);
        Status after = SetLClearR((LED_CTRL_t*)&RED_LED, (LED2_CTRL_t*)&RED_LED);
        printf("(%d, %d) -> (%d, %d)\n", before.left, before.right, after.left, after.right);
    }
    return 0;
}