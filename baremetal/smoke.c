int main(void)
{
    volatile int a = 10;
    volatile int b = 32;

    int result = a + b;

    if (result == 42) {
        return 0;
    }

    return 1;
}
