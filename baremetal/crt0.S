.section .text.init
.globl _start
.type _start, @function

_start:
    la sp, _stack_top

    call main

    # main return value:
    #   0 -> PASS
    #   non-zero -> FAIL
    #
    # HTIF convention:
    #   tohost = (exit_code << 1) | 1

    slli a0, a0, 1
    ori  a0, a0, 1

    la t0, tohost
    sd a0, 0(t0)

1:
    j 1b


.section .tohost, "aw", @progbits
.balign 64

.globl tohost
tohost:
    .dword 0

.globl fromhost
fromhost:
    .dword 0
