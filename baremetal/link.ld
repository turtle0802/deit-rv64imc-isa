ENTRY(_start)

SECTIONS
{
    . = 0x80000000;

    .text :
    {
        KEEP(*(.text.init))
        *(.text*)
    }

    . = ALIGN(16);

    .rodata :
    {
        *(.rodata*)
    }

    . = ALIGN(16);

    .data :
    {
        *(.data*)
        *(.sdata*)
    }

    . = ALIGN(16);

    .bss :
    {
        *(.bss*)
        *(.sbss*)
        *(COMMON)
    }

    /*
     * HTIF communication area.
     */
    . = ALIGN(64);

    .tohost :
    {
        KEEP(*(.tohost))
    }

    /*
     * Stack is deliberately placed AFTER all program/model data.
     *
     * 64 KiB is more than enough for the current bare-metal
     * verification and benchmark programs.
     */
    . = ALIGN(16);

    _stack_bottom = .;

    . += 0x10000;

    _stack_top = .;
}
