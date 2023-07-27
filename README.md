**PyU8** is an nX-U8/100 core emulator written in Python. It only comes in the form of one Python script, `u8.py`, and is meant to be used as a module.  
It only emulates the CPU and not any microcontroller, so a seperate script (that imports `u8.py`) would need to be used as the microcontroller emulator.

This emulator is **very bare-bones** and currently doesn't even have some kind of entry point, so good luck getting it to work.
It also currently supports a very low amount of CPU instructions.
