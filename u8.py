import sys
if __name__ == '__main__':
	print('Please use this script as a module.')
	sys.exit()

import ctypes
import logging
from functools import lru_cache

class CPUError(Exception): pass

class u_nibble(ctypes.Structure):
	_fields_ = [('value', ctypes.c_ubyte, 4)]

	def __init__(self, value: int = 0): self.value = value & 0xf
	def __repr__(self): return f'nibble({self.value})'
	def __str__(self): return str(self.value)

class imm_7(ctypes.Structure):
	_fields_ = [('value', ctypes.c_byte, 7)]

	def __init__(self, value: int = 0): self.value = value & 0x7f
	def __repr__(self): return f'imm_7({self.value})'
	def __str__(self): return str(self.value)

class U8:
	def __init__(self, rom: bytes, ram_size: int, rom_window_size: int):
		if len(rom) < 0xFF: raise CPUError('ROM must be at least 127 words (0xFF bytes)')
		elif len(rom) % 2 != 0: raise CPUError(f'ROM is of ODD length ({hex(len(self.rom))})')

		# https://stackoverflow.com/a/14543975
		self.rom_w = bytes([c for t in zip(rom[1::2], rom[::2]) for c in t])
		self.rom = rom

		self.rom_window_size = rom_window_size

		self.reset = False

		self.reset_el_2_ptr = ctypes.c_ushort(self.concat_bytes(self.rom_w[2:4]))
		self.reset_el_0_ptr = ctypes.c_ushort(self.concat_bytes(self.rom_w[4:6]))

		self.nmice_ptr = ctypes.c_ushort(self.concat_bytes(self.rom_w[6:8]))
		self.hwi_ptrs = [ctypes.c_ushort(self.concat_bytes(self.rom_w[i:i+1])) for i in range(0xa, 0x7f, 2)]
		self.swi_ptrs = [ctypes.c_ushort(self.concat_bytes(self.rom_w[i:i+1])) for i in range(0x80, 0xff, 2)]

		self.ram = b'\x00'*ram_size

		self.sp = ctypes.c_ushort()
		self.r = [ctypes.c_ubyte()] * 16
		self.cr = [ctypes.c_ubyte()] * 16
		self.psw = ctypes.c_ubyte()
		self.pc = ctypes.c_ushort()
		self.csr = u_nibble()
		self.lr = ctypes.c_ushort()
		self.elr1 = self.elr2 = self.elr3 = ctypes.c_ushort()
		self.lcsr = u_nibble()
		self.ecsr1 = self.ecsr2 = self.ecsr3 = u_nibble()
		self.epsw1 = self.epsw2 = self.epsw3 = ctypes.c_ubyte()
		self.ea = ctypes.c_ushort()
		self.ar = ctypes.c_ushort()
		self.dsr = ctypes.c_ubyte()

	def run(self):
		self.reset_registers()
		self.loop()

	def loop(self):
		while True:
			self.exec_instruction()
			if self.reset: break

	def reset_registers(self):
		self.sp.value = self.concat_bytes(self.rom[0:2])
		for reg in self.r: reg.value = 0
		for reg in self.cr: reg.value = 0
		self.psw.value = 0
		self.pc.value = self.addr_filter(self.reset_el_0_ptr.value)
		self.csr.value = 0
		self.lr.value = 0
		self.elr1.value = self.elr2.value = self.elr2.value = 0
		self.lcsr.value = 0
		self.ecsr1.value = self.ecsr2.value = self.ecsr3.value = 0
		self.epsw1.value = self.epsw2.value = self.epsw3.value = 0
		self.ea.value = 0
		self.ar.value = 0
		self.dsr.value = 0

	@staticmethod
	@lru_cache
	def addr_filter(addr: int) -> int:
		if addr % 2 == 0: return addr
		else: return addr | 1

	def read_mem(self, addr: int) -> int:
		if addr < self.rom_window_size: return self.rom[addr]
		elif addr < 0x8e00: return self.ram[addr - self.rom_window_size]
		elif addr < 0xf000: return 0
		elif addr < 0x10000: raise CPUError('SFRs not implemented')
		else: return self.rom[addr]

	def er(self, n: int) -> int: return self.concat_bytes(self.r[n+1].value, self.r[n].value)
	def xr(self, n: int) -> int: return self.concat_bytes(self.er(n), self.er(n+1))
	def qr(self, n: int) -> int: return self.concat_bytes(self.xr(n), self.xr(n+1))
	def ecr(self, n: int) -> int: return self.concat_bytes(self.cr[n+1].value, self.cr[n].value)
	def xcr(self, n: int) -> int: return self.concat_bytes(self.ecr(n), self.ecr(n+1))
	def qcr(self, n: int) -> int: return self.concat_bytes(self.xcr(n), self.xcr(n+1))

	# https://stackoverflow.com/a/20024864
	@staticmethod
	@lru_cache
	def split_bytes(data: bytes) -> list: return [data[i:i+2] for i in range(0, len(data), 2)]

	def set_er(self, n: int, val: int):
		byt = val.to_bytes(2, 'little')
		self.r[n].value = byt[0]; self.r[n + 1].value = byt[1]

	def set_xr(self, n: int, val: int):
		val_s = self.split_bytes(val)
		self.set_er(n, val_s[0]); self.set_er(n + 1, val_s[1])

	def set_qr(self, n: int, val: int):
		val_s = self.split_bytes(val)
		self.set_xr(n, val_s[0]); self.set_er(n + 1, val_s[1])

	def set_ecr(self, n: int, val: int):
		byt = val.to_bytes(2, 'little')
		self.cr[n].value = byt[0]; self.cr[n + 1].value = byt[1]

	def set_xcr(self, n: int, val: int):
		val_s = self.split_bytes(val)
		self.set_ecr(n, val_s[0]); self.set_ecr(n + 1, val_s[1])

	def set_qcr(self, n: int, val: int):
		val_s = self.split_bytes(val)
		self.set_xcr(n, val_s[0]); self.set_xcr(n + 1, val_s[1])

	@staticmethod
	@lru_cache
	def get_bits(byte: int, num_bits: int, start_bit: int = 0): return (byte >> start_bit) & ((1 << num_bits) - 1)

	@staticmethod
	@lru_cache
	def concat_bytes(*args: bytes) -> int:
		if len(args) == 1: return int('0x' + ''.join(hex(_)[2:] for _ in args[0]), 16)
		else: return int('0x' + ''.join(hex(_)[2:] for _ in args), 16)

	def update_psw(self, c: int = 0, z: int = 0, s: int = 0, ov: int = 0, mie: int = 0, hc: int = 0, errlvl: int = 0):
		psw_str = f'0b{c}{z}{s}{ov}{mie}{hc}{"0" if errlvl < 2 else ""}{str(errlvl)[2:]}'
		self.psw.value = int(psw_str, 2)

	def exec_instruction(self):
		csr_pc = self.concat_bytes(self.csr.value, self.pc.value)
		ins_code = self.rom_w[csr_pc:csr_pc+2]
		self.pc.value += 2

		dsr_prefix = False

		# === DSR Prefixes ===
		# DSR <- #imm8
		if ins_code[0] == 0xe3:
			dsr_prefix = True
			self.dsr.value = ins_code[1]
		# DSR <- Rd
		if ins_code[0] == 0x90 and self.get_bits(ins_code[1], 4) == 0xf:
			dsr_prefix = True
			self.dsr.value = self.r[self.get_bits(ins_code[1], 4, 4)].value
		# DSR <- DSR
		elif ins_code == b'\xfe\x9f': dsr_prefix = True

		# this section will be cleaned up once all instructions are added.

		# ADD Rn, Rm
		if self.get_bits(ins_code[0], 4, 4) == 8 and self.get_bits(ins_code[1], 4) == 1:
			n = self.get_bits(ins_code[0], 4); m = self.get_bits(ins_code[0], 4)
			self.r[n].value = self.add(self.r[n].value, self.r[m].value)
		# ADD Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 1:
			n = self.get_bits(ins_code[0], 4); imm8 = ins_code[1]
			self.r[n].value = self.add(self.r[n].value, ctypes.c_byte(imm8).value)
		# ADD ERn, ERm
		elif self.get_bits(ins_code[0], 4, 4) == 0xf and self.get_bits(ins_code[0], 1) == 0 and self.get_bits(ins_code[1], 5) == 6:
			n = 2*self.get_bits(ins_code[0], 3, 1); m = 2*self.get_bits(ins_code[1], 3, 5)
			result = self.add(self.er(n), self.er(m), True)
			self.set_er(n, result)
		# ADD ERn, #imm7
		elif self.get_bits(ins_code[0], 4, 4) == 0xe and self.get_bits(ins_code[0], 1) == 0 and self.get_bits(ins_code[1], 1, 7) == 1:
			n = 2*self.get_bits(ins_code[0], 3, 1); imm7 = self.get_bits(ins_code[1], 7)
			result = self.add(self.er(n), imm_7(imm7).value, True)
			self.set_er(n, result)
		# ADDC Rn, Rm
		elif self.get_bits(ins_code[0], 4, 4) == 8 and self.get_bits(ins_code[1], 4) == 6:
			n = self.get_bits(ins_code[0], 4); m = self.get_bits(ins_code[0], 4)
			self.r[n].value = self.add(self.r[n].value, self.r[m].value)
		# ADDC Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 6:
			n = self.get_bits(ins_code[0], 4); imm8 = ins_code[1]
			self.r[n].value = self.add(self.r[n].value, ctypes.c_byte(imm8).value)
		# AND Rn, Rm
		elif self.get_bits(ins_code[0], 4, 4) == 8 and self.get_bits(ins_code[1], 4) == 2:
			n = self.get_bits(ins_code[0], 4); m = self.get_bits(ins_code[0], 4)
			self.r[n].value = self._and(self.r[n].value, self.r[m].value)
		# AND Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 2:
			n = self.get_bits(ins_code[0], 4); imm8 = ins_code[1]
			self.r[n].value = self._and(self.r[n].value, ctypes.c_byte(imm8).value)
		# more instructions here...
		else:
			logging.basicConfig(datefmt = '%d/%m/%Y %H:%M:%S.%V', format = 'PyU8: %(levelname)s: %(message)s')
			logging.warning(f'unknown instruction code {format(self.concat_bytes(ins_code), "04X")} at address {format(self.csr.value, "02X")}:{format(self.pc.value, "04X")}')

	def add(self, op1: int, op2: int, short: bool = False):
		ctype = ctypes.c_ushort if short else ctypes.c_ubyte

		result = ctype(op1 + op2).value
		carry = (result >> 7) & 1 if ctype == ctypes.c_byte else (result >> 15) & 1
		overflow = result < op1
		half_carry = (result & (1 << 3)) >> 3

		self.update_psw(*[int(_) for _ in (carry, result == 0, result < 0, overflow)], hc=int((result & (1 << 3)) >> 3))
		return result

	def addc(self, op1: int, op2: int):
		result = ctypes.c_byte(op1 + op2 + self.get_bits(self.psw, 1, 7)).value
		carry = (result >> 7) & 1
		overflow = result < op1
		half_carry = (result & (1 << 3)) >> 3

		self.update_psw(*[int(_) for _ in (carry, self.get_bits(self.psw, 1, 6) == 1 and result == 0, result < 0, overflow)], hc=int((result & (1 << 3)) >> 3))
		return result

	def _and(self, op1: int, op2: int):
		result = ctypes.c_byte(op1 & op2).value

		self.update_psw(z = int(result == 0), s = int(result < 0))
		return result

	# more instructions here...
