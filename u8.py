import sys
if __name__ == '__main__':
	print('Please use this script as a module.')
	sys.exit()

import time
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
	def __init__(self, rom: bytes, rwind_size: int) -> None:
		if len(rom) < 0xFF: raise CPUError('ROM must be at least 127 words (0xFF bytes)')
		elif len(rom) % 2 != 0: raise CPUError(f'ROM is of ODD length ({hex(len(self.code_mem))})')

		# https://stackoverflow.com/a/14543975
		self.code_mem_w = bytes([c for t in zip(rom[1::2], rom[::2]) for c in t])
		self.code_mem = rom

		self.rwind_size = rwind_size

		self.reset = False

		self.reset_el_2_ptr = ctypes.c_ushort(self.concat_bytes(self.code_mem_w[2:4]))
		self.reset_el_0_ptr = ctypes.c_ushort(self.concat_bytes(self.code_mem_w[4:6]))

		self.nmice_ptr = ctypes.c_ushort(self.concat_bytes(self.code_mem_w[6:8]))
		self.hwi_ptrs = [ctypes.c_ushort(self.concat_bytes(self.code_mem_w[i:i+2])) for i in range(0xa, 0x7f, 2)]
		self.swi_ptrs = [ctypes.c_ushort(self.concat_bytes(self.code_mem_w[i:i+2])) for i in range(0x80, 0xff, 2)]

		self.data_mem = rom[:rwind_size]
		self.data_mem += [0] * (16384 - rwind_size)

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

	def run(self) -> None:
		self.reset_registers()
		while True:
			self.exec_instruction()
			if self.reset: self.reset_registers()

	def reset_registers(self) -> None:
		self.sp.value = self.concat_bytes(self.code_mem[0:2])
		for reg in self.r: reg.value = 0
		for reg in self.cr: reg.value = 0
		self.psw.value = 0
		self.pc.value = self.addr_filter(self.reset_el_2_ptr.value)
		self.csr.value = 0
		self.lr.value = 0
		self.elr1.value = self.elr2.value = self.elr2.value = 0
		self.lcsr.value = 0
		self.ecsr1.value = self.ecsr2.value = self.ecsr3.value = 0
		self.epsw1.value = self.epsw2.value = self.epsw3.value = 0
		self.ea.value = 0
		self.ar.value = 0
		self.dsr.value = 0

		self.reset = False

	@staticmethod
	@lru_cache
	def addr_filter(addr):
		if type(addr) == int: return addr if addr % 2 == 0 else addr|1
		elif type(addr) == bytes:
			last = addr[-1]
			return addr[:-1] + bytes(last if last % 2 == 0 else last|1)

	def read_mem(self, addr: int, bytes_to_fetch: int = 1, little = False) -> int:
		where2fetch = self.code_mem_w if little else self.code_mem
		return where2fetch[addr:addr + bytes_to_fetch]

	def er(self, n: int) -> int:
		if n % 2 != 0: self.warn(f'tried to fetch invalid ER register number ({n})')
		else: return self.concat_bytes(self.r[n+1].value, self.r[n].value)
	def xr(self, n: int) -> int:
		if n % 4 != 0: self.warn(f'tried to fetch invalid XR register number ({n})')
		else: return self.concat_bytes(self.er(n), self.er(n+1))
	def qr(self, n: int) -> int:
		if n % 8 != 0: self.warn(f'tried to fetch invalid QR register number ({n})')
		else: return self.concat_bytes(self.xr(n), self.xr(n+1))
	def ecr(self, n: int) -> int:
		if n % 2 != 0: self.warn(f'tried to fetch invalid ECR register number ({n})')
		else: return self.concat_bytes(self.cr[n+1].value, self.cr[n].value)
	def xcr(self, n: int) -> int:
		if n % 4 != 0: self.warn(f'tried to fetch invalid XCR register number ({n})')
		else: return self.concat_bytes(self.ecr(n), self.ecr(n+1))
	def qcr(self, n: int) -> int:
		if n % 8 != 0: self.warn(f'tried to fetch invalid QCR register number ({n})')
		else: return self.concat_bytes(self.xcr(n), self.xcr(n+1))

	# https://stackoverflow.com/a/20024864
	@staticmethod
	@lru_cache
	def split_bytes(data: bytes) -> list: return [data[i:i+2] for i in range(0, len(data), 2)]
	
	@staticmethod
	@lru_cache
	def split_nibble(data: int) -> list: return data >> 4, data & 0xf

	def set_er(self, n: int, val: int) -> None:
		byt = val.to_bytes(2, 'little')
		self.r[n].value = byt[0]; self.r[n + 1].value = byt[1]

	def set_xr(self, n: int, val: int) -> None:
		val_s = self.split_bytes(val)
		self.set_er(n, val_s[0]); self.set_er(n + 1, val_s[1])

	def set_qr(self, n: int, val: int) -> None:
		val_s = self.split_bytes(val)
		self.set_xr(n, val_s[0]); self.set_er(n + 1, val_s[1])

	def set_ecr(self, n: int, val: int) -> None:
		byt = val.to_bytes(2, 'little')
		self.cr[n].value = byt[0]; self.cr[n + 1].value = byt[1]

	def set_xcr(self, n: int, val: int) -> None:
		val_s = self.split_bytes(val)
		self.set_ecr(n, val_s[0]); self.set_ecr(n + 1, val_s[1])

	def set_qcr(self, n: int, val: int) -> None:
		val_s = self.split_bytes(val)
		self.set_xcr(n, val_s[0]); self.set_xcr(n + 1, val_s[1])

	@staticmethod
	@lru_cache
	def get_bits(byte: int, num_bits: int, start_bit: int = 0) -> int: return (byte >> start_bit) & ((1 << num_bits) - 1)

	@staticmethod
	@lru_cache
	def concat_bytes(*args: bytes) -> int:
		if len(args) == 1: return int('0x' + ''.join(format(_, "02x") for _ in args[0]), 16)
		else: return int('0x' + ''.join(format(_, "02x") for _ in args), 16)

	def fmt_psw(self) -> dict:
		psw = format(self.psw.value, '08b')
		return {'c': int(psw[0]), 'z': int(psw[1]), 's': int(psw[2]), 'ov': int(psw[3]), 'mie': int(psw[4]), 'hc': int(psw[5]), 'elevel': int('0b' + psw[6:], 2)}

	def set_psw(self, c: int = 0, z: int = 0, s: int = 0, ov: int = 0, mie: int = 0, hc: int = 0, elevel: int = 0) -> None:
		psw_str = f'0b{c}{z}{s}{ov}{mie}{hc}{"0" if elevel < 2 else "1"}{str(elevel)[2:]}'
		self.psw.value = int(psw_str, 2)

	def warn(self, msg): logging.warning(f'{format(self.csr.value, "02X")}:{format(self.pc.value, "04X")}: {msg}')

	def exec_instruction(self) -> None:
		def warn(): self.warn(f'unknown instruction code {format(self.concat_bytes(ins_code), "04X")}')

		self.pc.value = self.addr_filter(self.pc.value)

		csr_pc = self.concat_bytes(self.csr.value, self.pc.value)
		ins_code = self.read_mem(csr_pc, 2, True)
		next_instruction = self.pc.value + 2
		psw = self.fmt_psw()

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

		if dsr_prefix: next_instruction += 2
		
		# CPLC
		if ins_code == 'b\xfe\xcf': self.set_psw(int(not self.fmt_psw['c']))
		# BRK
		elif ins_code == b'\xff\xff':
			if psw['elevel'] > 1: self.reset = True
			elif psw['elevel'] < 2:
				self.elr2.value = csr_pc + 2
				self.ecsr2.value = self.csr.value
		# ___ Rn[, Rm]
		elif self.get_bits(ins_code[0], 4, 4) == 8:
			n = self.get_bits(ins_code[0], 4)
			rn = self.r[n].value
			rm = self.r[self.get_bits(ins_code[1], 4, 4)].value
			# ADD Rn, Rm
			if self.get_bits(ins_code[1], 4) == 1: self.r[n].value = self.add(rn, rm)
			# AND Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 2: self.r[n].value = self._and(rn, rm)
			# CMPC Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 5: self.cmpc(rn, rm)
			# ADDC Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 6: self.r[n].value = self.addc(rn, rm)
			# CMP Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 7: self.cmp(rn, rm)
			# DAA Rn
			elif ins_code[1] == 0x1f: self.r[n].value = self.daa(rn)
			# DAS Rn
			elif ins_code[1] == 0x3f: self.r[n].value = self.das(rn)
		# ___ ERn, ERm
		elif self.get_bits(ins_code[0], 4, 4) == 0xf:
			n = self.get_bits(ins_code[0], 4)
			ern = self.er(n)
			erm = self.er(self.get_bits(ins_code[1], 4, 4))
			if ern is not None and erm is not None:
				# ADD
				if self.get_bits(ins_code[1], 4) == 6:
					result = self.add(ern, erm, True)
					self.set_er(n, result)
				# CMP
				elif self.get_bits(ins_code[1], 4) == 7: self.cmp(ern, erm, True)
		# ___ ERn, #imm7
		elif self.get_bits(ins_code[0], 4, 4) == 0xe and self.get_bits(ins_code[0], 1) == 0:
			n = self.get_bits(ins_code[0], 4)
			ern = self.er(n); imm7 = self.get_bits(ins_code[1], 7)
			if ern is not None:
				# ADD
				if self.get_bits(ins_code[1], 1, 7) == 1:
					result = self.add(self.er(n), imm_7(imm7).value, True)
					self.set_er(n, result)
		# ADD Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 1:
			n = self.get_bits(ins_code[0], 4)
			self.r[n].value = self.add(self.r[n].value, ctypes.c_byte(ins_code[1]).value)
		# AND Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 2:
			n = self.get_bits(ins_code[0], 4)
			self.r[n].value = self._and(self.r[n].value, ctypes.c_byte(ins_code[1]).value)
		# CMPC Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 5: self.cmpc(self.r[self.get_bits(ins_code[0], 4)].value, ctypes.c_byte(ins_code[1]).value)
		# ADDC Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 6:
			n = self.get_bits(ins_code[0], 4)
			self.r[n].value = self.addc(self.r[n].value, ctypes.c_byte(ins_code[1]).value)
		# CMP Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 7: self.cmp(self.r[self.get_bits(ins_code[0], 4)].value, ctypes.c_byte(ins_code[1]).value)
		# ADD SP, #signed8
		elif ins_code[0] == 0xe1:
			signed8 = ins_code[1]
			self.sp.value = self.add(self.sp.value, ctypes.c_byte(signed8).value, set_psw_ = False)
		# Bcond Radr/BC cond, Radr
		elif self.get_bits(ins_code[0], 4, 4) == 0xc:
			radr = ins_code[1]
			cond = False
			# GE/NC
			if self.get_bits(ins_code[0], 4) == 0: cond = psw['c'] == 0
			# LT/CY
			elif self.get_bits(ins_code[0], 4) == 1: cond = psw['c'] == 1
			# GT
			elif self.get_bits(ins_code[0], 4) == 2: cond = psw['c'] == psw['z'] == 0
			# LE
			elif self.get_bits(ins_code[0], 4) == 3: cond = psw['z'] == 1 or psw['c'] == 1
			# GES
			elif self.get_bits(ins_code[0], 4) == 4: cond = psw['ov'] ^ psw['s'] == 0
			# LTS
			elif self.get_bits(ins_code[0], 4) == 5: cond = psw['ov'] ^ psw['s'] == 1
			# GTS
			elif self.get_bits(ins_code[0], 4) == 6: cond = (psw['ov'] ^ psw['s']) | psw['z'] == 0
			# LES
			elif self.get_bits(ins_code[0], 4) == 7: cond = (psw['ov'] ^ psw['s']) | psw['z'] == 1
			# NE/NZ
			elif self.get_bits(ins_code[0], 4) == 8: cond = psw['z'] == 0
			# EQ/ZF
			elif self.get_bits(ins_code[0], 4) == 9: cond = psw['z'] == 1
			# NV
			elif self.get_bits(ins_code[0], 4) == 0xa: cond = psw['ov'] == 0
			# OV
			elif self.get_bits(ins_code[0], 4) == 0xb: cond = psw['ov'] == 1
			# PS
			elif self.get_bits(ins_code[0], 4) == 0xc: cond = psw['s'] == 0
			# NS
			elif self.get_bits(ins_code[0], 4) == 0xd: cond = psw['s'] == 1
			# AL
			elif self.get_bits(ins_code[0], 4) == 0xe: cond = True
			else: warn()
			if cond: self.pc.value = radr
		elif self.get_bits(ins_code[0], 4, 4) == 0xf:
			# B Cadr
			if ins_code[1] == 0:
				self.csr.value = self.get_bits(ins_code[0], 4)
				next_instruction = self.read_mem(csr_pc + 2, 2)
			# BL Cadr
			if ins_code[1] == 1:
				self.lr.value = next_instruction + 2
				self.lcsr.value = self.csr.value
				self.csr.value = self.get_bits(ins_code[0], 4)
				next_instruction = self.read_mem(csr_pc + 2, 2)
		elif ins_code[0] == 0xf0:
			# B ERn
			if self.get_bits(ins_code[1], 4) == 2: self.pc.value = self.er(self.get_bits(ins_code[1], 4, 4))
			# BL ERn
			elif self.get_bits(ins_code[1], 4) == 3:
				next_instruction = self.er(self.get_bits(ins_code[1], 4, 4))
				self.lr.value = next_instruction
				self.lcsr.value = self.csr.value
		else: warn()

		self.pc.value = next_instruction

	def add(self, op1: int, op2: int, short: bool = False, set_psw_: bool = True):
		ctype = ctypes.c_ushort if short else ctypes.c_ubyte

		result = ctype(op1 + op2).value
		carry = (result >> 15) & 1 if short else (result >> 7) & 1
		overflow = result < op1
		half_carry = ((result >> 11) & 1 or (result & (1 << 11)) >> 11) if short else ((result >> 3) & 1 or (result & (1 << 3)) >> 3)

		if set_psw_: self.set_psw(*[int(_) for _ in (carry, result == 0, result < 0, overflow)], hc = half_carry)
		return result

	def addc(self, op1: int, op2: int):
		psw = self.fmt_psw()

		result = ctypes.c_byte(op1 + op2 + psw['c']).value
		carry = (result >> 7) & 1
		overflow = result < op1
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.set_psw(*[int(_) for _ in (carry, psw['z'] == 1 and result == 0, result < 0, overflow)], hc = half_carry)
		return result

	def _and(self, op1: int, op2: int):
		result = ctypes.c_byte(op1 & op2).value

		self.set_psw(z = int(result == 0), s = int(result < 0))
		return result

	def cmp(self, op1: int, op2: int, short: bool = False):
		ctype = ctypes.c_ushort if short else ctypes.c_ubyte

		result = ctype(op1 - op2).value
		carry = (result >> 15) & 1 if short else (result >> 7) & 1
		overflow = result < op1
		half_carry = ((result >> 11) & 1 or (result & (1 << 11)) >> 11) if short else ((result >> 3) & 1 or (result & (1 << 3)) >> 3)

		self.set_psw(*[int(_) for _ in (carry, result == 0, result < 0, overflow)], hc = half_carry)

	def cmpc(self, op1: int, op2: int):
		psw = self.fmt_psw()

		result = ctypes.c_short(op1 - op2 - psw['c']).value
		carry = (result >> 15) & 1
		overflow = result < op1
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.set_psw(*[int(_) for _ in (carry, psw['z'] == 1 and result == 0, result < 0, overflow)], hc = half_carry)

	def daa(self, op1: int):
		split = self.split_nibble(op1)
		psw = self.fmt_psw()

		result_orig = op1
		carry = psw['c']

		if psw['c'] == 0:
			if split[0] in range(10):
				if psw['hc'] == 0 and split[1] in range(10): result_orig += 0
				elif psw['hc'] == 1: result_orig += 6
			elif split[0] in range(9) and psw['hc'] == 0 and split[1] in range(10, 16): result_orig += 6
			elif split[0] in range(10, 16):
				if psw['hc'] == 0 and split[1] in range(10): result_orig += 0x60; carry = 1
				elif psw['hc'] == 1: result_orig += 0x66; carry = 1
			elif split[0] in range(9, 16) and psw['hc'] == 0 and split[1] in range(10, 16): result_orig += 0x66; carry = 1
		elif psw['c'] == 1:
			if psw['hc'] == 0:
				if split[1] in range(10): result_orig += 0x60
				elif split[1] in range(10, 16): result_orig += 0x66
			elif psw['hc'] == 1: result_orig += 66

		result = ctypes.c_byte(result_orig).value
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.set_psw(*[int(_) for _ in (carry, result == 0, result < 0)], hc = half_carry)
		return result

	def das(self, op1: int):
		split = self.split_nibble(op1)
		psw = self.fmt_psw()

		result_orig = op1

		if psw['c'] == 0:
			if split[0] in range(10):
				if psw['hc'] == 0 and split[1] in range(10): result_orig -= 0
				elif psw['hc'] == 1: result_orig -= 6
			elif split[0] in range(9) and psw['hc'] == 0 and split[1] in range(10, 16): result_orig -= 6
			elif split[0] in range(10, 16):
				if psw['hc'] == 0:
					if split[1] in range(10): result_orig -= 0x60
					elif split[1] in range(10, 16): result_orig -= 0x66
				elif psw['hc'] == 1: result_orig += 0x66
			elif split[0] in range(9, 16) and psw['hc'] == 0 and split[1] in range(10, 16): result_orig -= 0x66
		elif psw['c'] == 1:
			if psw['hc'] == 0:
				if split[1] in range(10): result_orig -= 0x60
				elif split[1] in range(10, 16): result_orig -= 0x66
			elif psw['hc'] == 1: result_orig += 66

		result = ctypes.c_byte(result_orig).value
		carry = result_orig > 0xff
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.set_psw(*[int(_) for _ in (carry, result == 0, result < 0)], hc = half_carry)
		return result
