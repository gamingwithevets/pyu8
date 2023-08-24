import sys
if __name__ == '__main__':
	print('Please use this script as a module.')
	sys.exit()

import time
import ctypes
import logging
from functools import lru_cache

class CPUError(Exception): pass

class uint_4(ctypes.Structure):
	_fields_ = [('value', ctypes.c_uint8, 4)]

	def __init__(self, value: int = 0): self.value = value & 0xf
	def __repr__(self): return f'uint_4({self.value})'
	def __str__(self): return str(self.value)

class imm_7(ctypes.Structure):
	_fields_ = [('value', ctypes.c_byte, 7)]

	def __init__(self, value: int = 0): self.value = value & 0x7f
	def __repr__(self): return f'imm_7({self.value})'
	def __str__(self): return str(self.value)

class GR_t(ctypes.Union):
	_fields_ = [
		('qrs', ctypes.c_uint64 * 2),
		('xrs', ctypes.c_uint32 * 4),
		('ers', ctypes.c_uint16 * 8),
		('rs', ctypes.c_uint8 * 16)
	]

class PSW_t_field(ctypes.Structure):
	_fields_ = [
		('elevel', ctypes.c_uint8, 2),
		('hc', ctypes.c_uint8, 1),
		('mie', ctypes.c_uint8, 1),
		('ov', ctypes.c_uint8, 1),
		('s', ctypes.c_uint8, 1),
		('z', ctypes.c_uint8, 1),
		('c', ctypes.c_uint8, 1),
	]

class PSW_t(ctypes.Union):
	_fields_ = [
		('raw', ctypes.c_uint8),
		('field', PSW_t_field),
	]

class U8:
	def __init__(self, rom: bytes, config: int) -> None:
		if len(rom) < 0xFF: raise CPUError('ROM must be at least 127 words (0xFF bytes)')
		elif len(rom) % 2 != 0: raise CPUError(f'ROM is of ODD length ({hex(len(self.code_mem))})')

		# https://stackoverflow.com/a/14543975
		self.code_mem_w = bytes([c for t in zip(rom[1::2], rom[::2]) for c in t])
		self.code_mem = rom

		self.rwind_size = config['rwind_size']
		self.ro_ranges = config['ro_ranges']

		self.reset = False

		self.reset_el_2_ptr = ctypes.c_uint16(self.concat_bytes(self.code_mem_w[2:4]))
		self.reset_el_0_ptr = ctypes.c_uint16(self.concat_bytes(self.code_mem_w[4:6]))

		self.nmice_ptr = ctypes.c_uint16(self.concat_bytes(self.code_mem_w[6:8]))
		self.hwi_ptrs = [ctypes.c_uint16(self.concat_bytes(self.code_mem_w[i:i+2])) for i in range(0xa, 0x7f, 2)]
		self.swi_ptrs = [ctypes.c_uint16(self.concat_bytes(self.code_mem_w[i:i+2])) for i in range(0x80, 0xff, 2)]

		self.data_mem = rom[:rwind_size]
		self.data_mem += [0] * (0x10000 - rwind_size)

		self.sp = ctypes.c_uint16()
		self.gr = GR_t()
		self.cr = GR_t()
		self.psw = PSW_t()
		self.pc = ctypes.c_uint16()
		self.csr = uint_4()
		self.lr = ctypes.c_uint16()
		self.elr1 = self.elr2 = self.elr3 = ctypes.c_uint16()
		self.lcsr = uint_4()
		self.ecsr1 = self.ecsr2 = self.ecsr3 = uint_4()
		self.epsw1 = self.epsw2 = self.epsw3 = PSW_t()
		self.ea = ctypes.c_uint16()
		self.ar = ctypes.c_uint16()
		self.dsr = ctypes.c_uint8()

	def run(self) -> None:
		self.reset_registers()
		while True:
			self.exec_instruction()
			if self.reset: self.reset_registers()

	def reset_registers(self) -> None:
		self.sp.value = self.concat_bytes(self.code_mem[0:2])
		for i in range(16): self.gr[i] = self.cr[i] = 0
		self.psw.raw = 0
		self.pc.value = self.addr_filter(self.reset_el_2_ptr.value)
		self.csr.value = 0
		self.lr.value = 0
		self.elr1.value = self.elr2.value = self.elr2.value = 0
		self.lcsr.value = 0
		self.ecsr1.value = self.ecsr2.value = self.ecsr3.value = 0
		self.epsw1.raw = self.epsw2.raw = self.epsw3.raw = 0
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

	# https://stackoverflow.com/a/20024864
	@staticmethod
	@lru_cache
	def split_bytes(data: bytes) -> list: return [data[i:i+2] for i in range(0, len(data), 2)]
	
	@staticmethod
	@lru_cache
	def split_nibble(data: int) -> list: return data >> 4, data & 0xf

	@staticmethod
	@lru_cache
	def get_bits(byte: int, num_bits: int, start_bit: int = 0) -> int: return (byte >> start_bit) & ((1 << num_bits) - 1)

	@staticmethod
	@lru_cache
	def concat_bytes(*args: bytes) -> int:
		if len(args) == 1: return int('0x' + ''.join(format(_, "02x") for _ in args[0]), 16)
		else: return int('0x' + ''.join(format(_, "02x") for _ in args), 16)

	def warn(self, msg): logging.warning(f'{format(self.csr.value, "02X")}:{format(self.pc.value, "04X")}: {msg}')

	def exec_instruction(self) -> None:
		def warn(): self.warn(f'unimplemented/invalid instruction code {format(self.concat_bytes(ins_code), "04X")}')

		self.pc.value = self.addr_filter(self.pc.value)

		csr_pc = self.concat_bytes(self.csr.value, self.pc.value)
		ins_code = self.read_mem(csr_pc, 2, True)
		next_instruction = csr_pc + 2

		dsr_prefix = False

		# === DSR Prefixes ===
		# DSR <- #imm8
		if ins_code[0] == 0xe3:
			dsr_prefix = True
			self.dsr.value = ins_code[1]
		# DSR <- Rd
		if ins_code[0] == 0x90 and self.get_bits(ins_code[1], 4) == 0xf:
			dsr_prefix = True
			self.dsr.value = self.gr.rs[self.get_bits(ins_code[1], 4, 4)].value
		# DSR <- DSR
		elif ins_code == b'\xfe\x9f': dsr_prefix = True

		if dsr_prefix: next_instruction += 2
		
		# CPLC
		if ins_code == 'b\xfe\xcf': self.psw.field.c = not self.psw.field.c
		# BRK
		elif ins_code == b'\xff\xff':
			if psw['elevel'] > 1: self.reset = True
			elif psw['elevel'] < 2:
				self.elr2.value = next_instruction
				self.ecsr2.value = self.csr.value
				self.epsw2 = self.psw.raw
				self.psw.field['elevel'] = 2
				self.pc.value = self.reset_el_2_ptr
		# ___ Rn[, Rm]
		elif self.get_bits(ins_code[0], 4, 4) == 8:
			n = self.get_bits(ins_code[0], 4)
			rn = self.gr.rs[n].value
			rm = self.gr.rs[self.get_bits(ins_code[1], 4, 4)].value
			# ADD Rn, Rm
			if self.get_bits(ins_code[1], 4) == 1: self.gr.rs[n].value = self.add(rn, rm)
			# AND Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 2: self.gr.rs[n].value = self._and(rn, rm)
			# CMPC Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 5: self.subc(rn, rm)
			# ADDC Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 6: self.gr.rs[n].value = self.addc(rn, rm)
			# CMP Rn, Rm
			elif self.get_bits(ins_code[1], 4) == 7: self.sub(rn, rm)
			# DAA Rn
			elif ins_code[1] == 0x1f: self.gr.rs[n].value = self.daa(rn)
			# DAS Rn
			elif ins_code[1] == 0x3f: self.gr.rs[n].value = self.das(rn)
		# ___ ERn, ERm
		elif self.get_bits(ins_code[0], 4, 4) == 0xf:
			n = self.get_bits(ins_code[0], 4)
			ern = self.gr.ers[n]
			erm = self.gr.ers[self.get_bits(ins_code[1], 4, 4)]
			if ern is not None and erm is not None:
				# ADD
				if self.get_bits(ins_code[1], 4) == 6:
					result = self.add(ern, erm, True)
					self.set_er(n, result)
				# CMP
				elif self.get_bits(ins_code[1], 4) == 7: self.sub(ern, erm, True)
		# ___ ERn, #imm7
		elif self.get_bits(ins_code[0], 4, 4) == 0xe and self.get_bits(ins_code[0], 1) == 0:
			n = self.get_bits(ins_code[0], 4)
			ern = self.gr.ers[n]; imm7 = self.get_bits(ins_code[1], 7)
			if ern is not None:
				# ADD
				if self.get_bits(ins_code[1], 1, 7) == 1:
					result = self.add(self.gr.ers[n], imm_7(imm7).value, True)
					self.set_er(n, result)
		# ADD Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 1:
			n = self.get_bits(ins_code[0], 4)
			self.gr.rs[n].value = self.add(self.value, ctypes.c_byte(ins_code[1]).value)
		# AND Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 2:
			n = self.get_bits(ins_code[0], 4)
			self.gr.rs[n].value = self._and(self.gr.rs[n].value, ctypes.c_byte(ins_code[1]).value)
		# CMPC Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 5: self.cmpc(self.gr.rs[self.get_bits(ins_code[0], 4)].value, ctypes.c_byte(ins_code[1]).value)
		# ADDC Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 6:
			n = self.get_bits(ins_code[0], 4)
			self.gr.rs[n].value = self.addc(self.gr.rs[n].value, ctypes.c_byte(ins_code[1]).value)
		# CMP Rn, #imm8
		elif self.get_bits(ins_code[0], 4, 4) == 7: self.sub(self.gr.rs[self.get_bits(ins_code[0], 4)].value, ctypes.c_byte(ins_code[1]).value)
		# ADD SP, #signed8
		elif ins_code[0] == 0xe1:
			signed8 = ins_code[1]
			self.sp.value = self.add(self.sp.value, ctypes.c_byte(signed8).value, set_psw_ = False)
		# Bcond Radr/BC cond, Radr
		elif self.get_bits(ins_code[0], 4, 4) == 0xc:
			radr = ctypes.c_byte(ins_code[1]).value
			cond = False
			cond_hex = self.get_bits(ins_code[0], 4)
			# GE/NC
			if cond_hex == 0: cond = self.psw.field.c == 0
			# LT/CY
			elif cond_hex == 1: cond = self.psw.field.c == 1
			# GT
			elif cond_hex == 2: cond = self.psw.field.c == psw['z'] == 0
			# LE
			elif cond_hex == 3: cond = self.psw.field.z == 1 or self.psw.field.c == 1
			# GES
			elif cond_hex == 4: cond = self.psw.field.ov ^ self.psw.field.s == 0
			# LTS
			elif cond_hex == 5: cond = self.psw.field.ov ^ self.psw.field.s == 1
			# GTS
			elif cond_hex == 6: cond = (self.psw.field.ov ^ self.psw.field.s) | self.psw.field.z == 0
			# LES
			elif cond_hex == 7: cond = (self.psw.field.ov ^ self.psw.field.s) | self.psw.field.z == 1
			# NE/NZ
			elif cond_hex == 8: cond = self.psw.field.z == 0
			# EQ/ZF
			elif cond_hex == 9: cond = self.psw.field.z == 1
			# NV
			elif cond_hex == 0xa: cond = self.psw.field.ov == 0
			# OV
			elif cond_hex == 0xb: cond = self.psw.field.ov == 1
			# PS
			elif cond_hex == 0xc: cond = self.psw.field.s == 0
			# NS
			elif cond_hex == 0xd: cond = self.psw.field.s == 1
			# AL
			elif cond_hex == 0xe: cond = True
			else: warn()
			if cond: self.pc.value += radr
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
			if self.get_bits(ins_code[1], 4) == 2: self.pc.value = self.gr.ers[self.get_bits(ins_code[1], 4, 4)]
			# BL ERn
			elif self.get_bits(ins_code[1], 4) == 3:
				next_instruction = self.gr.ers[self.get_bits(ins_code[1], 4, 4)]
				self.lr.value = next_instruction
				self.lcsr.value = self.csr.value
		else: warn()

		self.pc.value = next_instruction

	def add(self, op1: int, op2: int, short: bool = False, set_psw_: bool = True):
		ctype = ctypes.c_short if short else ctypes.c_byte

		result = ctype(op1 + op2).value
		carry = (result >> 15) & 1 if short else (result >> 7) & 1
		overflow = result < op1
		half_carry = ((result >> 11) & 1 or (result & (1 << 11)) >> 11) if short else ((result >> 3) & 1 or (result & (1 << 3)) >> 3)

		if set_psw_:
			self.psw.field.c = int(carry)
			self.psw.field.z = int(result == 0)
			self.psw.field.s = int(result < 0)
			self.psw.field.ov = int(overflow)
			self.psw.field.hc = int(half_carry)
	
		return result

	def addc(self, op1: int, op2: int):
		result = ctypes.c_byte(op1 + op2 + self.psw.field.c).value
		carry = (result >> 7) & 1
		overflow = result < op1
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.psw.field.c = int(carry)
		self.psw.field.z = int(self.psw.field.z == 1 and result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = int(overflow)
		self.psw.field.hc = int(half_carry)

		return result

	def _and(self, op1: int, op2: int):
		result = ctypes.c_byte(op1 & op2).value

		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		
		return result

	def sub(self, op1: int, op2: int, short: bool = False):
		ctype = ctypes.c_short if short else ctypes.c_byte

		result = ctype(op1 - op2).value
		carry = (result >> 15) & 1 if short else (result >> 7) & 1
		overflow = result < op1
		half_carry = ((result >> 11) & 1 or (result & (1 << 11)) >> 11) if short else ((result >> 3) & 1 or (result & (1 << 3)) >> 3)

		self.psw.field.c = int(carry)
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = int(overflow)
		self.psw.field.hc = int(half_carry)

		return result

	def subc(self, op1: int, op2: int):
		result = ctypes.c_short(op1 - op2 - self.psw.field.c).value
		carry = (result >> 15) & 1
		overflow = result < op1
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.psw.field.c = int(carry)
		self.psw.field.z = int(self.psw.field.z == 1 and result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = int(overflow)
		self.psw.field.hc = int(half_carry)

		return result

	def daa(self, op1: int):
		split = self.split_nibble(op1)

		result_orig = op1
		carry = self.psw.field.c

		if self.psw.field.c == 0:
			if split[0] in range(10):
				if self.psw.field.hc == 0 and split[1] in range(10): result_orig += 0
				elif self.psw.field.hc == 1: result_orig += 6
			elif split[0] in range(9) and self.psw.field.hc == 0 and split[1] in range(10, 16): result_orig += 6
			elif split[0] in range(10, 16):
				if self.psw.field.hc == 0 and split[1] in range(10): result_orig += 0x60; carry = 1
				elif self.psw.field.hc == 1: result_orig += 0x66; carry = 1
			elif split[0] in range(9, 16) and self.psw.field.hc == 0 and split[1] in range(10, 16): result_orig += 0x66; carry = 1
		elif self.psw.field.c == 1:
			if self.psw.field.hc == 0:
				if split[1] in range(10): result_orig += 0x60
				elif split[1] in range(10, 16): result_orig += 0x66
			elif self.psw.field.hc == 1: result_orig += 66

		result = ctypes.c_byte(result_orig).value
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.psw.field.c = int(carry)
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.hc = int(half_carry)

		return result

	def das(self, op1: int):
		split = self.split_nibble(op1)

		result_orig = op1

		if self.psw.field.c == 0:
			if split[0] in range(10):
				if self.psw.field.hc == 0 and split[1] in range(10): result_orig -= 0
				elif self.psw.field.hc == 1: result_orig -= 6
			elif split[0] in range(9) and self.psw.field.hc == 0 and split[1] in range(10, 16): result_orig -= 6
			elif split[0] in range(10, 16):
				if self.psw.field.hc == 0:
					if split[1] in range(10): result_orig -= 0x60
					elif split[1] in range(10, 16): result_orig -= 0x66
				elif self.psw.field.hc == 1: result_orig += 0x66
			elif split[0] in range(9, 16) and self.psw.field.hc == 0 and split[1] in range(10, 16): result_orig -= 0x66
		elif self.psw.field.c == 1:
			if self.psw.field.hc == 0:
				if split[1] in range(10): result_orig -= 0x60
				elif split[1] in range(10, 16): result_orig -= 0x66
			elif self.psw.field.hc == 1: result_orig += 66

		result = ctypes.c_byte(result_orig).value
		carry = result_orig > 0xff
		half_carry = (result >> 3) & 1 or (result & (1 << 3)) >> 3

		self.psw.field.c = int(carry)
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.hc = int(half_carry)

		return result
