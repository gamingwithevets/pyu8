import sys
if __name__ == '__main__':
	print('Please use this script as a module.')
	sys.exit()

import platform
if sys.version_info < (3, 6, 0, 'alpha', 4):
	print('This program requires at least Python 3.6.0a4. (You are running Python ', platform.python_version(), ')')
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
	_fields_ = [('value', ctypes.c_int8, 7)]

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
	def __init__(self, rom: bytes) -> None:
		if len(rom) != 0x20000: raise CPUError('ROM must be 0x20000 bytes')

		self.code_mem = rom

		self.reset_ptr = self.read_cmem(2)
		self.reset_brk_ptr = self.read_cmem(4)

		self.nmice_ptr = self.read_cmem(6)
		self.hwi_ptrs = [self.read_cmem(i) for i in range(0xa, 0x7f, 2)]
		self.swi_ptrs = [self.read_cmem(i) for i in range(0x80, 0xff, 2)]

		self.ram = [0] * 0xe00
		self.sfr_mem = [0] * 0x1000

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

		# configurations for step
		self.dsr_prefix = False
		self.ea_inc_delay = 0
		self.mask_cycle = 0
		self.romwin_acc = 0

	def reset(self) -> None:
		self.sp.value = self.read_cmem(0)
		for i in range(16): self.gr.rs[i] = self.cr.rs[i] = 0
		self.psw.raw = 0
		self.pc.value = self.reset_ptr & 0xfffe
		self.csr.value = 0
		self.lr.value = 0
		self.elr1.value = self.elr2.value = self.elr2.value = 0
		self.lcsr.value = 0
		self.ecsr1.value = self.ecsr2.value = self.ecsr3.value = 0
		self.epsw1.raw = self.epsw2.raw = self.epsw3.raw = 0
		self.ea.value = 0
		self.ar.value = 0
		self.dsr.value = 0

	def read_cmem(self, addr: int, segment: int = 0) -> int:
		addr &= 0xfffe
		segment &= 0xe
		return int.from_bytes(self.code_mem[addr:addr+2], 'little')

	def read_dmem(self, addr: int, bytes_to_fetch: int, segment: int = 0) -> int:
		addr %= 0x10000

		fetched_bytes = b''
		for i in range(addr, addr + bytes_to_fetch):
			j = i % 0x10000
			if segment == 0:
				if j < 0x8000:
					fetched_bytes += self.code_mem[j].to_bytes(1, 'big')
					self.romwin_acc += 1
				if j < 0x8e00: fetched_bytes += self.ram[j - 0x8000].to_bytes(1, 'big')
				elif j < 0xf000: fetched_bytes += b'\x00'
				else: fetched_bytes += self.sfr_mem[j - 0xf000].to_bytes(1, 'big')
			elif segment == 1: fetched_bytes += self.code_mem[j + 0x10000].to_bytes(1, 'big')
			elif segment == 8: fetched_bytes += self.code_mem[j].to_bytes(1, 'big')

		return fetched_bytes

	def write_dmem(self, addr: int, bytes_to_write: bytes, segment: int = 0) -> None:
		addr = addr % 0x10000

		for i in range(len(bytes_to_write)):
			j = (addr + i) % 0x10000
			if segment == 0:
				if j < 0x8000: return
				if j < 0x8e00: self.ram[j - 0x8000] = bytes_to_write[i]
				elif j < 0xf000: return
				else: self.sfr_mem[j - 0xf000] = bytes_to_write[i]
			else: return

	# https://stackoverflow.com/a/20024864
	@staticmethod
	@lru_cache
	def split_bytes(data: bytes) -> list: return [data[i:i+2] for i in range(0, len(data), 2)]
	
	@staticmethod
	@lru_cache
	def split_nibble(data: int) -> list: return data >> 4, data & 0xf

	@staticmethod
	@lru_cache
	def comb_nibbs(data: tuple) -> int: return int(hex(data[0]) + hex(data[1])[2:], 16)

	@staticmethod
	@lru_cache
	def get_bits(byte: int, num_bits: int, start_bit: int = 0) -> int: return (byte >> start_bit) & ((1 << num_bits) - 1)

	@staticmethod
	@lru_cache
	def concat_bytes(*args: bytes) -> int:
		if len(args) == 1: return int('0x' + ''.join(format(_, "02x") for _ in args[0]), 16)
		else: return int('0x' + ''.join(format(_, "02x") for _ in args), 16)

	@staticmethod
	def conv_nibbs(data: bytes) -> tuple: return (data[0] >> 4) & 0xf, data[0] & 0xf, (data[1] >> 4) & 0xf, data[1] & 0xf

	def warn(self, msg): logging.warning(f'{format(self.csr.value, "02X")}:{format(self.pc.value, "04X")}: {msg}')
	def err(self, msg): logging.error(f'{format(self.csr.value, "02X")}:{format(self.pc.value, "04X")}: {msg}')

	def step(self) -> None:
		self.pc.value &= 0xfffe
		ins_code_int = self.read_cmem(self.pc.value, self.csr.value)
		next_instruction = (self.pc.value + 2) & 0xfffe
	
		ins_code_raw = ins_code_int.to_bytes(2, 'big')
		ins_code = self.conv_nibbs(ins_code_raw)
		decode_index = self.comb_nibbs((ins_code[0], ins_code[3]))
		n = ins_code[1]
		m = ins_code[2]
		immnum = ctypes.c_uint8(ins_code_raw[1]).value
		adr = self.read_cmem(self.pc.value + 2, self.csr.value)

		cycle_count = 0
		ea_inc = False

		retval = 0

		if ins_code[0] == 0:
			# MOV Rn, #imm8
			cycle_count = 1
			self.gr.rs[n] = immnum
			self.psw.field.z = int(immnum == 0)
			self.psw.field.s = int(immnum < 0)
		elif ins_code[0] == 1:
			# ADD Rn, #imm8
			cycle_count = 1
			self.gr.rs[n] = self.add(self.gr.rs[n], ctypes.c_int8(ins_code[1]).value)
		elif ins_code[0] == 2:
			# AND Rn, #imm8
			cycle_count = 1
			self.gr.rs[n] = self._and(self.gr.rs[n], ctypes.c_int8(ins_code[1]).value)
		elif ins_code[0] == 3:
			# OR Rn, #imm8
			cycle_count = 1
			retval = 2
		elif ins_code[0] == 4:
			# XOR Rn, #imm8
			cycle_count = 1
			retval = 2
		elif ins_code[0] == 5:
			# CMPC Rn, #imm8
			cycle_count = 1
			self.subc(self.gr.rs[ins_code[1]], ctypes.c_int8(ins_code[1]).value)
		elif ins_code[0] == 6:
			# ADDC Rn, #imm8
			cycle_count = 1
			self.gr.rs[n] = self.addc(self.gr.rs[n], ctypes.c_int8(ins_code[1]).value)
		elif ins_code[0] == 7:
			# CMP Rn, #imm8
			cycle_count = 1
			self.sub(self.gr.rs[ins_code[1]], ctypes.c_int8(ins_code[1]).value)
		elif decode_index == 0x80:
			# MOV Rn, Rm
			cycle_count = 1
			src = self.gr.rs[m]
			self.gr.rs[n] = src
			self.psw.field.z = int(src == 0)
			self.psw.field.s = int(src < 0)
		elif decode_index == 0x81:
			# ADD Rn, Rm
			cycle_count = 1
			self.gr.rs[n] = self.add(self.gr.rs[n], self.gr.rs[m])
		elif decode_index == 0x82:
			# AND Rn, Rm
			cycle_count = 1
			self.gr.rs[n] = self._and(self.gr.rs[n], self.gr.rs[m])
		elif decode_index == 0x83:
			# OR Rn, Rm
			retval = 2
		elif decode_index == 0x84:
			# XOR Rn, Rm
			retval = 2
		elif decode_index == 0x85:
			# CMPC Rn, Rm
			cycle_count = 1
			self.gr.rs[n] = self.addc(self.gr.rs[n], self.gr.rs[m])
		elif decode_index == 0x86:
			# ADDC Rn, Rm
			cycle_count = 1
			self.gr.rs[n] = self.addc(self.gr.rs[n], self.gr.rs[m])
		elif decode_index == 0x87:
			# CMP Rn, Rm
			cycle_count = 1
			self.sub(self.gr.rs[n], self.gr.rs[m])
		elif decode_index == 0x8f:
			if ins_code_int & 0xf11f == 0x810f:
				# EXTBW ERn
				retval = 2
			elif ins_code_int & 0xf0ff == 0x801f:
				# DAA Rn
				cycle_count = 1
				self.gr.rs[n] = self.daa(rn)
			elif ins_code_int & 0xf0ff == 0x803f:
				# DAA Rn
				cycle_count = 1
				self.gr.rs[n] = self.das(rn)
			elif ins_code_int & 0xf0ff == 0x805f:
				# NEG Rn
				cycle_count = 1
				retval = 2
			else: retval = 1
		elif decode_index == 0x90:
			src = None
			if ins_code_int & 0x10 == 0:
				# L Rn, [ERm]
				src = self.gr.ers[m]
				cycle_count += ea_inc_delay
			elif ins_code_int & 0xf0ff == 0x9010:
				# L Rn, Dadr
				next_instruction += 2
				src = adr
				cycle_count += ea_inc_delay
			elif ins_code_int & 0xf0ff == 0x9030:
				# L Rn, [EA]
				src = self.ea.value
			elif ins_code_int & 0xf0ff == 0x9050:
				# L Rn, [EA+]
				src = self.ea.value
				self.ea.value += 1; ea_inc = True
			else: retval = 1

			if src is not None:
				val = self.read_dmem(src, 1, self.dsr.value if self.dsr_prefix else 0)
				self.dsr_prefix = False
				cycle_count += 1 + self.romwin_acc

				if val == 0: self.psw.z = 1
				elif val > 0x7f: self.psw.s = 1
				self.gr.rs[n] = val
		elif decode_index == 0x91:
			dest = None
			if ins_code_int & 0x10 == 0:
				# ST Rn, [ERm]
				dest = self.gr.ers[m]
				cycle_count += ea_inc_delay
			elif ins_code_int & 0xf0ff == 0x9011:
				# ST Rn, Dadr
				next_instruction += 2
				dest = adr
				cycle_count += ea_inc_delay
			elif ins_code_int & 0xf0ff == 0x9031:
				# ST Rn, [EA]
				dest = self.ea.value
			elif ins_code_int & 0xf0ff == 0x9051:
				# ST Rn, [EA+]
				dest = self.ea.value
				self.ea.value += 1; ea_inc = True
			else: retval = 1

			if dest is not None:
				val = self.write_dmem(dest, self.gr[n].to_bytes(1, 'big'), self.dsr.value if self.dsr_prefix else 0)
				self.dsr_prefix = False
				cycle_count += 1
		elif decode_index == 0x92:
			src = None
			if ins_code_int & 0x10 == 0:
				# L ERn, [ERm]
				src = self.gr.ers[m]
				cycle_count += ea_inc_delay
			elif ins_code_int & 0xf1ff == 0x9012:
				# L ERn, Dadr
				next_instruction += 2
				src = adr
				cycle_count += ea_inc_delay
			elif ins_code_int & 0xf1ff == 0x9032:
				# L ERn, [EA]
				src = self.ea.value
			elif ins_code_int & 0xf1ff == 0x9052:
				# L ERn, [EA+]
				src = self.ea.value
				self.ea.value += 1; ea_inc = True
			else: retval = 1

			if src is not None:
				val = self.read_dmem(src, 1, self.dsr.value if self.dsr_prefix else 0)[0]
				self.dsr_prefix = False
				cycle_count += 1 + self.romwin_acc

				if val == 0: self.psw.z = 1
				elif val > 0x7f: self.psw.s = 1
				self.gr.ers[n] = val
		elif decode_index == 0x9f:
			if ins_code_int & 0xf00 != 0: retval = 1
			else:
				# [DSR prefix] DSR <- Rd
				self.dsr.value = self.gr.rs[m]
				dsr_prefix = True
				cycle_count = 1
		elif ins_code[0] == 0xc:
			cond = False
			cond_hex = ins_code[1]
			# GE/NC
			if cond_hex == 0: cond = self.psw.field.c == 0
			# LT/CY
			elif cond_hex == 1: cond = self.psw.field.c == 1
			# GT
			elif cond_hex == 2: cond = self.psw.field.c == self.psw['z'] == 0
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
			else: retval = 1
			if cond:
				next_instruction += immnum * 2
				cycle_count = 3
			else: cycle_count = 1
		elif ins_code[0] == 0xe:
			if ins_code_int & 0x180 == 0:
				# MOV ERn, #imm7
				cycle_count = 2
				retval = 2
			elif ins_code_int & 0x180 == 0x80:
				# MOV ERn, #imm7
				cycle_count = 2
				self.gr.ers[n//2] = self.add(self.gr.ers[n//2], imm_7(self.get_bits(ins_code[1], 7)).value, True)
			elif ins_code[1] == 1:
				# ADD SP, #signed8
				self.sp.value += ctypes.c_int8(immnum).value
				cycle_count = 2
			elif ins_code[1] == 3:
				# [DSR prefix] DSR <- #imm8
				dsr_prefix = True
				self.dsr.value = ins_code[1]
				cycle_count = 1
			elif ins_code[1] == 5:
				# SWI #snum
				retval = 2
			elif ins_code[1] == 9:
				# MOV PSW, #unsigned8
				self.psw.raw = immnum
				cycle_count = 1
			elif ins_code[1] == 0xb:
				if ins_code_raw[1] == 0x7f:
					# RC
					self.psw.field.c = 0
					cycle_count = 1
				elif ins_code_raw[1] == 0xf7:
					# DI
					self.psw.field.mie = 0
					cycle_count = 3
				else: retval = 1
			elif ins_code[1] == 0xd:
				if ins_code_raw[1] == 8:
					# EI
					self.psw.field.mie = 1
					cycle_count = 3
					# TODO: Disable maskable interrupts for 2 cycles
				elif ins_code_raw[1] == 0x80:
					# SC
					self.psw.field.c = 1
					cycle_count = 1
				else: retval = 1
			else: retval = 1
		elif decode_index == 0xf0:
			if ins_code_int & 0xf0 != 0: retval = 1
			else:
				# B Cadr
				self.csr.value = ins_code[1]
				self.pc.value = adr
				cycle_count = 2 + self.ea_inc_delay
		elif decode_index == 0xf1:
			if ins_code_int & 0xf0 != 0: retval = 1
			else:
				# BL Cadr
				self.lr.value = next_instruction + 2
				self.lcsr.value = self.csr.value
				self.csr.value = ins_code[1]
				self.pc.value = adr
				cycle_count = 2 + self.ea_inc_delay
		elif decode_index == 0xf2:
			if ins_code_int & 0xf10 != 0: retval = 1
			else:
				# B ERn
				self.pc.value = self.gr.ers[m//2]
				cycle_count = 2 + self.ea_inc_delay
		elif decode_index == 0xf3:
			if ins_code_int & 0xf10 != 0: retval = 1
			else:
				# BL ERn
				self.pc.value = self.gr.ers[m//2]
				self.lr.value = next_instruction
				self.lcsr.value = self.csr.value
				cycle_count = 2 + self.ea_inc_delay
		elif decode_index == 0xf4:
			if ins_code_int & 0x100 != 0: retval = 1
			else:
				# MUL ERn, Rm
				retval = 2
		elif decode_index == 0xf5:
			if ins_code_int & 0x110 != 0: retval = 1
			else:
				# MOV ERn, ERm
				retval = 2
		elif decode_index == 0xf6:
			if ins_code_int & 0x110 != 0: retval = 1
			else:
				# ADD ERn, ERm
				self.gr.ers[n//2] = self.add(self.gr.ers[n//2], self.gr.ers[m//2], True)
				cycle_count = 2
		elif decode_index == 0xf7:
			if ins_code_int & 0x110 != 0: retval = 1
			else:
				# CMP ERn, ERm
				self.sub(self.gr.ers[n//2], self.gr.ers[m//2], True)
				cycle_count = 2
		elif decode_index == 0xf9:
			if ins_code_int & 0x100 != 0: retval = 1
			else:
				# DIV ERn, Rm
				retval = 2
		elif decode_index == 0xfa:
			if ins_code_int & 0x10 != 0: retval = 1
			else:
				# LEA [ERm]
				retval = 2
		elif decode_index == 0xfb:
			if ins_code_int & 0x10 != 0: retval = 1
			else:
				# LEA Disp16[ERm]
				retval = 2
		elif decode_index == 0xfc:
			if ins_code_int & 0x10 != 0: retval = 1
			else:
				# LEA Dadr
				self.ea.value = adr
				next_instruction += 2
				cycle_count = 2
		elif decode_index == 0xfd:
			# MOV CRn, [EA]
			# MOV CRn, [EA+]
			# MOV CERn, [EA]
			# MOV CERn, [EA+]
			# MOV CXRn, [EA]
			# MOV CXRn, [EA+]
			# MOV CQRn, [EA]
			# MOV CQRn, [EA+]
			# MOV [EA], CRm
			# MOV [EA+], CRm
			# MOV [EA], CERm
			# MOV [EA+], CERm
			# MOV [EA], CXRm
			# MOV [EA+], CXRm
			# MOV [EA], CQRm
			# MOV [EA+], CQRm
			retval = 2
		elif decode_index == 0xfe:
			# PUSH/POP
			retval = 2
		elif decode_index == 0xff:
			if ins_code_int & 0x10 != 0: retval = 1
			else:
				if ins_code_int == 0xfe0f:
					# RTI
					retval = 2
				elif ins_code_int == 0xfe1f:
					# RT
					retval = 2
				elif ins_code_int == 0xfe2f:
					# INC [EA]
					seg = self.dsr.value if self.dsr_prefix else 0
					currval = self.read_dmem(self.ea.value, 1, seg)[0]
					self.write_dmem(self.ea.value, (currval+1).to_bytes(1, 'big'), seg)
				elif ins_code_int == 0xfe3f:
					# DEC [EA]
					seg = self.dsr.value if self.dsr_prefix else 0
					currval = self.read_dmem(self.ea.value, 1, seg)[0]
					self.write_dmem(self.ea.value, (currval-1).to_bytes(1, 'big'), seg)
				elif ins_code_int == 0xfe8f:
					# NOP
					cycle_count = 1
				elif ins_code_int == 0xfe9f:
					# [DSR prefix] DSR <- DSR
					dsr_prefix = True
					cycle_count = 1
				elif ins_code_int == 0xfecf:
					# CPLC
					self.psw.field.c ^= 1
					cycle_count = 1
				elif ins_code_int == 0xffff:
					# BRK
					if self.psw.field.elevel > 1: self.reset_registers()
					else:
						self.elr2.value = next_instruction
						self.ecsr2.value = self.csr.value
						self.epsw2 = self.psw.raw
						self.psw.field.elevel = 2
						self.pc.value = self.reset_brk_ptr
						cycle_count = 7
				else: retval = 1
		else: retval = 2

		if retval == 0:
			self.ea_inc_delay = 1 if ea_inc else 0

			self.mask_cycle -= cycle_count
			if self.mask_cycle < 0: self.mask_cycle = 0

			if self.dsr_prefix and self.mask_cycle == 0: self.mask_cycle += 1

		self.pc.value = next_instruction
		return retval

	def add(self, op1: int, op2: int, short: bool = False):
		ctype = ctypes.c_int16 if short else ctypes.c_int8

		result_raw = op1 + op2
		result = ctype(result_raw).value

		self.psw.field.c = int(result_raw & (0x10000 if short else 0xff))
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = (((op2 & 0x7f) + (op1 & 0x7f)) >> 7) ^ self.psw.field.c
		hc_val = 0xfff if short else 0xf
		self.psw.field.hc = int((((op2 & hc_val) + (op1 & hc_val)) & (hc_val+1)))
	
		return result

	def addc(self, op1: int, op2: int):
		result_raw = op1 + op2 + self.psw.field.c
		result = ctypes.c_int8(result_raw).value

		self.psw.field.c = int(result_raw & 0xff)
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = (((op2 & 0x7f) + (op1 & 0x7f)) >> 7) ^ self.psw.field.c
		self.psw.field.hc = int((((dest & 0xf) + (src & 0xf)) & 0x10))

		return result

	def _and(self, op1: int, op2: int):
		result = ctypes.c_int8(op1 & op2).value

		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		
		return result

	def sub(self, op1: int, op2: int, short: bool = False):
		ctype = ctypes.c_int16 if short else ctypes.c_int8

		result_raw = op1 - op2
		result = ctype(result_raw).value

		self.psw.field.c = int(result_raw & (0x10000 if short else 0xff))
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = (((op2 & 0x7f) + (op1 & 0x7f)) >> 7) ^ self.psw.field.c
		hc_val = 0xfff if short else 0xf
		self.psw.field.hc = int((((op2 & hc_val) + (op1 & hc_val)) & (hc_val+1)))

		return result

	def subc(self, op1: int, op2: int):
		result_raw = op1 - op2 - self.psw.field.c
		result = ctypes.c_short(result_raw).value

		self.psw.field.c = int(result_raw & 0xff)
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.ov = (((op2 & 0x7f) + (op1 & 0x7f)) >> 7) ^ self.psw.field.c
		self.psw.field.hc = int((((dest & 0xf) + (src & 0xf)) & 0x10))

		return result

	def daa(self, op1: int):
		split = self.split_nibble(op1)

		result = op1

		if self.psw.field.c == 0:
			if split[0] in range(10):
				if self.psw.field.hc == 0 and split[1] in range(10): result += 0
				elif self.psw.field.hc == 1: result += 6
			elif split[0] in range(9) and self.psw.field.hc == 0 and split[1] in range(10, 16): result += 6
			elif split[0] in range(10, 16):
				if self.psw.field.hc == 0 and split[1] in range(10): result += 0x60
				elif self.psw.field.hc == 1: result += 0x66
			elif split[0] in range(9, 16) and self.psw.field.hc == 0 and split[1] in range(10, 16): result += 0x66
		elif self.psw.field.c == 1:
			if self.psw.field.hc == 0:
				if split[1] in range(10): result += 0x60
				elif split[1] in range(10, 16): result += 0x66
			elif self.psw.field.hc == 1: result += 0x66

		result = ctypes.c_int8(result).value
	
		if result > 0xff: self.psw.field.c = 1
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.hc = int((((dest & 0xf) + (src & 0xf)) & 0x10))

		return result

	def das(self, op1: int):
		split = self.split_nibble(op1)

		result = op1

		if self.psw.field.c == 0:
			if split[0] in range(10):
				if self.psw.field.hc == 0 and split[1] in range(10): result -= 0
				elif self.psw.field.hc == 1: result -= 6
			elif split[0] in range(9) and self.psw.field.hc == 0 and split[1] in range(10, 16): result -= 6
			elif split[0] in range(10, 16):
				if self.psw.field.hc == 0:
					if split[1] in range(10): result -= 0x60
					elif split[1] in range(10, 16): result -= 0x66
				elif self.psw.field.hc == 1: result += 0x66
			elif split[0] in range(9, 16) and self.psw.field.hc == 0 and split[1] in range(10, 16): result -= 0x66
		elif self.psw.field.c == 1:
			if self.psw.field.hc == 0:
				if split[1] in range(10): result -= 0x60
				elif split[1] in range(10, 16): result -= 0x66
			elif self.psw.field.hc == 1: result -= 0x66

		result = ctypes.c_int8(result).value

		if result > 0xff: self.psw.field.c = 1
		self.psw.field.z = int(result == 0)
		self.psw.field.s = int(result < 0)
		self.psw.field.hc = int((((dest & 0xf) + (src & 0xf)) & 0x10))

		return result
