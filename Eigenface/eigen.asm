;Ibaoc, Christian Gabriel P - S11

global asmNorm
global asmNormDiv
global subtNorm
global covMat
global jacobify
global asmweights
global eigPart1
global eigPart2

section .data
reset dd 0.0
cycle dw 0.0
length dw 0.0
secondCycle dw 0

section .text
bits 64
default rel
extern printf


asmNorm:
	; initialization
	push rsi
	push rbp
	push rbx
	mov r11,0
	loopy:
	;mov al,[rcx]
	vmovdqu ymm1,[rcx]
	vmovdqu ymm2,[rdx]
	;add [rdx],al
	vpaddw ymm3,ymm2,ymm1
	vmovdqu [rdx],ymm3
	add rcx, 32
	add rdx, 32
	sub r8w,16
	cmp r8w,0
	ja loopy
end:
	pop rbx
	pop rbp
	pop rsi
ret

asmNormDiv:
; initialization
	push rsi
	push rbp
	push rbx

	movzx rax, dl
	cvtsi2ss xmm1, rax			; cvt 2nd param to sp float
	vbroadcastss ymm3,xmm1		; broadcast 2nd param to ymm
	vbroadcastss ymm4, [reset]	; ymm4 is 0
	mov rax, 0
	mov ax, r9w
	mov r9,0
	mov r9,rax
normalloop:
	vmovdqu ymm0, [rcx]
	vpunpcklwd ymm1, ymm0, ymm4	;ymm1 contains doubleword version of first 8 values in ymm0
	vpunpckhwd ymm2, ymm0, ymm4 ;ymm2 contains doubleword version of last 8 values in ymm0
	vcvtdq2ps ymm1,ymm1			;cvt dw to sp
	vcvtdq2ps ymm2,ymm2			;cvt dw to sp
	vdivps ymm1,ymm1,ymm3		;div
	vdivps ymm2,ymm2,ymm3		;div
	vcvtps2dq ymm1,ymm1			;cvt sp to dw
	vcvtps2dq ymm2,ymm2			;cvt sp to dw
	vpackusdw ymm7,ymm1,ymm2	;cvt 16dw to 16w
	vmovdqu ymm0, [rcx+32]
	vpunpcklwd ymm1, ymm0, ymm4	;ymm1 contains doubleword version of first 8 values in ymm0
	vpunpckhwd ymm2, ymm0, ymm4 ;ymm2 contains doubleword version of last 8 values in ymm0
	vcvtdq2ps ymm1,ymm1			;cvt dw to sp
	vcvtdq2ps ymm2,ymm2			;cvt dw to sp
	vdivps ymm1,ymm1,ymm3		;div
	vdivps ymm2,ymm2,ymm3		;div
	vcvtps2dq ymm1,ymm1			;cvt sp to dw
	vcvtps2dq ymm2,ymm2			;cvt sp to dw
	vpackusdw ymm8,ymm1,ymm2	;cvt 16dw to 16w
	vperm2i128 ymm9,ymm7,ymm8,  0b00100000
	vperm2i128 ymm10,ymm7,ymm8, 0b00110001
	vpackuswb ymm0,ymm9,ymm10
	vmovdqu [r8],ymm0
	add rcx,64
	add r8,32
	sub r9,32
	cmp r9,0
	ja normalloop
end2:
	pop rbx
	pop rbp
	pop rsi
ret

subtNorm:
; initialization
	push rsi
	push rbp
	push rbx

	subtractloopy:
	vmovdqu ymm0, [rcx]
	vmovdqu ymm1, [rdx]
	vpsubusb ymm1,ymm1,ymm0
	vmovdqu [rdx],ymm1
	add rcx,32
	add rdx,32
	sub r8,32
	cmp r8,0
	ja subtractloopy
	
	pop rbx
	pop rbp
	pop rsi
ret


covMat:
; initialization
	push rsi
	push rbp
	push rbx
	push r9
	push r10

	mov rax,0
	mov rbx,0
	mov r9,0
	mov r10, r8

	vbroadcastss ymm7,[reset]
	covLoop:
	vbroadcastss ymm4,[reset]
	vmovdqu ymm0, [rcx]
	vpunpcklbw ymm1, ymm0, ymm4	;ymm1 contains word version of first 16 values in ymm0
	vpunpckhbw ymm2, ymm0, ymm4 ;ymm2 contains word version of last 16 values in ymm0
	vmovdqu ymm0,[rdx]
	vpunpcklbw ymm3,ymm0,ymm4
	vpunpckhbw ymm4,ymm0,ymm4
	vpmaddwd ymm5,ymm1,ymm3		; 8 dws
	vpmaddwd ymm6,ymm2,ymm4		; 8 dws
	vphaddd ymm7,ymm5,ymm6		; 
	vextracti128 xmm0,ymm7,0	; extract first 4 dw from ymm7
	vextracti128 xmm1,ymm7,1
	vpextrd rax,xmm0,0
	vpextrd rbx,xmm0,1
	add rax,rbx
	add r9, rax
	vpextrd rax,xmm0,2
	vpextrd rbx,xmm0,3
	add rax,rbx
	add r9, rax
	vpextrd rax,xmm1,0
	vpextrd rbx,xmm1,1
	add rax,rbx
	add r9,rax
	vpextrd rax,xmm1,2
	vpextrd rbx,xmm1,3
	add rax,rbx
	add r9,rax
	mov rax,r9
	add rcx,32
	add rdx,32
	sub r8,32
	cmp r8,0
	ja covLoop
	push rdx
	mov rdx,0
	mov rax,r9
	div r10
	mov r9,rax
	pop rdx

	pop r10
	pop r9
	pop rbx
	pop rbp
	pop rsi
ret

jacobify:
; initialization
	push rsi
	push rbp
	push rbx
	push r10
	push rdi
	push r11
	push r12
	push r13
	sub rsp, 320
	vbroadcastss ymm0,[reset]
	vmovdqu [rsp], ymm6
	vmovdqu [rsp+32], ymm7
	vmovdqu [rsp+64], ymm8
	vmovdqu [rsp+96], ymm9
	vmovdqu [rsp+128], ymm10
	vmovdqu [rsp+160], ymm11
	vmovdqu [rsp+192], ymm11
	vmovdqu [rsp+224], ymm11
	vmovdqu [rsp+256], ymm11
	vmovdqu [rsp+288], ymm11
	push r14
	push r15

	mov r11,0
	mov r12,0
	mov r10,0
	mov r13,0
	mov rax,0
	mov al, r9b
	mov r13,rax
	mov rax,r13
	mov rbx,r13
	greaterloopy:
	mov r11,r8
	mov rax,r13
		jacobyloop:
		mov r9,r13
		vbroadcastss ymm4,[reset]
		vbroadcastss ymm5,[reset]
		mov r12,rcx
			innerloop:
				vmovdqu ymm0,[rcx]
				vmovdqu ymm2,[r8]
				vmulpd ymm3,ymm0,ymm2
				vaddpd ymm4,ymm3
				add rcx,32
				add r8,32
				sub r9,4
				cmp r9,0
				ja innerloop
		vhaddpd ymm5, ymm4,ymm5
		vextractf128 xmm0,ymm5,0
		vextractf128 xmm1,ymm5,1
		vaddpd xmm0,xmm0,xmm1
		movsd [rdx],xmm0
		add rdx,8
		mov rcx,r12
		sub rax,1
		cmp rax,0
		ja jacobyloop
	mov r8,r11
	mov rax,r13
	imul rax,32
	shr rax,2
	add rcx,rax
	sub rbx,1
	cmp rbx,0
	ja greaterloopy

	pop r15
	pop r14
	vmovdqu ymm6,[rsp]
	vmovdqu ymm7,[rsp+32]
	vmovdqu ymm8,[rsp+64]
	vmovdqu ymm9,[rsp+96]
	vmovdqu ymm10,[rsp+128]
	vmovdqu ymm11,[rsp+160]
	vmovdqu ymm12,[rsp+192]
	vmovdqu ymm13,[rsp+224]
	vmovdqu ymm14,[rsp+256]
	vmovdqu ymm15,[rsp+288]
	add rsp,320
	pop r13
	pop r12
	pop r11
	pop rdi
	pop r10
	pop rbx
	pop rbp
	pop rsi
ret

asmweights:
; initialization
	push rsi
	push rbp
	push rbx
	push r10
	push r11
	push r12
	push r13
	mov rax,0
	mov r11,0
	mov r11b,r9b
	mov r10,0
	mov r10w,[rsp+96]
	mov r12,r11
	mov rax,r12
	mov r13, r10


	thirdloopy:
		mov rax,r12
		push rcx
		secondloopy:
		mov r11,r12
		push rdx
		vbroadcastss ymm5,[reset]
		vbroadcastss ymm6,[reset]
			firstloopy:
			vmovdqu ymm0,[rcx]
			vmovdqu ymm1,[rdx]
			vmulpd ymm4,ymm0,ymm1
			vaddpd ymm5,ymm4
			sub r11,4
			add rcx,32
			add rdx,32
			cmp r11,0
			ja firstloopy
		vhaddpd ymm6,ymm5,ymm6
		vextractf128 xmm0,ymm6,0
		vextractf128 xmm1,ymm6,1
		vaddpd xmm0,xmm0,xmm1
		movsd [r8],xmm0
		add r8,8
		pop rdx
		sub rax,1
		cmp rax,0
		ja secondloopy
	mov rax,r12
	shr rax,2
	imul rax,32
	pop rcx
	add rdx,rax
	sub r10,1
	cmp r10,0
	ja thirdloopy

	pop r13
	pop r12
	pop r11
	pop r10
	pop rbx
	pop rbp
	pop rsi
ret
eigPart1:
; initialization
	push rsi
	push rbp
	push rbx
	push r10
	push r11
	push r12
	push r13
	push r14

	mov r10,0
	mov r11,0
	mov r12,0
	mov r13,0
	mov r14,0

	mov r10b,r9b				;counter reference				
	mov r12,r10					;inner counter reference
	mov r13w,[rsp+104]			;size reference
	mov r14b,[rsp+112]			;current Img number

	vmovdqu ymm0,[rcx]			;rcx is weightPtr
	vmovdqu ymm1,[rdx]			;rdx is currentImg ptr
	vmovdqu ymm3,[r8]			;r8 is weights ptr 
	mov rax,r14
	imul rax,r13
	shl rax,3
	add rdx,rax					;shift imgVector pointer to current img
	mov r11,rdx					;save rdx pointer
	
		eigloopone:
		push r13
		mov rdx,r11
		vbroadcastss ymm5,[reset]
		vbroadcastss ymm6,[reset]
			eiglooptwo:
			vmovdqu ymm0,[rcx]
			vmovdqu ymm1,[rdx]
			vmulpd ymm4,ymm1,ymm0
			vaddpd ymm5,ymm4
			add rcx,32
			add rdx,32
			sub r13,4
			cmp r13,0
			ja eiglooptwo
		vhaddpd ymm6,ymm5,ymm6
		vextractf128 xmm0, ymm6,0
		vextractf128 xmm1, ymm6,1
		vaddpd xmm0, xmm0,xmm1
		movsd xmm0,xmm0
		pop r13
		cvtsi2sd xmm1,r13
		divsd xmm0,xmm1
		movsd [r8],xmm0
		add r8,8
		sub r10,1
		cmp r10,0
		ja eigloopone

	pop r14
	pop r13
	pop r12
	pop r11
	pop r10
	pop rbx
	pop rbp
	pop rsi
ret
eigPart2:
; initialization
	push rsi
	push rbp
	push rbx
	push r10
	push r11
	push r12
	push r13

	mov r10,0
	mov r11,0
	mov r12,0
	mov r13,0

	mov r10b,r9b			;counter reference
	mov r11,r10				;internal counter reference
	mov r12w,[rsp+96]		;size reference
	mov r13b,[rsp+104]		;curr img number

	mov rax,r13
	imul rax,r12
	shl rax,3
	add r8,rax				;offset to Eigenface ptr based on img number


	part2outloopy:
	push rdx
	push r11
	vbroadcastss ymm4,[reset]
	vbroadcastss ymm5,[reset]
		part2inloopy:
		vmovdqu ymm0,[rcx]
		vmovdqu ymm1,[rdx]
		vmulpd ymm3,ymm1,ymm0
		vaddpd ymm4,ymm3
		add rcx,32
		add rdx,32
		sub r11,4
		cmp r11,0
		ja part2inloopy
	vhaddpd ymm5,ymm4,ymm5
	vextractf128 xmm0,ymm5,0
	vextractf128 xmm1,ymm5,1
	vaddpd xmm0,xmm0,xmm1
	pop r11
	cvtsi2sd xmm1,r10
	divsd xmm0,xmm1
	movsd [r8],xmm0
	pop rdx
	add r8,8
	sub r12,1
	cmp r12,0
	ja part2outloopy

	pop r13
	pop r12
	pop r11
	pop r10
	pop rbx
	pop rbp
	pop rsi
ret