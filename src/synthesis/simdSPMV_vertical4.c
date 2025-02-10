#include <stdio.h>
#include <stdlib.h>
#include "./spmv.h"

#define DEBUGPRINT 0
//Exclusive mask   Interferes each other asm code
#define LEVEL1MASK 1  // 0:mask
#define LEVEL2MASK 0  // 0:mask

void spmv_vertical_level4(SparseMatrix *A, double *x, double *y)
{
#if LEVEL1MASK

#if DEBUGPRINT
	// debug
	// for(int i=0;i<TOTAL;i++) printf("x[%d]=%f,y[%d]=%f\n",i,x[i],i,y[i]);

	double dst0[32];
	for(int i=0;i<32;i++) dst0[i]=0.0;
	long long int *pv, v;
	pv = &v;
	int vl;
#endif
	long long int pdebug, debug;
	pdebug = (long long int)&debug;

	//pointer
	long long int *pnrow, nrow = ROWSIZE;
	long long int *pntotal, ntotal = TOTAL;
	//volatile long long int *pnrow, nrow =  1;
	volatile long long int pAnIR, pAv, pAi, px, py;

	long long int indexVertical[TOTAL];
	volatile long long int pIV;

	pAnIR = (long long int) &(A->nonzeroInRow[0]);
	pAv = (long long int) &(A->value[0][0]);
	pAi = (long long int) &(A->indexAS[0][0]);
	px = (long long int) &(x[0]);
	py = (long long int) &(y[0]);
	pIV = (long long int) indexVertical;

	// 垂直用のindexデータ
	for(int cur = 0; cur < TOTAL; cur ++) {
		// データの並び方
		// indexVert[0] = value[ 0][0], value[ 1][0], value[ 2][0], ...
		// indexVert[1] = value[16][0], value[17][0], value[18][0], ...
		// indexVert[2] = value[32][0], value[33][0], value[34][0], ...
		// indexVert[3] = value[48][0], value[49][0], value[50][0], ...
		// byte単位 

		indexVertical[cur] = sizeof(double) * ROWSIZE * cur;
#if 0
		for(int cur2 = 0; cur2 < ROWSIZE ; cur2++) {
			// indexVertical[cur][cur2] = (cur2 + (VLMAX * cur)) * sizeof(double) * ROWSIZE;
			indexVertical[cur][cur2] = 
				(sizeof(double) * TOTAL * cur) + (sizeof(double) * ROWSIZE * cur2);
		}
#if DEBUGPRINT
		printf("indexVertical[%d][ ] %d %d %d %d\n", cur, indexVertical[cur][0],
		       indexVertical[cur][1], indexVertical[cur][2], indexVertical[cur][3]);
#endif
#endif
	}

	double d00[32];
	for(int i=0;i<32;i++) d00[i]=0.0;

	//dummy
	printf("pAnIR=%x\n",pAnIR);
  
	//t0 ROWSIZE
	//t1 TOTAL
	//t2 vl
	//t3 
	//t4 j
	//a0 &A.nonzeroInRow[0]
	//a1 &A.value[0][0]
	//a2 &A.index[0][0]
	//a3 &x[0]
	//a4 &y[0]
	//a5 temporary &A.value
	//a6 temporary &A.index
	//a7 temporary &y
	//t5 temporary &y

	//v0 "0"vector
	//v1 A.value
	//v2 A.value
	//v3 A.value
	//v4 A.value
	//v5 A.index
	//v6 A.index
	//v7 A.index
	//v8 A.index
	//v9 x[index[]]
	//v10 x[index[]]
	//v11 x[index[]]
	//v12 x[index[]]
	//v13 y=A*x  
	//v14 y=A*x 
	//v15 y=A*x
	//v16 y=A*x
	//v21 indexVertical
	//v22 indexVertical
	//v23 indexVertical
	//v24 indexVertical

	//s1 32*8
	//s2 8
	//s3 vl*8byte
	//s7 debug address
	//s8 debug data

#if 0
	for(int i=0;i<75;i++){
		if((i<12)||(A->nonzeroInRow[i]==27))    printf("nrow[%d]=%d\n",i,A->nonzeroInRow[i]);
	}
#endif

	asm volatile(
		"spmv_initialize:\n"
		"ld a0, (%[pAnIR])\n"	// A.nonzeroInRow
		"ld a1, (%[pAv])\n"	// A.value
		"ld a2, (%[pAi])\n"	// A.indexAS
		"ld a3, (%[px])\n"	// X
		"ld a4, (%[py])\n"	// Y
		"ld t0, (%[pnrow])\n"	// ROWSIZE
		"ld t1, (%[ntotal])\n" // TOTAL
		//"vle64.v v0, (%[d00])\n"
		"addi s1, zero, 8\n"	// sizeof(A->value[][ROWSIZE])*vl = sizeof(double)*ROWSIZE*VL
		"mul  s1, s1,  t0\n"	//
#if VLMAX == 8 
		"li t2, 8\n"  		// vl
#elif VLMAX == 16
		"li t2, 16\n"  		// vl
#else
		"li t2, 32\n"  		// vl
#endif
		"mul  s1, s1, t2\n"	//
		"addi s2, zero, 8\n"	// sizeof(long long int)
		"li   s3, 8\n"
		"mul  s3, s3, t2\n"	// sizeof(y[0])*vl = sizeof(double)*VL
		"mv   s4, t2\n"
		"mul  s4, s4, s2\n"	// sizeof(indexVert[0][]) = sizeof(long long int)* 32
		"li   t2, 4\n"
		"mul  s1, s1, t2\n"	// sizeof(double)* ROWSIZE * vl * 4並列

		"ld s7, (%[pdebug])\n"  //debug
	       
		"mv t3, t1\n"  			// TOTAL GNZ*GNY*GNX
#if VLMAX == 8 
		"li t2, 8\n"  			// vl
#elif VLMAX == 16
		"li t2, 16\n"  			// vl
#else
		"li t2, 32\n"  			// vl
#endif
		"vsetvli t2, t2, e64, m1\n"	// vl

		"spmv_start:\n"		// 未使用のラベル

		"ld	a0, (%[pIV])\n"
		"vle64.v v21, (a0)\n"		// v21 = indexVertical
		"add     a0, a0, s4\n"		// indexVertical[][0] -> [][vl]
		"vle64.v v22, (a0)\n"		// v22 = indexVertical
		"add     a0, a0, s4\n"		// indexVertical[][vl] -> [][vl+vl]
		"vle64.v v23, (a0)\n"		// v23 = indexVertical
		"add     a0, a0, s4\n"		// indexVertical[][vl+vl] -> [][vl+vl+vl]
		"vle64.v v24, (a0)\n"		// v24 = indexVertical
		"add     a0, a0, s4\n"		// indexVertical[][vl+vl+vl] -> [][vl+vl+vl+vl]

		"mv	a7, a4\n"		// y
		"mv	t3, t1\n"  			// TOTAL GNZ*GNY*GNX
		//"li	t3, 128\n"  	// test

		"spmv_kernel:\n"	// for( i=0; i< TOTAL; i++ )

		"mv a5, a1\n"		// A.value
		"mv a6, a2\n"		// A.index

#if VLMAX == 8
		"li      t6,  64\n"		// 8* 8
#elif VLMAX == 16
		"li      t6,  128\n"		// 8*16
#else
		"li      t6,  256\n"		// 8*32
#endif
		"vfmul.vv v13, v0, v0\n"	// reset Y
		"vfmul.vv v14, v0, v0\n"	// reset Y
		"vfmul.vv v15, v0, v0\n"	// reset Y
		"vfmul.vv v16, v0, v0\n"	// reset Y


		//"mv   t4, t0\n"		// j = 33 ROWSIZE
		"li   t4, 27\n"			// j = 27 ROWSIZE test
#if VLMAX == 8 
		//"li t2, 8\n"  			// vl
#elif VLMAX == 16
		//"li t2, 16\n"  			// vl
#else
		//"li t2, 32\n"  			// vl
#endif
		//"vsetvli t2, t2, e64, m1\n"	// vl
		//"sd t2, (s7)\n"  //debug

		"spmv_kernel_vl:\n"	// for( j=0; j<ROWSIZE; j+=vl )

		"vlxei64.v  v5, (a6), v21\n"	// A.index[ 0-15] indexVerticalに沿って取り出す
		"vlxei64.v  v6, (a6), v22\n"	// A.index[16-31] indexVerticalに沿って取り出す
		"vlxei64.v  v7, (a6), v23\n"	// A.index[32-47] indexVerticalに沿って取り出す
		"vlxei64.v  v8, (a6), v24\n"	// A.index[48-63] indexVerticalに沿って取り出す

		"vlxei64.v  v1, (a5), v21\n"	// A.value[ 0-15] indexVerticalに沿って取り出す
		"vlxei64.v  v2, (a5), v22\n"	// A.value[16-31] indexVerticalに沿って取り出す
		"vlxei64.v  v3, (a5), v23\n"	// A.value[32-47] indexVerticalに沿って取り出す
		"vlxei64.v  v4, (a5), v24\n"	// A.value[48-63] indexVerticalに沿って取り出す

		"vlxei64.v  v9,  (a3), v5\n"	// x[index[ 0-15]]
		"vlxei64.v  v10, (a3), v6\n"	// x[index[16-31]]
		"vlxei64.v  v11, (a3), v7\n"	// x[index[32-47]]
		"vlxei64.v  v12, (a3), v8\n"	// x[index[48-63]]

		"vfmacc.vv  v13, v1, v9\n"	// y[i+ 0] += A[i+ 0][j]*x[index[i+ 0][j]]
		                                // y[i+ 1] += A[i+ 1][j]*x[index[i+ 1][j]]
		"vfmacc.vv  v14, v2, v10\n"	// y[i+15] += A[i+15][j]*x[index[i+15][j]]
		                                // y[i+16] += A[i+16][j]*x[index[i+16][j]]
		"vfmacc.vv  v15, v3, v11\n"	// y[i+32] += A[i+32][j]*x[index[i+32][j]]
		                                // y[i+33] += A[i+33][j]*x[index[i+33][j]]
		"vfmacc.vv  v16, v4, v12\n"	// y[i+48] += A[i+48][j]*x[index[i+48][j]]
		                                // y[i+49] += A[i+49][j]*x[index[i+49][j]]

		"add a5, a5, s2\n"	// A.value[i][j] -> [i][j+1] に移動する(8を足す)
		"add a6, a6, s2\n"	// A.index[i][j] -> [i][j+1] に移動する(8を出す)

		"addi  t4, t4, -1\n"	// j -= 1
		"bnez  t4, spmv_kernel_vl\n"

		"vse64.v v13, (a7)\n"		// y[] 保存
		"add     a7,  a7, t6\n"
		"vse64.v v14, (a7)\n"		// y[] 保存
		"add     a7,  a7, t6\n"
		"vse64.v v15, (a7)\n"		// y[] 保存
		"add     a7,  a7, t6\n"
		"vse64.v v16, (a7)\n"		// y[] 保存
		"add     a7,  a7, t6\n"

		"add a1, a1, s1\n"		// A.value[i][j] -> [i+vl][j] に移動する
		"add a2, a2, s1\n"		// A.index[i][j] -> [i+vl][j] に移動する

		"sub  t3, t3, t2\n"		// i -= vl
		"sub  t3, t3, t2\n"		// i -= vl
		"sub  t3, t3, t2\n"		// i -= vl
		"sub  t3, t3, t2\n"		// i -= vl 4並列だからVL*4を引く
		"bnez t3, spmv_kernel\n"

		// ループ終了
		"spmv_kernel_vl_end:\n"

		//"addi s8, s8, 1\n"   //debug count

		// ループ終了
		"spmv_kernel_end:\n"

		//"sd s8, (%[pdebug])\n"  //debug
		"spmv_finish:\n"
		::[pAnIR]"r"(&pAnIR),[pAv]"r"(&pAv),[pAi]"r"(&pAi)
		 ,[px]"r"(&px),[py]"r"(&py),[pnrow]"r"(&nrow),[ntotal]"r"(&ntotal)
		 ,[pIV]"r"(&pIV),[pdebug]"r"(&pdebug)
		:"t0","t1","t2","t3","t4","t5"
		 ,"a0","a1","a2","a3","a4","a5","a6","a7"
		 ,"s1","s2","s3","s8"
		);
	exit(1);
  
#if DEBUGPRINT

	asm volatile("vse64.v v1, (%0)"::"r"(&dst0[0]));
	for(int i=0;i<VLMAX;i++) printf("v1[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v2, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v2[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v3, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v3[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v4, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v4[%d]=%f\n",i,dst0[i]);

	asm volatile("vse64.v v5, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v5[%d]=%d\n",i,dst0[i]);
	asm volatile("vse64.v v6, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v6[%d]=%d\n",i,dst0[i]);
	asm volatile("vse64.v v7, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v7[%d]=%d\n",i,dst0[i]);
	asm volatile("vse64.v v8, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v8[%d]=%d\n",i,dst0[i]);

	asm volatile("vse64.v v9, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v9[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v10, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v10[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v11, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v11[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v12, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v12[%d]=%f\n",i,dst0[i]);

	asm volatile("vse64.v v13, (%0)"::"r"(&dst0[0]));
	for(int i=0;i<VLMAX;i++) printf("v13[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v14, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v14[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v15, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v15[%d]=%f\n",i,dst0[i]);
	asm volatile("vse64.v v16, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v16[%d]=%f\n",i,dst0[i]);

	asm volatile("vse64.v v21, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v21[%d]=%d\n",i,dst0[i]);
	asm volatile("vse64.v v22, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v22[%d]=%d\n",i,dst0[i]);
	asm volatile("vse64.v v23, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v23[%d]=%d\n",i,dst0[i]);
	asm volatile("vse64.v v24, (%0)"::"r"(dst0));
	for(int i=0;i<VLMAX;i++) printf("v24[%d]=%d\n",i,dst0[i]);

	printf("debug value  s8=%d\n",debug);

	for(int i=0;i<TOTAL;i++)printf("y[%d]=%f\n",i,y[i]);
  
#endif

#endif //LEVEL1MASK
 
}

