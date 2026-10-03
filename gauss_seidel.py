import numpy as np

def get_input():
    print("=" * 60)
    print(" METODE GAUSS-SEIDEL - SISTEM PERSAMAAN LINEAR")
    print("=" * 60)
    
    n = int(input("Jumlah variabel/persamaan: "))
    
    print("\nMasukkan koefisien matriks A (pisahkan angka dengan spasi):")
    A = []
    for i in range(n):
        row = list(map(float, input(f"Baris {i+1}: ").split()))
        while len(row) != n:
            print(f"Jumlah elemen harus {n}. Silakan masukkan ulang.")
            row = list(map(float, input(f"Baris {i+1}: ").split()))
        A.append(row)
    A = np.array(A, dtype=float)
    
    print("\nMasukkan vektor b (pisahkan angka dengan spasi):")
    b_input = list(map(float, input("b = ").split()))
    while len(b_input) != n:
        print(f"Jumlah elemen harus {n}. Silakan masukkan ulang.")
        b_input = list(map(float, input("b = ").split()))
    b = np.array(b_input, dtype=float)
    
    tol_input = input("\nToleransi error (tekan Enter untuk default 1e-5): ")
    tol = float(tol_input) if tol_input.strip() != "" else 1e-5
    
    return A, b, n, tol

def run_gauss_seidel():
    A, b, n, tol = get_input()
    max_iter = 100
    C = np.zeros(n)
    
    print("\n" + "=" * 62)
    print(" HASIL ITERASI GAUSS-SEIDEL")
    print("=" * 62)
    
    header = f"{'Iterasi':<8}" + "".join([f"{f'C{i+1}':<14}" for i in range(n)]) + f"{'Error':<12}"
    print(header)
    print("-" * len(header))
    
    row_init = f"{0:<8}" + "".join([f"{C[i]:<14.5f}" for i in range(n)]) + f"{'-':<12}"
    print(row_init)
    
    for k in range(1, max_iter + 1):
        C_old = C.copy()
        
        for i in range(n):
            sigma = sum(A[i][j] * C[j] for j in range(n) if j != i)
            C[i] = (b[i] - sigma) / A[i][i]
            
        err = np.max(np.abs(C - C_old))
        
        row_str = f"{k:<8}" + "".join([f"{C[i]:<14.5f}" for i in range(n)]) + f"{err:<12.2e}"
        print(row_str)
        
        if err < tol:
            print("-" * len(header))
            print(f"--> Konvergen pada iterasi ke-{k}!")
            break

if __name__ == "__main__":
    run_gauss_seidel()
