# Testing: 131059513 Composite

import numpy as np
import sympy as sp


# Tonelli's Equation: https://en.wikipedia.org/w/index.php?title=Tonelli%E2%80%93Shanks_algorithm&oldid=1014170710#Tonelli%27s_algorithm_will_work_on_mod_p%5Ek
def tonelli_equation(n: int, p: int, r: int, i: int):
    prime = p**i
    tmp = p**(i-1)
    r = pow(r, tmp, prime)
    c = pow(n, (prime - 2*(tmp) + 1) // 2, prime)
    return (r*c) % prime


# Euler's criterion: https://en.wikipedia.org/wiki/Euler's_criterion
def is_quadratic_residue(n: int, p: int):
    d = pow(n, (p-1)//2, p)

    if d == 0 or d == 1:
        return True
    else:
        return False

# Tonelli-Shanks algorithm for computing solutions to a^2 = n (mod p): https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm#The_algorithm
def tonelli_shanks(n: int, p: int):
    if n % p == 0:
        return 0

    if p == 2:
        return 1      # Because n = 1 (mod p) and 1 raised to any power is still equal to 1.
    
    # From here it is assumed that n is a quadratic residue mod p, since we filtered before with Euler's criterion
    assert(is_quadratic_residue(n,p))
    
    # if p % 4 == 3 we can solve directly: (https://asfdfsaf.github.io/js-latex/index.html?ml=false&tex=\text{if } p \equiv 3 \pmod{4}%2C%0Aa^2 \equiv n \pmod{p} \Rightarrow a %3D n^{(p%2B1)%2F4} \bmod p%20) 
    if p % 4 == 3:
        R = pow(n, (p+1)//4, p)
        #print(f"Found R={R}\n")
        return R
    
    Q = p-1
    S = 0

    S = (Q & -Q).bit_length() - 1
    Q >>= S

    z = 2
    while is_quadratic_residue(z,p):
        z += 1
    
    #print (f"z={z}")

    M = S
    c = pow(z, Q, p) # non-residue
    t = pow(n, Q, p) # residue
    R = pow(n, (Q+1)//2, p) # R = n^(Q+1)/2 mod p

    #print (f"M={M}, c={c}, t={t}, R={R}")

    while t != 1:
        i = 0
        tmp = t
        while tmp != 1:
            i += 1
            tmp = pow(tmp, 2, p)
        
        b = pow(c, 2**(M-i-1), p)
        
        M = i
        c = pow(b, 2, p)
        t = (t * c) % p
        R = (R*b) % p

        #print(f"i={i}, b={b}, M={M}, c={c}, t={t}, R={R}")

    #print(f"Found R={R}\n")
    return R

# Quadratic Sieve: https://en.wikipedia.org/wiki/Quadratic_sieve
def qs(n):

    prime_table = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29,
                            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                            73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                            127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                            179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                            233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                            283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
                            353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
                            419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                            467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
                            547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
                            607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
                            661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
                            739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
                            811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
                            877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
                            947, 953, 967, 971, 977, 983, 991, 997])
    '''
    # Smoothness bound B chosen with this heuristic:
    B = min(int(np.floor(np.exp(0.5 * np.sqrt(np.log(n) * np.log(np.log(n)))))), prime_table.size)

    print(B)
    '''

    factor_base = prime_table

    print(f"bounded_primes={factor_base}\n")
    
    filter = np.zeros(factor_base.size, dtype=np.int8)

    for i in range(0, factor_base.size):
        p = factor_base[i]
        
        if is_quadratic_residue(n,int(p)):
            filter[i] = 1

    # Applying filter
    factor_base = np.array([i for i,j in zip(factor_base,filter) if j == 1])

    print(f"factor_base={factor_base}\n")

    smooth_values_count = 0
    smooth_values = np.empty(factor_base.size+1, dtype=np.int64)
    smooth_exponents = np.empty((factor_base.size+1, factor_base.size), dtype=np.int8)

    values = np.zeros((factor_base.size*30, 1), dtype=np.int64)
    
    # Calculate all base roots pre-emptevely
    roots = np.empty(factor_base.size, np.int64)
    for i in range(0, factor_base.size):
        roots[i] = tonelli_shanks(n, int(factor_base[i]))
        
        if roots[i] == 0: # Found the factor
            return factor_base[i], n // factor_base[i] 

    print(f"roots={roots}\n")

    base_sqrt = int(np.ceil(np.sqrt(n)))

    # (sqrt(n) + i)^2 - n
    for i in range(0, factor_base.size*30):
        values[i] = np.power(base_sqrt + i, 2) - n

    tmp_values = np.copy(values)
    tmp_exponents = np.zeros((factor_base.size*30, factor_base.size), dtype=np.int8)

    sieve = 1
    while smooth_values_count < factor_base.size+1:        

        # Solve (sqrt(n) + i)^2 = n (mod p) for at least π(B)+1 values
        for p in range(0, factor_base.size):
            
            if smooth_values_count >= factor_base.size+1:
                    break
            
            prime = factor_base[p] ** sieve

            R0 = tonelli_equation(n, int(factor_base[p]), int(roots[p]), sieve) # for finding R^2 = n (mod p^sieve)

            R1 = -R0 % prime

            print("R0={}, R1={}, n={}, p={}, base_sqrt={}".format(R0, R1, n, prime, base_sqrt))

            R0 = (R0 - base_sqrt) % prime
            R1 = (R1 - base_sqrt) % prime


            # Divide values[R0 + kp] by p for k = 0, 1, 2, ...
            for j in range(R0, tmp_values.size, prime):
                
                if smooth_values_count >= factor_base.size+1:
                    break

                print(f"j={j}, values_factor[j]={tmp_values[j]}, prime={prime}")
                assert tmp_values[j] % factor_base[p] == 0
                tmp_values[j] //= factor_base[p]

                # Record factor in the exponents matrix mod 2
                tmp_exponents[j,p] += 1

                if tmp_values[j] == 1:
                    print("FOUND A SMOOTH VALUE: {}".format(values[j]))
                    smooth_values[smooth_values_count] = values[j]
                    smooth_exponents[smooth_values_count, :] = tmp_exponents[j, :]
                    smooth_values_count += 1

            if R0 != R1:
                for j in range(R1, tmp_values.size, prime):

                    if smooth_values_count >= factor_base.size+1:
                        break
                    
                    print(f"j={j}, values_factor[j]={tmp_values[j]}, prime={prime}")
                    assert tmp_values[j] % factor_base[p] == 0

                    tmp_values[j] //= factor_base[p]

                    # Record factor in the exponents matrix
                    tmp_exponents[j,p] += 1

                    if tmp_values[j] == 1:
                        print("FOUND A SMOOTH VALUE: {}".format(values[j]))
                        smooth_values[smooth_values_count] = values[j]
                        smooth_exponents[smooth_values_count, :] = tmp_exponents[j, :]
                        smooth_values_count += 1

        print(f"END OF SIEVE {sieve}\n")
        sieve += 1

    print(smooth_values)

    print(f"Found {smooth_values_count} smooth values!\n")
    
    # Compute a vector of the kernel of the exponents matrix mod 2
    smooth_exponents_mod2 = smooth_exponents % 2
    nullspace = sp.Matrix(smooth_exponents_mod2.T).nullspace()
    print(f"nullspace.len={len(nullspace)}")

    r = None
    q = None

    for k in range(0, len(nullspace)):
        # Finding kernel mod 2
        nullcolumn = nullspace[k]
        denominators = [fraction.denominator for fraction in nullcolumn]
        lcm = np.lcm.reduce(denominators)
        nullcolumn *= lcm
        nullcolumn %= 2
        nullcolumn = np.array([x for x in nullcolumn])
        print(f"nullcolumn={nullcolumn}\n")

        a = 1
        final_exponents = np.zeros(factor_base.size, dtype=np.int64)

        for i in range(0, len(nullcolumn)):
            if nullcolumn[i] == 1:
                
                a = (a * int(np.sqrt( smooth_values[i] + n ))) % n
                
                for j in range(0, len(factor_base)):
                    final_exponents[j] += smooth_exponents[i,j] 

        
        print(f"final_exponents={final_exponents}\n")

        b = 1
        for p,f in zip(factor_base, final_exponents):
            b = (b * (p ** (f>>1))) % n
        
        print(f"a = {a}, b = {b}\n")
        r = np.gcd(a-b, n)
        print(f"gcd(a-b, n) = {r}")

        if r != 1 and r != n:
            q = n//r
            break

    return r, q


r,q = qs(int(input("Insert a composite number to find the factors of: ")))

print(f"r={r}, q={q}")