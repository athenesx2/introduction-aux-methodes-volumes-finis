from scipy.sparse import coo_matrix
import numpy as np


def InTrap(xi, fxi):
    integral = 0.0
    for k in range(len(xi) - 1):
        integral += (fxi[k + 1] + fxi[k]) * (xi[k + 1] - xi[k]) / 2

    return integral


def vf_laplacien_dirh(maillage , donnees ):
    N = maillage["nb_vol"]
    i = 0
    dist = maillage["dist"]
    M = coo_matrix(([], ([], [])), shape=(N, N))
    x = np.zeros((N, 2))

    for k in range(N + 1):
        data = []
        row = []
        col = []
        if "s_KL" in maillage:
            K = maillage["s_KL"][k, 0]
            L = maillage["s_KL"][k, 1]
        else:
            if k == 0:
                K = 0
                L = -1
            else:
                K = k - 1
                L = (k + 1) % (N + 1) - 1

        data.append(1 / dist[k])
        row.append(K)
        col.append(K)
        x[K, 0] += maillage["sommets"][k]
        if L != -1:
            x[L, 1] += maillage["sommets"][k]
            data.append(-1 / (dist[k]))
            row.append(K)
            col.append(L)

            data.append(-1 / dist[k])
            row.append(L)
            col.append(K)

            data.append((1 / dist[k]))
            row.append(L)
            col.append(L)
        Mi = coo_matrix((data, (row, col)), shape=(N, N))
        M += Mi
        dense_mat = Mi.toarray()
    dense_mat = M.toarray()

    b = np.zeros((N))
    f = donnees["source"]
    for k in range(N):
        Ki = np.linspace(x[k, 1], x[k, 0], N)

        b[k] = InTrap(Ki, f(Ki))
    if "dirac" in donnees:
        i = 0
        K = ""
        while K == "":
            if "s_KL" in maillage:
                k = maillage["s_KL"][i, 0]
                l = maillage["s_KL"][i, 1]
            else:
                if i == 0:
                    k = 0
                    l = -1
                else:
                    k = i - 1
                    l = (i + 1) % (N + 1) - 1
            if l != -1:
                if maillage["centres"][k] < donnees["dirac"] <= maillage["centres"][l]:
                    K = k
                    L = l
                    I = i
            i += 1
        b[K] += (maillage["centres"][L] - donnees["dirac"]) / dist[I]
        b[L] += (donnees["dirac"] - maillage["centres"][K]) / dist[I]
    return M, b


def vf_laplacien_dirnh(maillage , donnees ):
    N = maillage["nb_vol"]
    i = 0
    dist = maillage["dist"]
    M = coo_matrix(([], ([], [])), shape=(N, N))
    x = np.zeros((N, 2))

    b = np.zeros((N))
    for k in range(N + 1):
        data = []
        row = []
        col = []
        if "s_KL" in maillage:
            K = maillage["s_KL"][k, 0]
            L = maillage["s_KL"][k, 1]
        else:
            if k == 0:
                K = 0
                L = -1
            else:
                K = k - 1
                L = (k + 1) % (N + 1) - 1
        data.append(1 / dist[k])
        row.append(K)
        col.append(K)
        x[K, 0] += maillage["sommets"][k]
        if L != -1:
            x[L, 1] += maillage["sommets"][k]
            data.append(-1 / (dist[k]))
            row.append(K)
            col.append(L)

            data.append(-1 / dist[k])
            row.append(L)
            col.append(K)

            data.append((1 / dist[k]))
            row.append(L)
            col.append(L)
        else:

            b[K] += donnees["bordD"](maillage["sommets"][k]) / dist[k]
        Mi = coo_matrix((data, (row, col)), shape=(N, N))
        M += Mi
        dense_mat = Mi.toarray()
    dense_mat = M.toarray()

    f = donnees["source"]
    for k in range(N):
        Ki = np.linspace(x[k, 1], x[k, 0], N)

        b[k] += InTrap(Ki, f(Ki))

    if "dirac" in donnees:
        i = 0
        K = ""
        while K == "":
            if "s_KL" in maillage:
                k = maillage["s_KL"][i, 0]
                l = maillage["s_KL"][i, 1]
            else:
                if i == 0:
                    k = 0
                    l = -1
                else:
                    k = i - 1
                    l = (i + 1) % (N + 1) - 1
            if l != -1:
                if maillage["centres"][k] < donnees["dirac"] <= maillage["centres"][l]:
                    K = k
                    L = l
                    I = i
            i += 1
        b[K] += (maillage["centres"][L] - donnees["dirac"]) / dist[I]
        b[L] += (donnees["dirac"] - maillage["centres"][K]) / dist[I]
    return M, b


def vf_diffusion_dirnh(maillage , donnees ):
    N = maillage["nb_vol"]
    i = 0
    dist = maillage["dist"]
    M = coo_matrix(([], ([], [])), shape=(N, N))
    x = np.zeros((N, 2))

    b = np.zeros((N))
    for k in range(N + 1):

        data = []
        row = []
        col = []
        if "s_KL" in maillage:
            K = maillage["s_KL"][k, 0]
            L = maillage["s_KL"][k, 1]
        else:
            if k == 0:
                K = 0
                L = -1
            else:
                K = k - 1
                L = (k + 1) % (N + 1) - 1
        if "methode" in donnees:
            if donnees["methode"] == "exacte":
                coefk = donnees["coeff_k"](maillage["sommets"][k])
            elif donnees["methode"] == "harmonique":
                if L == -1:
                    coefk = donnees["coeff_k"](maillage["sommets"][k])
                else:

                    coefk = (
                        donnees["coeff_k"](maillage["centres"][K])
                        * donnees["coeff_k"](maillage["centres"][L])
                        * (maillage["centres"][L] - maillage["centres"][K])
                        / (
                            donnees["coeff_k"](maillage["centres"][L])
                            * (maillage["sommets"][k] - maillage["centres"][K])
                            + donnees["coeff_k"](maillage["centres"][K])
                            * (maillage["centres"][L] - maillage["sommets"][k])
                        )
                    )

            elif donnees["methode"] == "arithmetique":
                if L == -1:
                    coefk = donnees["coeff_k"](maillage["sommets"][k])
                else:

                    coefk = (
                        donnees["coeff_k"](maillage["centres"][L])
                        + donnees["coeff_k"](maillage["centres"][K])
                    ) / 2

        else:
            coefk = donnees["coeff_k"](maillage["sommets"][i + 1])
        data.append(coefk / dist[k])
        row.append(K)
        col.append(K)
        x[K, 0] += maillage["sommets"][k]
        if L != -1:
            x[L, 1] += maillage["sommets"][k]
            data.append(-coefk / (dist[k]))
            row.append(K)
            col.append(L)

            data.append(-coefk / dist[k])
            row.append(L)
            col.append(K)

            data.append((coefk / dist[k]))
            row.append(L)
            col.append(L)
        else:

            b[K] += (
                donnees["bordD"](maillage["sommets"][k])
                * donnees["coeff_k"](maillage["sommets"][k])
                / dist[k]
            )
        Mi = coo_matrix((data, (row, col)), shape=(N, N))
        M += Mi
        dense_mat = Mi.toarray()
    dense_mat = M.toarray()
    f = donnees["source"]
    for k in range(N):
        Ki = np.linspace(x[k, 1], x[k, 0], N)

        b[k] += InTrap(Ki, f(Ki))
    if "dirac" in donnees:
        i = 0
        K = ""
        while K == "":
            if "s_KL" in maillage:
                k = maillage["s_KL"][i, 0]
                l = maillage["s_KL"][i, 1]
            else:
                if i == 0:
                    k = 0
                    l = -1
                else:
                    k = i - 1
                    l = (i + 1) % (N + 1) - 1
            if l != -1:
                if maillage["centres"][k] < donnees["dirac"] <= maillage["centres"][l]:
                    K = k
                    L = l
                    I = i
            i += 1
        b[K] += (maillage["centres"][L] - donnees["dirac"]) / dist[I]
        b[L] += (donnees["dirac"] - maillage["centres"][K]) / dist[I]
    return M, b


def vf_diffusion_neunh(maillage , donnees ):
    N = maillage["nb_vol"]
    i = 0
    mes = maillage["mes"]
    dist = maillage["dist"]
    data = [0]
    row = [0]
    col = [0]

    for i in range(N - 1):
        if "methode" in donnees:
            if donnees["methode"] == "exacte":
                k = donnees["coeff_k"](maillage["sommets"][i + 1])
            elif donnees["methode"] == "harmonique":
                k = (
                    donnees["coeff_k"](maillage["centres"][i + 1])
                    * donnees["coeff_k"](maillage["centres"][i])
                    * (maillage["centres"][i + 1] - maillage["centres"][i])
                    / (
                        donnees["coeff_k"](maillage["centres"][i + 1])
                        * (maillage["sommets"][i] - maillage["centres"][i])
                        + donnees["coeff_k"](maillage["centres"][i])
                        * (maillage["centres"][i + 1] - maillage["sommets"][i])
                    )
                )
            elif donnees["methode"] == "arithmetique":
                k = (
                    donnees["coeff_k"](maillage["centres"][i + 1])
                    + donnees["coeff_k"](maillage["centres"][i])
                ) / 2
        else:
            k = donnees["coeff_k"](maillage["sommets"][i + 1])

        data[-1] += k / dist[i + 1]

        data.append(-k / (dist[i + 1]))
        row.append(i)
        col.append(i + 1)

        data.append(-k  / dist[i + 1])
        row.append(i + 1)
        col.append(i)

        data.append(k / dist[i + 1])
        row.append(i + 1)
        col.append(i + 1)
    data[0] = 1
    data[1] = 0
    M = coo_matrix((data, (row, col)), shape=(N, N))
    b = np.zeros((N))
    x = maillage["sommets"]
    f = donnees["source"]
    for i in range(N):
        Ki = np.linspace(x[i], x[i + 1], N)

        b[i] = InTrap(Ki, f(Ki)) 
    b[0] -= donnees["bordN"](0) 
    b[N - 1] += donnees["bordN"](1) 
    return M, b


def vf_diffusion_fournh(maillage , donnees ):
    N = maillage["nb_vol"]
    i = 0
    mes = maillage["mes"]
    dist = maillage["dist"]
    data = [donnees["coeffF"]]
    row = [0]
    col = [0]

    for i in range(N - 1):
        if "methode" in donnees:
            if donnees["methode"] == "exacte":
                k = donnees["coeff_k"](maillage["sommets"][i + 1])
            elif donnees["methode"] == "harmonique":
                k = (
                    donnees["coeff_k"](maillage["centres"][i + 1])
                    * donnees["coeff_k"](maillage["centres"][i])
                    * (maillage["centres"][i + 1] - maillage["centres"][i])
                    / (
                        donnees["coeff_k"](maillage["centres"][i + 1])
                        * (maillage["sommets"][i] - maillage["centres"][i])
                        + donnees["coeff_k"](maillage["centres"][i])
                        * (maillage["centres"][i + 1] - maillage["sommets"][i])
                    )
                )
            elif donnees["methode"] == "arithmetique":
                k = (
                    donnees["coeff_k"](maillage["centres"][i + 1])
                    + donnees["coeff_k"](maillage["centres"][i])
                ) / 2
        else:
            k = donnees["coeff_k"](maillage["sommets"][i + 1])

        data[-1] += k * (1 / dist[i + 1])

        data.append(-k / (dist[i + 1]))
        row.append(i)
        col.append(i + 1)

        data.append(-k / dist[i + 1])
        row.append(i + 1)
        col.append(i)

        data.append(k / dist[i + 1])
        row.append(i + 1)
        col.append(i + 1)

    data[-1] -= donnees["coeffF"]

    M = coo_matrix((data, (row, col)), shape=(N, N))
    b = np.zeros((N))
    x = maillage["sommets"]
    f = donnees["source"]
    for i in range(N):
        Ki = np.linspace(x[i], x[i + 1], N)
        b[i] = InTrap(Ki, f(Ki))
    b[0] -= donnees["bordF"](0)
    b[-1] += donnees["bordF"](1)
    return M, b


def vf_laplacien_probg(maillage , donnees , choix):
    N = maillage["nb_vol"]
    i = 0
    mes = maillage["mes"]
    dist = maillage["dist"]
    ca = donnees["coeffa"]
    cb = donnees["coeffb"]
    row = [0]
    col = [0]
    if choix == 1:
        data = [1 / (dist[0])]
        for i in range(N - 1):
            data[-1] += 1 / dist[i + 1] + cb * mes[i]

            data.append(-1 / (dist[i + 1]) + ca / 2)
            row.append(i)
            col.append(i + 1)

            data.append(-1 / dist[i + 1] - ca / 2)
            row.append(i + 1)
            col.append(i)

            data.append((1 / dist[i + 1]))
            row.append(i + 1)
            col.append(i + 1)

        data[-1] += 1 / dist[N] + cb * mes[N - 1]
    elif choix == 2:
        data = [1 / (dist[0]) + cb * mes[0] / 2]
        for i in range(N - 1):
            data[-1] += 1 / dist[i + 1] + cb * mes[i] / 2 + ca

            data.append(-1 / (dist[i + 1]))
            row.append(i)
            col.append(i + 1)

            data.append(-1 / dist[i + 1] - ca)
            row.append(i + 1)
            col.append(i)

            data.append((1 / dist[i + 1]) + cb * mes[i + 1] / 2)
            row.append(i + 1)
            col.append(i + 1)

        data[-1] += 1 / dist[N] + cb * mes[N - 1] / 2 + ca
    if choix == 3:
        data = [1 / (dist[0]) + cb * mes[0] / 2-ca]
        for i in range(N - 1):
            data[-1] += 1 / dist[i + 1] + cb * mes[i] / 2

            data.append(-1 / (dist[i + 1]) + ca)
            row.append(i)
            col.append(i + 1)

            data.append(-1 / dist[i + 1])
            row.append(i + 1)
            col.append(i)

            data.append((1 / dist[i + 1]) + cb * mes[i + 1] / 2 - ca)
            row.append(i + 1)
            col.append(i + 1)

        data[-1] += 1 / dist[N] + cb * mes[N - 1] / 2
    M = coo_matrix((data, (row, col)), shape=(N, N))
    b = np.zeros((N))
    x = maillage["sommets"]
    f = donnees["source"]
    for i in range(N):
        Ki = np.linspace(x[i], x[i + 1], N)
        b[i] = InTrap(Ki, f(Ki))

    if "dirac" in donnees:
        k = 0
        while donnees["dirac"] > maillage["centres"][k]:
            k += 1

        b[k - 1] += (maillage["sommets"][k + 1] - donnees["dirac"]) / dist[k]
        b[k] += (donnees["dirac"] - maillage["sommets"][k]) / dist[k]
    b[0] += donnees["bordD"](0) / dist[0]
    b[-1] += donnees["bordD"](1) / dist[-1]
    return M, b
