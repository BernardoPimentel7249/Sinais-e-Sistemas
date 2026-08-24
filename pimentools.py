#!/usr/bin/env python
"""\
Esse arquivo contém funções que resolvem computacionalmente algumas situações de SSTC.

As funções mais importantes* são destacadas nos notebooks e podem auxiliar o estudante na resolução de exercícios.

Outras funções que, em geral, compões as funções principais usadas nos notebooks podem ser manipuladas abaixo.

* : resolver_sistema_LCIT(Q, x, conds)
1 - Funções de análise no tempo
2 - Funções de análise de Fourier
3 - Funções sobre transformada de Laplace
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display

t = sp.symbols('t')
s = sp.symbols('s')


#################
#
#       1
#
#################


def resolver_sistema_LCIT(Q, x, conds, P=[1], T=6, deslocamentos_dirac=[0]):
    # Obtém a resposta total e as funções de interesse nesse processo:
    y_z  = resolve_entrada_nula(Q, conds)
    h, H = resposta_ao_impulso(Q, P)
    y_s  = resolve_estado_nulo(x, h)
    y    = resolve_EDO(Q, x, P, conds)

    # Exibe as funções obtidas:
    funcs = {'y(t)'  : y,   'x(t)'  : x,
             'y_z(t)': y_z, 'y_s(t)': y_s,
             'h(t)'  : h}
    plota_sistema_LCIT(funcs, Q, P, H, t, T, deslocamentos_dirac)


def resolve_entrada_nula(Q, conds):
    y_z = sp.Function('y_z')(t)

    # Monta a equação para ser resolvida:
    EDO = Q[-1]*y_z
    for n in range(len(Q[:-1])):
        EDO = EDO + Q[-n - 2] * y_z.diff(t, n + 1)

    # Prepara dict de condições iniciais:
    conds_iniciais = {}
    for n in range(len(conds)):
        conds_iniciais.update({y_z.diff(t, n).subs(t, 0): conds[n]})

    # Resolve a EDO homogênea sujeita às condições iniciais:
    y_z = sp.dsolve(EDO, y_z, ics=conds_iniciais)
    y_z = y_z.rhs * sp.Heaviside(t)

    return y_z


def resposta_ao_impulso(Q, P):
    # Monta função de transferência e faz a transformada de Laplace inversa para obter a resposta ao impulso:
    Qs = 0
    Ps = 0

    for n in range(len(Q)):
        Qs += Q[-n - 1] * s**(n)
    for n in range(len(P)):
        Ps += P[-n - 1] * s**(n)

    H = Ps / Qs
    h = sp.inverse_laplace_transform(H, s, t)

    return h, H


def resolve_estado_nulo(x, h):
    # Convolução causal entre a entrada e a resposta ao impulso para obter a saída de estado nulo:
    tau = sp.symbols('tau')
    y_s = sp.integrate(h.subs(t, tau) * x.subs(t, t - tau), (tau, 0, t))

    return y_s


def plota_sistema_LCIT(funcs, Q, P, H, t, T, deslocamentos_dirac):
    # Exibe a EDO do sistema, cada função de interesse e troca o delta típico pelo delta simulado:
    y_p = sp.Function('y')(t)
    EDO = Q[-1]*y_p
    for n in range(len(Q[:-1])):
        EDO = EDO + Q[-n - 2] * y_p.diff(t, n + 1)

    x_p = sp.Function('x')(t)
    entrada = P[-1]*x_p
    for n in range(len(P[:-1])):
        entrada = entrada + P[-n - 2] * x_p.diff(t, n + 1)
    
    display(sp.simplify(sp.Eq(EDO, entrada)))

    for func in funcs:
        display(sp.Eq(sp.Symbol(func), funcs[func]))
        funcs[func] = dirac_simulado(t, funcs[func], deslocamentos_dirac)
    display(t >= 0)

    propriedades = [verifica_memoria(funcs['h(t)']),
                    verifica_causalidade(funcs['h(t)']),
                    verifica_BIBO(funcs['h(t)']),
                    verifica_Lyapunov(Q),
                    'Função de Transferência: ' + str(H)]
    
    # Versão Numérica de cada função para o plot:
    y   = sp.lambdify(t, funcs['y(t)'])
    x   = sp.lambdify(t, funcs['x(t)'])
    y_z = sp.lambdify(t, funcs['y_z(t)'])
    y_s = sp.lambdify(t, funcs['y_s(t)']) 
    h   = sp.lambdify(t, funcs['h(t)'])

    t = np.linspace(0, T, 1000)
    fig, axs = plt.subplots(3, 2, figsize=(9, 8))

    axs[0][0].plot(t, y(t),   color='purple')
    axs[0][1].plot(t, x(t),   color='magenta')
    if (isinstance(y_z(t), int) or isinstance(y_z(t), float)):
        axs[1][0].plot(t, [0] * 1000, color='red')
    else:
        axs[1][0].plot(t, y_z(t), color='red')
    axs[1][1].plot(t, y_s(t), color='blue')
    axs[2][0].plot(t, h(t),   color='orange')

    # Caixa separada para as propriedades
    axs[2][1].axis((0, T, 0, T))
    axs[2][1].set_yticks([])
    axs[2][1].set_xticks([])
    axs[2][1].spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    for i in range(len(propriedades)):
        axs[2][1].text(0, T * 0.1 * (9 - i), propriedades[i], ha='left', wrap=True)

    titulos = ["Resposta Total", "Entrada", "Resposta de Entrada Nula", 
               "Resposta de Estado Nulo", "Resposta ao Impulso", "Propriedades"]
    rotulos = ["$y$", "$x$", "$y_z$", "$y_s$", "$h$"]
    titulo = 0

    for i in range (3):
        for j in range(2):
            axs[i][j].set_title(titulos[titulo])
            if (not(i == 2 and j == 1)):
                axs[i][j].grid(True)
                axs[i][j].set_xlabel('$t$')
                axs[i][j].set_ylabel(rotulos[titulo], rotation=1)
            titulo += 1

    plt.tight_layout()
    plt.show()


def verifica_memoria(h):
    if(isinstance(h/sp.DiracDelta(t), sp.Number)):
        return "Possui Memória: Não"
    else:
        return "Possui Memória: Sim"

    
def verifica_causalidade(h):
    if(h.has(sp.Heaviside)):
        return "Causalidade: Causal"
    elif(h.subs(t, -1).simplify()):
        return "Causal: Causal"    # Teste pontual, não 100% confiável
    else:
        return "Causal: Não Causal"


def verifica_BIBO(h):
    BIBO = sp.integrate(sp.Abs(h), (t, -sp.oo, sp.oo))
    if(BIBO == sp.oo):
        return "BIBO Estabilidade: Instável"
    else:
        return "BIBO Estabilidade: Estável"
    

def verifica_Lyapunov(Q):
    # Monta o polinômio característico:
    polinomio_caracteristico = 0
    for n in range(len(Q)):
        polinomio_caracteristico += Q[-n - 1] * t**(n)
    
    # Obtém as raízes:
    raizes = sp.roots(polinomio_caracteristico)

    # Teste de instabilidade:
    for raiz in raizes:
        if (sp.re(raiz) > 0 or (sp.re(raiz) == 0 and sp.im(raiz) != 0 and raizes[raiz] > 1)):
            return "Estabilidade Assintótica: Instável"
    
    # Teste de estabilidade marginal:
    estabilidade_marginal = False
    for raiz in raizes:
        if (sp.re(raiz) == 0 and sp.Abs(sp.im(raiz)) >= 0 and raizes[raiz] == 1):
            estabilidade_marginal = True
    if (estabilidade_marginal):
        return "Estabilidade Assintótica: Marginal"

    # Opção restante: estável
    return "Estabilidade Assintótica: Estável"


def delta_dirac_gaussiano(t, sigma=0.05):
    # Simula o delta de Dirac numericamente:
    return sp.exp(-t**2/(2*sigma**2)) / (sigma*sp.sqrt(2*sp.pi))

def delta_dirac_gaussiano_derivada_n(t, s=0.05):
    # Simula as derivadas do delta de Dirac numericamente:
    return sp.exp(-(t+s)**2/(2*s**2))/(s*sp.sqrt(2*sp.pi))-sp.exp(-(t-s)**2/(2*s**2))/(s*sp.sqrt(2*sp.pi))

def dirac_simulado(t, f, deslocamentos=[0]):
    # Substitui os deltas analíticos e suas derivadas de até segunda ordem pelos deltas numéricos:
    for deslocamento in deslocamentos:
        if(f.has(sp.DiracDelta(t - deslocamento))) :
            f = f.subs(sp.DiracDelta(t - deslocamento), delta_dirac_gaussiano(t - deslocamento))
        if(f.has(sp.diff(sp.DiracDelta(t - deslocamento), t))) : 
            f = f.subs(sp.diff(sp.DiracDelta(t - deslocamento)), delta_dirac_gaussiano_derivada_n(t - deslocamento))
        if(f.has(sp.diff(sp.DiracDelta(t - deslocamento), t, t))) :
            f = f.subs(sp.diff(sp.DiracDelta(t - deslocamento), t, t), delta_dirac_gaussiano_derivada_n(t - deslocamento))
    return f


def resolve_EDO(Q, x, P, conds):
    y = sp.Function('y')(t)
    EDO = Q[-1]*y
    for n in range(len(Q[:-1])):
        EDO = EDO + Q[-n - 2] * y.diff(t, n + 1)

    entrada = P[-1]*x
    for n in range(len(P[:-1])):
        entrada = entrada + P[-n - 2] * x.diff(t, n + 1)

    conds_iniciais = {}
    for n in range(len(conds)):
        conds_iniciais.update({y.diff(t, n).subs(t, 0): conds[n]})

    y = sp.dsolve(sp.Eq(EDO, entrada), y, ics=conds_iniciais)

    return y.rhs


#################
#
#       2
#
#################


def plota_espectros_s(x, N, w_0):
    amplitude = []
    fase = []
    nw_0 = []
    T_0  = 2*np.pi / w_0

    for n in range(-N, N + 1):
        D_n = sp.integrate(x * sp.exp(-sp.I*n*w_0*t), (t, 0, T_0), conds='none') / T_0
        
        # Obtém amplitude da harmônica:
        D_n = complex(sp.N(D_n))
        re = D_n.real
        im = D_n.imag
        modulo = np.sqrt(re**2 + im**2)
        amplitude.append(modulo)
        
        # Obtém fase da harmônica:
        arg = np.angle(D_n)
        if (np.isnan(arg)):
            arg = 0
        fase.append(arg)

        # Legenda do eixo horizontal do plot:
        match n:
            case -1:
                nw_0.append(r"$-\omega_0$")
            case 0:
                nw_0.append("0")
            case 1:
                nw_0.append(r"$\omega_0$")
            case _:
                nw_0.append(str(n) + r"$\omega_0$")

    # Plot:
    fig, axs = plt.subplots(2, 1, figsize=(8, 7))
    axs[0].stem(nw_0, amplitude)
    axs[0].set_ylabel(r"$|D_n|$", rotation=1)
    axs[0].set_title("Espectro de Amplitude")

    axs[1].stem(nw_0, fase)
    axs[1].set_ylabel(r"$∠D_n$", rotation=1)
    axs[1].set_title("Espectro de Fase")
    plt.show()


def transformada_de_fourier(x, t, w):
    dt = t[1] - t[0]
    return dt*np.exp(-1j*np.outer(w, t)) @ x

def transformada_de_fourier_inversa(X, w, t):
    dw = w[1] - w[0]
    return (dw/(2*np.pi))*np.exp(1j*np.outer(t, w)) @ X


def Tx_DSB_SC(m, H, w_c, T, t, w):
    mp = m*np.cos(w_c*t)
    MP = transformada_de_fourier(mp, t, w)
    e  = m*(np.cos(w_c*t)**2)
    E  = transformada_de_fourier(e, t, w)
    y  = transformada_de_fourier_inversa(E*H, w, t)

    fig, axs = plt.subplots(4, 1, figsize=(10, 13))

    axs[0].plot(t, mp)
    axs[0].plot(t, np.cos(w_c*t)*max(m), linestyle='--', alpha=0.3)
    axs[0].set_xlabel(r'$t$')
    axs[0].set_ylabel(r'$m(t)*cos(\omega_c t)$')

    axs[1].plot(w, MP)
    axs[1].set_xlabel(r'$\omega$')
    axs[1].set_ylabel(r'$\frac{1}{2}[M(\omega + \omega_c) + M(\omega - \omega_c)]$')

    axs[2].plot(w, H)
    axs[2].plot(w, E)
    axs[2].set_xlabel(r'$\omega$')
    axs[2].set_ylabel(r'$\frac{1}{4}[M(\omega + 2\omega_c) + M(\omega - 2\omega_c)] + \frac{1}{2}M(\omega)$')

    axs[3].plot(t, y)
    axs[3].set_xlabel('$t$')
    axs[3].set_ylabel('$Sinal Recuperado$')

    for i in range(4):
        axs[i].grid(True)

    plt.show()


def plota_espectros_t(x, T, w_c, t, w):
    # Expressão da transformada:
    X = sp.integrate(x*sp.exp(-sp.I*w*t), (t, -sp.oo, sp.oo), conds='none')
    display("F{x(t)}: ",X)
    # Demais cálculos são numéricos:
    x_num = sp.lambdify(t, x)

    t_i = np.linspace(-T, T, 1000)
    w_i = np.linspace(-1.5*w_c, 1.5*w_c, 1000)

    X_num = transformada_de_fourier(x_num(t_i), t_i, w_i)

    # Obtém espectro de amplitude:
    re = X_num.real
    im = X_num.imag
    amplitude = np.sqrt(re**2 + im**2)

    # Obtém espectro de fase:
    fase = np.angle(X_num)
    fase[np.abs(X_num) < 1e-6] = np.nan
    for n in range(len(fase)):
        if (np.isnan(fase[n])):
            fase[n] = 0

    # Plot:
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), layout='constrained')
    axs[0].plot(t_i, x_num(t_i))
    axs[0].set_ylabel(r"$x(t)$", rotation=1)
    axs[0].set_title("Sinal no Tempo")

    axs[1].plot(w_i, amplitude)
    axs[1].set_ylabel(r"$|H(\omega)|$", rotation=1)
    axs[1].set_title("Espectro de Amplitude")

    axs[2].plot(w_i, fase)
    axs[2].set_ylabel(r"$∠H(\omega)$", rotation=1)
    axs[2].set_title("Espectro de Fase")

    for i in range(3):
        axs[i].grid(True)
    plt.show()


#################
#
#       3
#
#################


def plota_Laplace(x):
    # Obtém expressão da transformada, o denominador e o numerador dela:
    X = sp.laplace_transform(x, t, s, noconds=True)
    num, den = sp.fraction(X)
    num = sp.simplify(num)
    den = sp.simplify(den)

    # Obtém polos e zeros para plotar:
    z = sp.roots(num, s)
    p = sp.roots(den, s)
    zeros = list(z.keys())
    polos = list(p.keys())

    polos_unicos = []
    polos_repetidos = []
    for polo in polos:
        if p[polo] > 1:
            polos_repetidos.append(polo)
        else:
            polos_unicos.append(polo)

    re_z, im_z = [], []
    for zero in zeros:
        re_z.append(float(sp.re(zero)))
        im_z.append(float(sp.im(zero)))

    re_p_u, im_p_u = [], []
    for polo in polos_unicos:
        for n in range(p[polo]):
            re_p_u.append(float(sp.re(polo)))
            im_p_u.append(float(sp.im(polo)))

    re_p_r, im_p_r = [], []
    for polo in polos_repetidos:
        for n in range(p[polo]):
            re_p_r.append(float(sp.re(polo)))
            im_p_r.append(float(sp.im(polo)))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    l1 = ax.scatter(re_p_u, im_p_u, marker='x', s=200, color='Orange', linewidths=3)
    l2 = ax.scatter(re_p_r, im_p_r, marker='X', s=200, color='Red')
    l3 = ax.scatter(re_z, im_z, marker='o', s=150, color='Blue', linewidths=3)

    ax.grid(True)
    if ax.get_xlim()[1] < 0:
        ax.set_xlim((ax.get_xlim()[0], 0.5))
    plt.axhline(y=0, color='k', linestyle='-')
    l4 = plt.axvline(x=0, color='k', linestyle='--')
    ax.legend((l1, l2, l3, l4), ('Raízes não repetidas', 'Raízes repetidas', 'Zeros', 'Eixo imaginário'), loc='upper right', shadow=True)

    X_s = sp.symbols('X(s)') 
    display(sp.Eq(X_s, X))
    plt.show()


#################
#
#       4
#
#################

def bode(H, ganho=1, w_l=1e4):
    # Obtém expressão da transformada, o denominador e o numerador dela:
    num, den = sp.fraction(H)
    num = sp.simplify(num)
    den = sp.simplify(den)

    # Obtém polos e zeros para plotar:
    z = sp.roots(num, s)
    p = sp.roots(den, s)
    zeros = list(z.keys())
    polos = list(p.keys())

    polos_unicos = []
    polos_repetidos = []
    for polo in polos:
        if p[polo] > 1:
            polos_repetidos.append(polo)
        else:
            polos_unicos.append(polo)

    re_z, im_z = [], []
    for zero in zeros:
        re_z.append(float(sp.re(zero)))
        im_z.append(float(sp.im(zero)))

    re_p_u, im_p_u = [], []
    for polo in polos_unicos:
        for n in range(p[polo]):
            re_p_u.append(float(sp.re(polo)))
            im_p_u.append(float(sp.im(polo)))

    re_p_r, im_p_r = [], []
    for polo in polos_repetidos:
        for n in range(p[polo]):
            re_p_r.append(float(sp.re(polo)))
            im_p_r.append(float(sp.im(polo)))

    # --- DEFINIÇÃO DA ESCALA DE FREQUÊNCIA (LOGARÍTMICA) ---
    omega_l = w_l
    # Usamos logspace para evitar w = 0 (que resultaria em log10(0) = -inf)
    w = np.logspace(-2, np.log10(omega_l), 5000) 
    omega = 1j * w

    # --- DIAGRAMA EXATO ---
    H_func = sp.lambdify(s, H * ganho)
    H_val = H_func(omega)

    amp_exata = 20 * np.log10(np.abs(H_val))
    fas_exata = np.degrees(np.angle(H_val)) # Fase em graus

    # --- DIAGRAMA ASSINTÓTICO (MANUAL) ---
    # 1. Ganho em forma de Bode K_bode = lim_{s->0} s^(-N) * H(s)
    # Calculamos a contribuição inicial de ganho contínuo
    H_0 = float(sp.re((H * ganho).subs(s, 0))) if (H * ganho).subs(s, 0) != 0 else float(ganho)
    K_dB = 20 * np.log10(abs(H_0)) if H_0 != 0 else 0.0

    amp_assint = np.full_like(w, K_dB, dtype=float)
    fas_assint = np.full_like(w, 0.0 if H_0 >= 0 else -180.0, dtype=float)

    # Consolidação de polos (únicos e repetidos)
    todos_polos_re = re_p_u + re_p_r
    todos_polos_im = im_p_u + im_p_r

    # Contribuição dos Zeros
    for r, i in zip(re_z, im_z):
        if r == 0 and i == 0:  # Zero na origem
            amp_assint += 20 * np.log10(w)
            fas_assint += 90.0
        elif i == 0:  # Zero real de 1ª ordem
            w_c = abs(r)
            amp_assint += np.where(w >= w_c, 20 * np.log10(w / w_c), 0.0)
            w1, w2 = w_c / 10.0, 10.0 * w_c
            m = (w >= w1) & (w <= w2)
            m_hi = w > w2
            fas_assint[m] += 45.0 * np.log10(w[m] / w1)
            fas_assint[m_hi] += 90.0
        else:  # Zero complexo de 2ª ordem (agrupado em par)
            w_n = np.sqrt(r**2 + i**2)
            amp_assint += np.where(w >= w_n, 40 * np.log10(w / w_n), 0.0)
            fas_assint += np.where(w >= w_n, 180.0, 0.0)

    # Contribuição dos Polos
    idx = 0
    N_p = len(todos_polos_re)
    while idx < N_p:
        r, i = todos_polos_re[idx], todos_polos_im[idx]
        
        if r == 0 and i == 0:  # Polo na origem
            amp_assint -= 20 * np.log10(w)
            fas_assint -= 90.0
            idx += 1
        elif i == 0:  # Polo real de 1ª ordem
            w_c = abs(r)
            amp_assint -= np.where(w >= w_c, 20 * np.log10(w / w_c), 0.0)
            w1, w2 = w_c / 10.0, 10.0 * w_c
            m = (w >= w1) & (w <= w2)
            m_hi = w > w2
            fas_assint[m] -= 45.0 * np.log10(w[m] / w1)
            fas_assint[m_hi] -= 90.0
            idx += 1
        else:  # Polo complexo de 2ª ordem (par conjugado)
            w_n = np.sqrt(r**2 + i**2)
            amp_assint -= np.where(w >= w_n, 40 * np.log10(w / w_n), 0.0)
            fas_assint -= np.where(w >= w_n, 180.0, 0.0)
            idx += 2  # Pula o conjugado do mesmo par

    # --- PLOT COMPARATIVO ---
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))

    # Espectro de Amplitude
    axs[0].semilogx(w, amp_exata, label='Exato (Curva Real)', color='tab:blue', linewidth=2)
    axs[0].semilogx(w, amp_assint, '--r', label='Esboço Manual (Assintótico)', linewidth=1.8)
    axs[0].set_ylabel(r"$|H(j\omega)|_{dB}$")
    axs[0].set_title("Diagrama de Bode - Espectro de Amplitude")
    axs[0].grid(True, which="both", linestyle="--", alpha=0.6)
    axs[0].legend()

    # Espectro de Fase
    axs[1].semilogx(w, fas_exata, label='Exato (Curva Real)', color='tab:blue', linewidth=2)
    axs[1].semilogx(w, fas_assint, '--r', label='Esboço Manual (Assintótico)', linewidth=1.8)
    axs[1].set_xlabel(r"Frequência $\omega$ (rad/s)")
    axs[1].set_ylabel(r"Fase $\angle H(j\omega)$ (graus)")
    axs[1].set_title("Diagrama de Bode - Espectro de Fase")
    axs[1].grid(True, which="both", linestyle="--", alpha=0.6)
    axs[1].legend()

    plt.tight_layout()
    plt.show()