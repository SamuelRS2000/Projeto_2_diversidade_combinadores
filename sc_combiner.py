"""
sc_combiner.py
Implementação do combinador Selection Combining (SC).

Contém a classe base SelectionCombining com:
- Geração de envoltória (Rayleigh e Rice) com E[alpha^2] = 1;
- Regra de combinação por seleção: gamma_SC = gamma_s_med * max(alpha_l^2);
- Expressões analíticas sob desvanecimento Rayleigh (OP, SEP 4-ASK, Cerg);
- Métodos de simulação Monte Carlo.
"""

import os
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class SelectionCombining:
    """
    Classe para simulação e cálculo analítico do combinador Selection Combining (SC).
    Serve como classe base para os combinadores MRC e EGC.
    """

    def __init__(self, nome="Selection Combining (SC)"):
        self.nome = nome

    @staticmethod
    def gerar_envoltoria(L, N_amostras, Kr=0):
        """
        Gera matriz de envoltórias independentes com potência média unitária (E[alpha^2] = 1).
        
        Parâmetros:
            L (int): Número de ramos de diversidade.
            N_amostras (int): Quantidade de realizações estocásticas.
            Kr (float): Fator de Rice (Kr=0 corresponde ao canal Rayleigh).
            
        Retorna:
            alpha (np.ndarray): Matriz de envoltórias de dimensões (N_amostras, L).
        """
        if Kr == 0:
            # Canal Rayleigh puro: componentes em fase e quadratura gaussianas
            x_i = np.random.normal(0.0, np.sqrt(0.5), size=(N_amostras, L))
            x_q = np.random.normal(0.0, np.sqrt(0.5), size=(N_amostras, L))
            alpha = np.sqrt(x_i**2 + x_q**2)
        else:
            # Canal Riceano: componente especular (LoS) em quadratura e dispersão difusa
            # Normalização: s^2 = Kr / (Kr + 1), 2*sigma^2 = 1 / (Kr + 1)
            s = np.sqrt(Kr / (Kr + 1.0))
            sigma = np.sqrt(1.0 / (2.0 * (Kr + 1.0)))
            x_i = np.random.normal(s, sigma, size=(N_amostras, L))
            x_q = np.random.normal(0.0, sigma, size=(N_amostras, L))
            alpha = np.sqrt(x_i**2 + x_q**2)

        return alpha

    def combinar_snr(self, alpha, gamma_s_med):
        """
        Calcula a SNR instantânea combinada na saída do SC:
        gamma_SC = max{gamma_1, ..., gamma_L} = gamma_s_med * max{alpha_l^2} (Slide 14, Eq. 5).
        """
        alpha_max_sq = np.max(alpha**2, axis=1)
        return gamma_s_med * alpha_max_sq

    # -------------------------------------------------------------
    # Métodos Analíticos (Rayleigh)
    # -------------------------------------------------------------
    def calcular_op_teorico(self, gamma_th, gamma_s_med, L):
        """
        Probabilidade de Outage analítica sob Rayleigh (Slide 16, Eq. 9):
        P_out(gamma_th) = (1 - exp(-gamma_th / gamma_s_med))^L
        """
        gamma_th_arr = np.atleast_1d(gamma_th)
        termo = 1.0 - np.exp(-gamma_th_arr / gamma_s_med)
        termo = np.clip(termo, 0.0, 1.0)
        p_out = termo**L
        return p_out if np.ndim(gamma_th) > 0 else p_out.item()

    def calcular_cerg_teorico(self, gamma_s_med, L, Bw=1.0):
        """
        Capacidade Ergódica analítica sob Rayleigh em bits/s/Hz (Slide 20, Eq. 11):
        C_SC = (Bw * L / ln(2)) * sum_{k=0}^{L-1} binom(L-1, k) * ((-1)^k / (k+1))
               * exp((k+1) / gamma_s_med) * E_1((k+1) / gamma_s_med)
        """
        gamma_arr = np.atleast_1d(gamma_s_med).astype(float)
        capacidade = np.zeros_like(gamma_arr)

        for k in range(L):
            binom = scipy.special.comb(L - 1, k)
            coef = binom * ((-1.0)**k / (k + 1.0))
            c_k = (k + 1.0) / gamma_arr
            termo = coef * np.exp(c_k) * scipy.special.expn(1, c_k)
            capacidade += termo

        capacidade *= (Bw * L / np.log(2.0))
        return capacidade if np.ndim(gamma_s_med) > 0 else capacidade.item()

    def calcular_sep_teorico(self, gamma_s_med, L, M=4):
        """
        Probabilidade de Erro de Símbolo analítica para 4-ASK sob Rayleigh (Slide 18, Eq. 10):
        a_M = 2*(M-1)/M, b_M = 6/(M^2 - 1).
        """
        gamma_arr = np.atleast_1d(gamma_s_med).astype(float)
        a_M = 2.0 * (M - 1.0) / M
        b_M = 6.0 / (M**2 - 1.0)

        soma = np.zeros_like(gamma_arr)
        for l in range(L):
            binom = scipy.special.comb(L - 1, l)
            fator = binom * ((-1.0)**l * L) / (2.0 * (l + 1.0))
            arg = (b_M * gamma_arr) / (2.0 * (l + 1.0))
            radical = np.sqrt(arg / (1.0 + arg))
            soma += fator * (1.0 - radical)

        sep = a_M * soma
        return sep if np.ndim(gamma_s_med) > 0 else sep.item()

    # -------------------------------------------------------------
    # Métodos de Simulação Monte Carlo
    # -------------------------------------------------------------
    def simular_op(self, gamma_th_vec, gamma_s_med, L, N_amostras=2500000, Kr=0):
        """
        Simula a Probabilidade de Outage por contagem: P_out = P(gamma_comb < gamma_th).
        """
        alpha = self.gerar_envoltoria(L, N_amostras, Kr=Kr)
        gamma_comb = self.combinar_snr(alpha, gamma_s_med)

        gamma_th_arr = np.atleast_1d(gamma_th_vec)
        op_sim = np.array([np.mean(gamma_comb < th) for th in gamma_th_arr])
        return op_sim if np.ndim(gamma_th_vec) > 0 else op_sim.item()

    def simular_cerg(self, gamma_s_med_vec, L, N_amostras=1000000, Kr=0, Bw=1.0):
        """
        Simula a Capacidade Ergódica: C_erg = Bw * E[log2(1 + gamma_comb)].
        """
        alpha = self.gerar_envoltoria(L, N_amostras, Kr=Kr)
        gamma_vec = np.atleast_1d(gamma_s_med_vec)
        cerg_sim = []

        for g_med in gamma_vec:
            gamma_comb = self.combinar_snr(alpha, g_med)
            c = Bw * np.mean(np.log2(1.0 + gamma_comb))
            cerg_sim.append(c)

        res = np.array(cerg_sim)
        return res if np.ndim(gamma_s_med_vec) > 0 else res.item()

    def simular_sep(self, gamma_s_med_vec, L, N_amostras=2500000, Kr=0, M=4):
        """
        Simula a SEP calculando a média condicional de erro:
        P_e(gamma) = a_M * Q(sqrt(b_M * gamma)) = a_M * 0.5 * erfc(sqrt(b_M * gamma / 2)).
        """
        a_M = 2.0 * (M - 1.0) / M
        b_M = 6.0 / (M**2 - 1.0)

        alpha = self.gerar_envoltoria(L, N_amostras, Kr=Kr)
        gamma_vec = np.atleast_1d(gamma_s_med_vec)
        sep_sim = []

        for g_med in gamma_vec:
            gamma_comb = self.combinar_snr(alpha, g_med)
            pe_cond = a_M * 0.5 * scipy.special.erfc(np.sqrt(b_M * gamma_comb / 2.0))
            sep_sim.append(np.mean(pe_cond))

        res = np.array(sep_sim)
        return res if np.ndim(gamma_s_med_vec) > 0 else res.item()

    def gerar_graficos_validacao(self, pasta_saida="Resultados_Projeto2/SC", N_sim=2500000):
        """
        Gera as curvas de validação teórica vs. simulação para o combinador SC.
        """
        os.makedirs(pasta_saida, exist_ok=True)
        L_lista = [1, 2, 3, 4]
        cores = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

        # 1. Probabilidade de Outage
        gamma_s_op_dB = 10.0
        gamma_s_op_lin = 10.0**(gamma_s_op_dB / 10.0)
        gamma_th_dB = np.linspace(-10, 20, 200)
        gamma_th_lin = 10.0**(gamma_th_dB / 10.0)
        gamma_th_sim_dB = np.arange(-10, 20.1, 2.5)
        gamma_th_sim_lin = 10.0**(gamma_th_sim_dB / 10.0)

        plt.figure(figsize=(8.5, 6))
        for L in L_lista:
            op_theo = self.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, L)
            op_sim = self.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L, N_amostras=N_sim)
            plt.semilogy(gamma_th_dB, op_theo, color=cores[L], linewidth=1.8, label=f'L = {L} (Rayleigh Teo.)')
            mask = op_sim >= 1e-5
            plt.semilogy(gamma_th_sim_dB[mask], op_sim[mask], '+', color=cores[L], markersize=8, markeredgewidth=1.6)

        # Curva de Rice para L=4
        op_rice_sim = self.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L=4, N_amostras=N_sim, Kr=4)
        mask_r = op_rice_sim >= 1e-5
        plt.semilogy(gamma_th_sim_dB[mask_r], op_rice_sim[mask_r], 'k^--', markersize=6, linewidth=1.4,
                     label=r'L = 4 (Rice $K_R=4$)')

        plt.plot([], [], '+k', markersize=8, markeredgewidth=1.6, label='Simulação (+)')
        plt.title(f'OP, Selection Combining (SC), SNR média por ramo = {int(gamma_s_op_dB)} dB')
        plt.xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
        plt.ylabel('Probabilidade de Outage')
        plt.xlim([-10, 20])
        plt.ylim([1e-5, 1.0])
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Figura_2_OP_SC.png"), dpi=300)
        plt.close()

        # 2. SEP 4-ASK
        snr_sep_dB = np.linspace(0, 30, 200)
        snr_sep_lin = 10.0**(snr_sep_dB / 10.0)
        snr_sep_sim_dB = np.arange(0, 30.1, 2.5)
        snr_sep_sim_lin = 10.0**(snr_sep_sim_dB / 10.0)

        plt.figure(figsize=(8.5, 6))
        for L in L_lista:
            sep_theo = self.calcular_sep_teorico(snr_sep_lin, L, M=4)
            sep_sim = self.simular_sep(snr_sep_sim_lin, L, N_amostras=N_sim, M=4)
            plt.semilogy(snr_sep_dB, sep_theo, color=cores[L], linewidth=1.8, label=f'L = {L} (Rayleigh Teo.)')
            mask = sep_sim >= 1e-5
            plt.semilogy(snr_sep_sim_dB[mask], sep_sim[mask], '+', color=cores[L], markersize=8, markeredgewidth=1.6)

        # Curva de Rice para L=4
        sep_rice_sim = self.simular_sep(snr_sep_sim_lin, L=4, N_amostras=N_sim, Kr=4, M=4)
        mask_sep_r = sep_rice_sim >= 1e-5
        plt.semilogy(snr_sep_sim_dB[mask_sep_r], sep_rice_sim[mask_sep_r], 'k^--', markersize=6, linewidth=1.4,
                     label=r'L = 4 (Rice $K_R=4$)')

        plt.plot([], [], '+k', markersize=8, markeredgewidth=1.6, label='Simulação (+)')
        plt.title('SEP, Selection Combining (SC) – 4-ASK')
        plt.xlabel(r'SNR média por ramo (dB)')
        plt.ylabel('SEP')
        plt.xlim([0, 30])
        plt.ylim([1e-5, 1.0])
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Figura_3_SEP_SC.png"), dpi=300)
        plt.close()

        # 3. Capacidade Ergódica
        snr_cap_dB = np.linspace(-5, 25, 200)
        snr_cap_lin = 10.0**(snr_cap_dB / 10.0)
        snr_cap_sim_dB = np.arange(-5, 25.1, 2.5)
        snr_cap_sim_lin = 10.0**(snr_cap_sim_dB / 10.0)

        plt.figure(figsize=(8.5, 6))
        for L in L_lista:
            cap_theo = self.calcular_cerg_teorico(snr_cap_lin, L)
            cap_sim = self.simular_cerg(snr_cap_sim_lin, L, N_amostras=1000000)
            plt.plot(snr_cap_dB, cap_theo, color=cores[L], linewidth=1.8, label=f'L = {L} (Rayleigh Teo.)')
            plt.plot(snr_cap_sim_dB, cap_sim, '+', color=cores[L], markersize=8, markeredgewidth=1.6)

        # Curva de Rice para L=4
        cap_rice_sim = self.simular_cerg(snr_cap_sim_lin, L=4, N_amostras=1000000, Kr=4)
        plt.plot(snr_cap_sim_dB, cap_rice_sim, 'k^--', markersize=6, linewidth=1.4,
                 label=r'L = 4 (Rice $K_R=4$)')

        plt.plot([], [], '+k', label='Simulação (+)')
        plt.title('Capacidade ergódica, Selection Combining (SC)')
        plt.xlabel(r'SNR média por ramo (dB)')
        plt.ylabel('Capacidade ergódica (bits/s/Hz)')
        plt.xlim([-5, 25])
        plt.ylim([0, 10])
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Figura_4_Cerg_SC.png"), dpi=300)
        plt.close()

        print(f"[SC] Figuras salvas em: {pasta_saida}")


if __name__ == "__main__":
    combinador_sc = SelectionCombining()
    combinador_sc.gerar_graficos_validacao()
