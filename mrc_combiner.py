"""
mrc_combiner.py
Implementação do combinador Maximal Ratio Combining (MRC).

Herda a estrutura da classe SelectionCombining e sobrescreve:
- Regra de combinação ótima: gamma_MRC = gamma_s_med * sum_{l=1}^L alpha_l^2 (Slide 24, Eq. 15-16);
- Expressões analíticas sob desvanecimento Rayleigh (distribuição Erlang/Qui-quadrada de 2L graus de liberdade):
  * Outage (Slide 26, Eq. 19);
  * SEP 4-ASK (Slide 28, Eq. 20);
  * Capacidade Ergódica (Slide 30, Eq. 22).
"""

import os
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from sc_combiner import SelectionCombining


class MaximalRatioCombining(SelectionCombining):
    """
    Classe para simulação e cálculo analítico do combinador Maximal Ratio Combining (MRC).
    Herda métodos de canal e simulação da classe SelectionCombining.
    """

    def __init__(self, nome="Maximal Ratio Combining (MRC)"):
        super().__init__(nome=nome)

    def combinar_snr(self, alpha, gamma_s_med):
        """
        Calcula a SNR instantânea combinada na saída do MRC.
        Cada ramo é ponderado pelo conjugado do canal complexo (alinhamento em fase e amplitude):
        gamma_MRC = sum_{l=1}^L gamma_l = gamma_s_med * sum_{l=1}^L alpha_l^2 (Slide 24, Eq. 15-16).
        """
        alpha_sum_sq = np.sum(alpha**2, axis=1)
        return gamma_s_med * alpha_sum_sq

    # -------------------------------------------------------------
    # Métodos Analíticos (Rayleigh)
    # -------------------------------------------------------------
    def calcular_op_teorico(self, gamma_th, gamma_s_med, L):
        """
        Probabilidade de Outage analítica sob Rayleigh (Slide 26, Eq. 19):
        P_out(gamma_th) = 1 - exp(-gamma_th / gamma_s_med) * sum_{k=0}^{L-1} (1/k!) * (gamma_th / gamma_s_med)^k
        """
        gamma_th_arr = np.atleast_1d(gamma_th).astype(float)
        razao = gamma_th_arr / gamma_s_med

        soma = np.zeros_like(razao)
        for k in range(L):
            termo = (razao**k) / scipy.special.factorial(k)
            soma += termo

        p_out = 1.0 - np.exp(-razao) * soma
        p_out = np.clip(p_out, 0.0, 1.0)
        return p_out if np.ndim(gamma_th) > 0 else p_out.item()

    def calcular_cerg_teorico(self, gamma_s_med, L, Bw=1.0):
        """
        Capacidade Ergódica analítica sob Rayleigh em bits/s/Hz (Slide 30, Eq. 22):
        C_MRC = (Bw / ln(2)) * exp(1 / gamma_s_med) * sum_{k=1}^L E_{L-k+1}(1 / gamma_s_med)
        """
        gamma_arr = np.atleast_1d(gamma_s_med).astype(float)
        inv_gamma = 1.0 / gamma_arr

        soma_expn = np.zeros_like(gamma_arr)
        for k in range(1, L + 1):
            ordem = L - k + 1
            soma_expn += scipy.special.expn(ordem, inv_gamma)

        capacidade = (Bw / np.log(2.0)) * np.exp(inv_gamma) * soma_expn
        return capacidade if np.ndim(gamma_s_med) > 0 else capacidade.item()

    def calcular_sep_teorico(self, gamma_s_med, L, M=4):
        """
        Probabilidade de Erro de Símbolo analítica para 4-ASK sob Rayleigh (Slide 28, Eq. 20):
        mu = sqrt( (b_M * gamma / 2) / (1 + b_M * gamma / 2) )
        P_s = a_M * ((1 - mu) / 2)^L * sum_{k=0}^{L-1} binom(L-1+k, k) * ((1 + mu) / 2)^k
        """
        gamma_arr = np.atleast_1d(gamma_s_med).astype(float)
        a_M = 2.0 * (M - 1.0) / M
        b_M = 6.0 / (M**2 - 1.0)

        arg = (b_M * gamma_arr) / 2.0
        mu = np.sqrt(arg / (1.0 + arg))

        fator_base = ((1.0 - mu) / 2.0)**L

        soma = np.zeros_like(gamma_arr)
        for k in range(L):
            binom = scipy.special.comb(L - 1 + k, k)
            termo = binom * (((1.0 + mu) / 2.0)**k)
            soma += termo

        sep = a_M * fator_base * soma
        return sep if np.ndim(gamma_s_med) > 0 else sep.item()

    def gerar_graficos_validacao(self, pasta_saida="Resultados_Projeto2/MRC", N_sim=2500000):
        """
        Gera as curvas de validação teórica vs. simulação para o combinador MRC.
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
        plt.title(f'OP, Maximum Ratio Combining (MRC), SNR média por ramo = {int(gamma_s_op_dB)} dB')
        plt.xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
        plt.ylabel('Probabilidade de Outage')
        plt.xlim([-10, 20])
        plt.ylim([1e-5, 1.0])
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Figura_5_OP_MRC.png"), dpi=300)
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
        plt.title('SEP, Maximum Ratio Combining (MRC) – 4-ASK')
        plt.xlabel(r'SNR média por ramo (dB)')
        plt.ylabel('SEP')
        plt.xlim([0, 30])
        plt.ylim([1e-5, 1.0])
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Figura_6_SEP_MRC.png"), dpi=300)
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
        plt.title('Capacidade ergódica, Maximum Ratio Combining (MRC)')
        plt.xlabel(r'SNR média por ramo (dB)')
        plt.ylabel('Capacidade ergódica (bits/s/Hz)')
        plt.xlim([-5, 25])
        plt.ylim([0, 10])
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Figura_7_Cerg_MRC.png"), dpi=300)
        plt.close()

        print(f"[MRC] Figuras salvas em: {pasta_saida}")


if __name__ == "__main__":
    combinador_mrc = MaximalRatioCombining()
    combinador_mrc.gerar_graficos_validacao()
