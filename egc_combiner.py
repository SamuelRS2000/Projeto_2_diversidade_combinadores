"""
egc_combiner.py
Implementação do combinador Equal Gain Combining (EGC).

Herda a estrutura da classe MaximalRatioCombining e sobrescreve:
- Regra de combinação: ramos co-faseados com pesos de módulo unitário (|w_l| = 1);
  gamma_EGC = (gamma_s_med / L) * (sum_{l=1}^L alpha_l)^2 (Slide 33, Eq. 24-25).

Sob desvanecimento Rayleigh, a soma de variáveis aleatórias Rayleigh não possui distribuição
em forma fechada elementar simples (Slide 34). O desempenho é obtido via simulação Monte Carlo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from mrc_combiner import MaximalRatioCombining


class EqualGainCombining(MaximalRatioCombining):
    """
    Classe para simulação do combinador Equal Gain Combining (EGC).
    Herda métodos de canal e simulação estocástica de Monte Carlo.
    """

    def __init__(self, nome="Equal Gain Combining (EGC)"):
        super().__init__(nome=nome)

    def combinar_snr(self, alpha, gamma_s_med):
        """
        Calcula a SNR instantânea combinada na saída do EGC (Slide 33, Eq. 25):
        gamma_EGC = (gamma_s_med / L) * (sum_{l=1}^L alpha_l)^2.
        """
        L = alpha.shape[1]
        soma_envoltorias = np.sum(alpha, axis=1)
        gamma_comb = (gamma_s_med / L) * (soma_envoltorias**2)
        return gamma_comb

    # -------------------------------------------------------------
    # Métodos Analíticos (Sem forma fechada simples para EGC sob Rayleigh)
    # -------------------------------------------------------------
    def calcular_op_teorico(self, gamma_th, gamma_s_med, L):
        """Não há expressão fechada elementar para a OP do EGC (Slide 34). Avaliado via simulação."""
        return None

    def calcular_cerg_teorico(self, gamma_s_med, L, Bw=1.0):
        """Não há expressão fechada elementar para a Cerg do EGC (Slide 34). Avaliado via simulação."""
        return None

    def calcular_sep_teorico(self, gamma_s_med, L, M=4):
        """Não há expressão fechada elementar para a SEP do EGC (Slide 34). Avaliado via simulação."""
        return None

    def gerar_graficos_validacao(self, pasta_saida="Resultados_Projeto2/EGC", N_sim=2500000):
        """
        Gera as curvas simuladas de validação para o combinador EGC.
        """
        os.makedirs(pasta_saida, exist_ok=True)
        L_lista = [1, 2, 3, 4]
        cores = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728'}

        # 1. Probabilidade de Outage Simulada
        gamma_s_op_dB = 10.0
        gamma_s_op_lin = 10.0**(gamma_s_op_dB / 10.0)
        gamma_th_sim_dB = np.arange(-10, 20.1, 2.5)
        gamma_th_sim_lin = 10.0**(gamma_th_sim_dB / 10.0)

        plt.figure(figsize=(8.5, 6))
        for L in L_lista:
            op_sim = self.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L, N_amostras=N_sim)
            mask = op_sim >= 1e-5
            plt.semilogy(gamma_th_sim_dB[mask], op_sim[mask], 's--', color=cores[L], label=f'L = {L} (Rayleigh Sim.)',
                         markersize=6, linewidth=1.6)

        # Curva de Rice para L=4
        op_rice_sim = self.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L=4, N_amostras=N_sim, Kr=4)
        mask_r = op_rice_sim >= 1e-5
        plt.semilogy(gamma_th_sim_dB[mask_r], op_rice_sim[mask_r], 'k^--', markersize=6, linewidth=1.4,
                     label=r'L = 4 (Rice $K_R=4$)')

        plt.title(f'OP, Equal Gain Combining (EGC), SNR média por ramo = {int(gamma_s_op_dB)} dB')
        plt.xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)')
        plt.ylabel('Probabilidade de Outage')
        plt.xlim([-10, 20])
        plt.ylim([1e-5, 1.0])
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "OP_EGC_Simulado.png"), dpi=300)
        plt.close()

        # 2. SEP 4-ASK Simulada
        snr_sep_sim_dB = np.arange(0, 30.1, 2.5)
        snr_sep_sim_lin = 10.0**(snr_sep_sim_dB / 10.0)

        plt.figure(figsize=(8.5, 6))
        for L in L_lista:
            sep_sim = self.simular_sep(snr_sep_sim_lin, L, N_amostras=N_sim, M=4)
            mask = sep_sim >= 1e-5
            plt.semilogy(snr_sep_sim_dB[mask], sep_sim[mask], 's--', color=cores[L],
                         label=f'L = {L} (Rayleigh Sim.)', markersize=6, linewidth=1.6)

        # Curva de Rice para L=4
        sep_rice_sim = self.simular_sep(snr_sep_sim_lin, L=4, N_amostras=N_sim, Kr=4, M=4)
        mask_sep_r = sep_rice_sim >= 1e-5
        plt.semilogy(snr_sep_sim_dB[mask_sep_r], sep_rice_sim[mask_sep_r], 'k^--', markersize=6, linewidth=1.4,
                     label=r'L = 4 (Rice $K_R=4$)')

        plt.title('SEP, Equal Gain Combining (EGC) – 4-ASK (Simulação Monte Carlo)')
        plt.xlabel(r'SNR média por ramo (dB)')
        plt.ylabel('SEP')
        plt.xlim([0, 30])
        plt.ylim([1e-5, 1.0])
        plt.grid(True, which='both', linestyle=':', alpha=0.6)
        plt.legend(loc='lower left', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "SEP_EGC_Simulado.png"), dpi=300)
        plt.close()

        # 3. Capacidade Ergódica Simulada
        snr_cap_sim_dB = np.arange(-5, 25.1, 2.5)
        snr_cap_sim_lin = 10.0**(snr_cap_sim_dB / 10.0)

        plt.figure(figsize=(8.5, 6))
        for L in L_lista:
            cap_sim = self.simular_cerg(snr_cap_sim_lin, L, N_amostras=1000000)
            plt.plot(snr_cap_sim_dB, cap_sim, 's--', color=cores[L], label=f'L = {L} (Rayleigh Sim.)',
                     markersize=6, linewidth=1.6)

        # Curva de Rice para L=4
        cap_rice_sim = self.simular_cerg(snr_cap_sim_lin, L=4, N_amostras=1000000, Kr=4)
        plt.plot(snr_cap_sim_dB, cap_rice_sim, 'k^--', markersize=6, linewidth=1.4,
                 label=r'L = 4 (Rice $K_R=4$)')

        plt.title('Capacidade ergódica, Equal Gain Combining (EGC) (Simulação Monte Carlo)')
        plt.xlabel(r'SNR média por ramo (dB)')
        plt.ylabel('Capacidade ergódica (bits/s/Hz)')
        plt.xlim([-5, 25])
        plt.ylim([0, 10])
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, "Cerg_EGC_Simulado.png"), dpi=300)
        plt.close()

        print(f"[EGC] Figuras salvas em: {pasta_saida}")


if __name__ == "__main__":
    combinador_egc = EqualGainCombining()
    combinador_egc.gerar_graficos_validacao()
