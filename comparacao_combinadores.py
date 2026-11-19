"""
comparacao_combinadores.py
Módulo de análise comparativa de combinadores de diversidade (SC, MRC, EGC).

Executa:
- Impressão das tabelas comparativas de ganho em SNR média e métricas de desempenho;
- Item 1: Curvas analíticas e simuladas sob Rayleigh para L in {1, 2, 4} (OP, SEP 4-ASK, Cerg);
- Item 2: Comparação de desempenho direta entre SC, MRC e EGC para L = 4 (Slide 37);
- Item 3: Razão com diversidade / sem diversidade (L = 4 vs. L = 1) para Rayleigh e Rice
          com Kr in {2, 6, 10}, demonstrando a proposição do Slide 11.
"""

import os
import sys
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# Garante a importação dos módulos locais na mesma pasta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sc_combiner import SelectionCombining
from mrc_combiner import MaximalRatioCombining
from egc_combiner import EqualGainCombining


class AnaliseDiversidade:
    """
    Classe gerenciadora de experimentos comparativos entre as técnicas de diversidade.
    """

    def __init__(self, pasta_resultados="Resultados_Projeto2"):
        self.pasta_resultados = pasta_resultados
        os.makedirs(self.pasta_resultados, exist_ok=True)

        self.sc = SelectionCombining()
        self.mrc = MaximalRatioCombining()
        self.egc = EqualGainCombining()

    # =========================================================================
    # TABELAS COMPARATIVAS
    # =========================================================================
    def imprimir_tabelas_comparativas(self):
        """
        Imprime as tabelas comparativas no terminal:
        1. Ganho em SNR média linear e em dB até L = 1024 ramos (Rayleigh);
        2. Métricas de desempenho para L = 4 (OP, SEP 4-ASK e Cerg) em Rayleigh e Riceano.
        """
        print("\n" + "=" * 80)
        print(" TABELA 1: GANHO EM SNR MÉDIA (Rayleigh) - Slide 6 / Slide 37")
        print(" Expressões teóricas:")
        print("   - MRC: E[gamma] = L * gamma_s              => Ganho = 10*log10(L)")
        print("   - SC:  E[gamma] = gamma_s * sum(1/k, k=1..L)=> Ganho = 10*log10(sum(1/k))")
        print("   - EGC: E[gamma] = gamma_s * [1 + (L-1)*pi/4]=> Ganho = 10*log10(1 + (L-1)*pi/4)")
        print("=" * 80)

        L_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        sc_g = [10.0 * np.log10(sum(1.0 / k for k in range(1, L + 1))) for L in L_vals]
        mrc_g = [10.0 * np.log10(L) for L in L_vals]
        egc_g = [10.0 * np.log10(1.0 + (L - 1) * np.pi / 4.0) for L in L_vals]

        header = f"{'L':>6} | {'Ganho SC':>10} | {'Ganho EGC':>10} | {'Ganho MRC':>10} | {'SC - MRC':>10} | {'EGC - MRC':>10} | {'SC - EGC':>10}"
        print(header)
        print("-" * len(header))

        for L, sc, egc, mrc in zip(L_vals, sc_g, egc_g, mrc_g):
            diff_sc_mrc = sc - mrc
            diff_egc_mrc = egc - mrc
            diff_sc_egc = sc - egc
            print(f"{L:6d} | {sc:+9.2f} dB | {egc:+9.2f} dB | {mrc:+9.2f} dB | {diff_sc_mrc:+9.2f} dB | {diff_egc_mrc:+9.2f} dB | {diff_sc_egc:+9.2f} dB")

        print("=" * 80)
        print(" Limite assintótico para L -> infinito:")
        print("   EGC - MRC -> 10*log10(pi / 4) = -1.05 dB (perda máxima assintótica)")
        print("   SC - MRC  -> -infinito (SC ganha apenas logaritmicamente, ln(L)/L -> 0)")
        print("=" * 80)

        print("\n" + "=" * 80)
        print(" TABELA 2: COMPARAÇÃO DE DESEMPENHO DOS COMBINADORES (L = 4)")
        print("=" * 80)
        header_l4 = f"{'Combinador':<15} | {'Limiar OP (10^-3)':<18} | {'SNR SEP (10^-3)':<17} | {'Capacidade (10 dB)':<20} | {'Perda vs MRC':<12}"
        print(header_l4)
        print("-" * len(header_l4))
        print(f"{'MRC (Ótimo)':<15} | {'6.3 dB':<18} | {'11.6 dB':<17} | {'5.25 bps/Hz':<20} | {'Referência':<12}")
        print(f"{'EGC (Simulado)':<15} | {'5.4 dB':<18} | {'12.5 dB':<17} | {'5.05 bps/Hz':<20} | {'-0.76 a -0.9 dB':<12}")
        print(f"{'SC (Seleção)':<15} | {'2.8 dB':<18} | {'14.9 dB':<17} | {'4.28 bps/Hz':<20} | {'-2.83 a -3.3 dB':<12}")
        print("=" * 80)
        print(" Hierarquia de Desempenho: MRC > EGC > SC (Slide 37)")
        print("=" * 80 + "\n")

    # =========================================================================
    # ITEM 1: Desempenho sob Rayleigh para diferentes ordens de diversidade L
    # =========================================================================
    def executar_item_1(self, L_lista=[1, 2, 4], N_amostras=2500000):
        """
        Gera curvas analíticas e de simulação sob desvanecimento Rayleigh para
        cada uma das três métricas (OP, SEP e Cerg), avaliando o impacto de L.
        """
        print("\n--- Executando Item 1: Métricas sob Rayleigh com variação de L ---")
        cores_L = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c'}

        # 1.1 Probabilidade de Outage (OP)
        gamma_s_op_dB = 10.0
        gamma_s_op_lin = 10.0**(gamma_s_op_dB / 10.0)
        gamma_th_dB = np.linspace(-10, 20, 200)
        gamma_th_lin = 10.0**(gamma_th_dB / 10.0)
        gamma_th_sim_dB = np.arange(-10, 20.1, 2.5)
        gamma_th_sim_lin = 10.0**(gamma_th_sim_dB / 10.0)

        fig, axs = plt.subplots(1, 3, figsize=(15, 6.8))
        for L in L_lista:
            # SC
            axs[0].semilogy(gamma_th_dB, self.sc.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, L),
                            color=cores_L[L], linestyle=':', linewidth=2.0, label=f'L={L} (Teo)')
            op_sc_sim = self.sc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L, N_amostras=N_amostras)
            mask_sc = op_sc_sim >= 1e-5
            axs[0].semilogy(gamma_th_sim_dB[mask_sc], op_sc_sim[mask_sc], '+', color=cores_L[L],
                            markersize=8, markeredgewidth=1.8)

            # MRC
            axs[1].semilogy(gamma_th_dB, self.mrc.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, L),
                            color=cores_L[L], linewidth=2.0, label=f'L={L} (Teo)')
            op_mrc_sim = self.mrc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L, N_amostras=N_amostras)
            mask_mrc = op_mrc_sim >= 1e-5
            axs[1].semilogy(gamma_th_sim_dB[mask_mrc], op_mrc_sim[mask_mrc], '+', color=cores_L[L],
                            markersize=8, markeredgewidth=1.8)

            # EGC (Puramente Simulado)
            op_egc_sim = self.egc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L, N_amostras=N_amostras)
            mask_egc = op_egc_sim >= 1e-5
            axs[2].semilogy(gamma_th_sim_dB[mask_egc], op_egc_sim[mask_egc], 's--', color=cores_L[L],
                            linewidth=1.8, markersize=6, markeredgewidth=1.4, label=f'L={L} (Sim)')

        for idx, (ax, tit) in enumerate(zip(axs, ['SC (Eq. 9)', 'MRC (Eq. 19)', 'EGC (Simulação)'])):
            ax.set_title(f'Probabilidade de Outage - {tit}', fontsize=12.5, fontweight='bold', pad=8)
            ax.set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)', fontsize=11.5)
            ax.set_ylabel('Probabilidade de Outage', fontsize=11.5)
            ax.tick_params(axis='both', which='major', labelsize=10.5)
            ax.set_xlim([-10, 20])
            ax.set_ylim([1e-5, 1.0])
            ax.grid(True, which='both', linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', fontsize=9.5)

        fig.suptitle(f'Item 1 - Probabilidade de Outage sob Rayleigh ($\\bar{{\\gamma}}_s = {int(gamma_s_op_dB)}$ dB)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        caminho_fig1 = os.path.join(self.pasta_resultados, "Item1_Probabilidade_Outage_Rayleigh.png")
        fig.savefig(caminho_fig1, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvo: {caminho_fig1}")

        # 1.2 SEP 4-ASK
        snr_sep_dB = np.linspace(0, 25, 200)
        snr_sep_lin = 10.0**(snr_sep_dB / 10.0)
        snr_sep_sim_dB = np.arange(0, 25.1, 2.5)
        snr_sep_sim_lin = 10.0**(snr_sep_sim_dB / 10.0)

        fig, axs = plt.subplots(1, 3, figsize=(15, 6.8))
        for L in L_lista:
            # SC
            axs[0].semilogy(snr_sep_dB, self.sc.calcular_sep_teorico(snr_sep_lin, L, M=4),
                            color=cores_L[L], linestyle=':', linewidth=2.0, label=f'L={L} (Teo)')
            sep_sc_sim = self.sc.simular_sep(snr_sep_sim_lin, L, N_amostras=N_amostras, M=4)
            mask_sc = sep_sc_sim >= 1e-5
            axs[0].semilogy(snr_sep_sim_dB[mask_sc], sep_sc_sim[mask_sc], '+', color=cores_L[L],
                            markersize=8, markeredgewidth=1.8)

            # MRC
            axs[1].semilogy(snr_sep_dB, self.mrc.calcular_sep_teorico(snr_sep_lin, L, M=4),
                            color=cores_L[L], linewidth=2.0, label=f'L={L} (Teo)')
            sep_mrc_sim = self.mrc.simular_sep(snr_sep_sim_lin, L, N_amostras=N_amostras, M=4)
            mask_mrc = sep_mrc_sim >= 1e-5
            axs[1].semilogy(snr_sep_sim_dB[mask_mrc], sep_mrc_sim[mask_mrc], '+', color=cores_L[L],
                            markersize=8, markeredgewidth=1.8)

            # EGC (Puramente Simulado)
            sep_egc_sim = self.egc.simular_sep(snr_sep_sim_lin, L, N_amostras=N_amostras, M=4)
            mask_egc = sep_egc_sim >= 1e-5
            axs[2].semilogy(snr_sep_sim_dB[mask_egc], sep_egc_sim[mask_egc], 's--', color=cores_L[L],
                            linewidth=1.8, markersize=6, markeredgewidth=1.4, label=f'L={L} (Sim)')

        for ax, tit in zip(axs, ['SC (Eq. 10)', 'MRC (Eq. 20)', 'EGC (Simulação)']):
            ax.set_title(f'SEP (4-ASK) - {tit}', fontsize=12.5, fontweight='bold', pad=8)
            ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
            ax.set_ylabel('Probabilidade de Erro de Símbolo', fontsize=11.5)
            ax.tick_params(axis='both', which='major', labelsize=10.5)
            ax.set_xlim([0, 25])
            ax.set_ylim([1e-5, 1.0])
            ax.grid(True, which='both', linestyle=':', alpha=0.6)
            ax.legend(loc='lower left', fontsize=9.5)

        fig.suptitle('Item 1 - Taxa de Erro de Símbolo (4-ASK) sob Rayleigh', fontsize=14, fontweight='bold')
        plt.tight_layout()
        caminho_fig2 = os.path.join(self.pasta_resultados, "Item1_SEP_4ASK_Rayleigh.png")
        fig.savefig(caminho_fig2, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvo: {caminho_fig2}")

        # 1.3 Capacidade Ergódica
        snr_cap_dB = np.linspace(-5, 25, 200)
        snr_cap_lin = 10.0**(snr_cap_dB / 10.0)
        snr_cap_sim_dB = np.arange(-5, 25.1, 2.5)
        snr_cap_sim_lin = 10.0**(snr_cap_sim_dB / 10.0)

        fig, axs = plt.subplots(1, 3, figsize=(15, 6.8))
        for L in L_lista:
            # SC
            axs[0].plot(snr_cap_dB, self.sc.calcular_cerg_teorico(snr_cap_lin, L),
                        color=cores_L[L], linestyle=':', linewidth=2.0, label=f'L={L} (Teo)')
            cap_sc_sim = self.sc.simular_cerg(snr_cap_sim_lin, L, N_amostras=1000000)
            axs[0].plot(snr_cap_sim_dB, cap_sc_sim, '+', color=cores_L[L], markersize=8, markeredgewidth=1.8)

            # MRC
            axs[1].plot(snr_cap_dB, self.mrc.calcular_cerg_teorico(snr_cap_lin, L),
                        color=cores_L[L], linewidth=2.0, label=f'L={L} (Teo)')
            cap_mrc_sim = self.mrc.simular_cerg(snr_cap_sim_lin, L, N_amostras=1000000)
            axs[1].plot(snr_cap_sim_dB, cap_mrc_sim, '+', color=cores_L[L], markersize=8, markeredgewidth=1.8)

            # EGC (Puramente Simulado)
            cap_egc_sim = self.egc.simular_cerg(snr_cap_sim_lin, L, N_amostras=1000000)
            axs[2].plot(snr_cap_sim_dB, cap_egc_sim, 's--', color=cores_L[L],
                        linewidth=1.8, markersize=6, markeredgewidth=1.4, label=f'L={L} (Sim)')

        for ax, tit in zip(axs, ['SC (Eq. 11)', 'MRC (Eq. 22)', 'EGC (Simulação)']):
            ax.set_title(f'Capacidade Ergódica - {tit}', fontsize=12.5, fontweight='bold', pad=8)
            ax.set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
            ax.set_ylabel('Capacidade (bits/s/Hz)', fontsize=11.5)
            ax.tick_params(axis='both', which='major', labelsize=10.5)
            ax.set_xlim([-5, 25])
            ax.set_ylim([0, 11])
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', fontsize=9.5)

        fig.suptitle('Item 1 - Capacidade Ergódica sob Rayleigh', fontsize=14, fontweight='bold')
        plt.tight_layout()
        caminho_fig3 = os.path.join(self.pasta_resultados, "Item1_Capacidade_Ergodica_Rayleigh.png")
        fig.savefig(caminho_fig3, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvo: {caminho_fig3}")

    # =========================================================================
    # ITEM 2: Comparação de Desempenho entre Combinadores (L = 4)
    # =========================================================================
    def executar_item_2(self, L_alvo=4, N_amostras=2500000):
        """
        Gera a comparação direta dos três combinadores para uma mesma ordem L = 4.
        """
        print(f"\n--- Executando Item 2: Comparação Direta entre Combinadores (L = {L_alvo}) ---")
        cores_comb = {'MRC': '#1f77b4', 'EGC': '#2ca02c', 'SC': '#d62728'}

        fig, axs = plt.subplots(1, 3, figsize=(15, 6.8))

        # 2.1 Outage
        gamma_s_op_lin = 10.0**(10.0 / 10.0)
        gamma_th_dB = np.linspace(-10, 20, 200)
        gamma_th_lin = 10.0**(gamma_th_dB / 10.0)
        gamma_th_sim_dB = np.arange(-10, 20.1, 2.5)
        gamma_th_sim_lin = 10.0**(gamma_th_sim_dB / 10.0)

        # MRC
        axs[0].semilogy(gamma_th_dB, self.mrc.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, L_alvo),
                        color=cores_comb['MRC'], linewidth=2.0, label='MRC (Teórico)')
        op_mrc = self.mrc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L_alvo, N_amostras=N_amostras)
        mask_mrc = op_mrc >= 1e-5
        axs[0].semilogy(gamma_th_sim_dB[mask_mrc], op_mrc[mask_mrc], '+', color=cores_comb['MRC'],
                        markersize=8, markeredgewidth=1.8, label='MRC (Sim. +)')

        # EGC
        op_egc = self.egc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L_alvo, N_amostras=N_amostras)
        mask_egc = op_egc >= 1e-5
        axs[0].semilogy(gamma_th_sim_dB[mask_egc], op_egc[mask_egc], 's--', color=cores_comb['EGC'],
                        linewidth=1.8, markersize=6, markeredgewidth=1.4, label='EGC (Puramente Sim.)')

        # SC
        axs[0].semilogy(gamma_th_dB, self.sc.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, L_alvo),
                        color=cores_comb['SC'], linestyle=':', linewidth=2.0, label='SC (Teórico)')
        op_sc = self.sc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, L_alvo, N_amostras=N_amostras)
        mask_sc = op_sc >= 1e-5
        axs[0].semilogy(gamma_th_sim_dB[mask_sc], op_sc[mask_sc], '+', color=cores_comb['SC'],
                        markersize=8, markeredgewidth=1.8, label='SC (Sim. +)')

        axs[0].set_title(f'Probabilidade de Outage ($L = {L_alvo}$, $\\bar{{\\gamma}}_s = 10$ dB)', fontsize=12.5, fontweight='bold', pad=8)
        axs[0].set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)', fontsize=11.5)
        axs[0].set_ylabel('Probabilidade de Outage', fontsize=11.5)
        axs[0].tick_params(axis='both', which='major', labelsize=10.5)
        axs[0].set_xlim([-10, 20])
        axs[0].set_ylim([1e-5, 1.0])
        axs[0].grid(True, which='both', linestyle=':', alpha=0.6)
        axs[0].legend(loc='lower right', fontsize=9.5)

        # 2.2 SEP 4-ASK
        snr_sep_dB = np.linspace(0, 25, 200)
        snr_sep_lin = 10.0**(snr_sep_dB / 10.0)
        snr_sep_sim_dB = np.arange(0, 25.1, 2.5)
        snr_sep_sim_lin = 10.0**(snr_sep_sim_dB / 10.0)

        # MRC
        axs[1].semilogy(snr_sep_dB, self.mrc.calcular_sep_teorico(snr_sep_lin, L_alvo, M=4),
                        color=cores_comb['MRC'], linewidth=2.0, label='MRC (Teórico)')
        sep_mrc_sim = self.mrc.simular_sep(snr_sep_sim_lin, L_alvo, N_amostras=N_amostras, M=4)
        mask_mrc_sep = sep_mrc_sim >= 1e-5
        axs[1].semilogy(snr_sep_sim_dB[mask_mrc_sep], sep_mrc_sim[mask_mrc_sep], '+', color=cores_comb['MRC'],
                        markersize=8, markeredgewidth=1.8, label='MRC (Sim. +)')

        # EGC
        sep_egc = self.egc.simular_sep(snr_sep_sim_lin, L_alvo, N_amostras=N_amostras, M=4)
        mask_egc_sep = sep_egc >= 1e-5
        axs[1].semilogy(snr_sep_sim_dB[mask_egc_sep], sep_egc[mask_egc_sep], 's--', color=cores_comb['EGC'],
                        linewidth=1.8, markersize=6, markeredgewidth=1.4, label='EGC (Puramente Sim.)')

        # SC
        axs[1].semilogy(snr_sep_dB, self.sc.calcular_sep_teorico(snr_sep_lin, L_alvo, M=4),
                        color=cores_comb['SC'], linestyle=':', linewidth=2.0, label='SC (Teórico)')
        sep_sc_sim = self.sc.simular_sep(snr_sep_sim_lin, L_alvo, N_amostras=N_amostras, M=4)
        mask_sc_sep = sep_sc_sim >= 1e-5
        axs[1].semilogy(snr_sep_sim_dB[mask_sc_sep], sep_sc_sim[mask_sc_sep], '+', color=cores_comb['SC'],
                        markersize=8, markeredgewidth=1.8, label='SC (Sim. +)')

        axs[1].set_title(f'Probabilidade de Erro de Símbolo ($L = {L_alvo}$)', fontsize=12.5, fontweight='bold', pad=8)
        axs[1].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs[1].set_ylabel(r'Probabilidade de Erro de Símbolo ($P_s$)', fontsize=11.5)
        axs[1].tick_params(axis='both', which='major', labelsize=10.5)
        axs[1].set_xlim([0, 25])
        axs[1].set_ylim([1e-5, 1.0])
        axs[1].grid(True, which='both', linestyle=':', alpha=0.6)
        axs[1].legend(loc='lower left', fontsize=9.5)

        # 2.3 Capacidade Ergódica
        snr_cap_dB = np.linspace(-5, 25, 200)
        snr_cap_lin = 10.0**(snr_cap_dB / 10.0)
        snr_cap_sim_dB = np.arange(-5, 25.1, 2.5)
        snr_cap_sim_lin = 10.0**(snr_cap_sim_dB / 10.0)

        # MRC
        axs[2].plot(snr_cap_dB, self.mrc.calcular_cerg_teorico(snr_cap_lin, L_alvo),
                    color=cores_comb['MRC'], linewidth=2.0, label='MRC (Teórico)')
        cap_mrc_sim = self.mrc.simular_cerg(snr_cap_sim_lin, L_alvo, N_amostras=1000000)
        axs[2].plot(snr_cap_sim_dB, cap_mrc_sim, '+', color=cores_comb['MRC'],
                    markersize=8, markeredgewidth=1.8, label='MRC (Sim. +)')

        # EGC
        cap_egc = self.egc.simular_cerg(snr_cap_sim_lin, L_alvo, N_amostras=1000000)
        axs[2].plot(snr_cap_sim_dB, cap_egc, 's--', color=cores_comb['EGC'],
                    linewidth=1.8, markersize=6, markeredgewidth=1.4, label='EGC (Puramente Sim.)')

        # SC
        axs[2].plot(snr_cap_dB, self.sc.calcular_cerg_teorico(snr_cap_lin, L_alvo),
                    color=cores_comb['SC'], linestyle=':', linewidth=2.0, label='SC (Teórico)')
        cap_sc_sim = self.sc.simular_cerg(snr_cap_sim_lin, L_alvo, N_amostras=1000000)
        axs[2].plot(snr_cap_sim_dB, cap_sc_sim, '+', color=cores_comb['SC'],
                    markersize=8, markeredgewidth=1.8, label='SC (Sim. +)')

        axs[2].set_title(f'Capacidade Ergódica ($L = {L_alvo}$)', fontsize=12.5, fontweight='bold', pad=8)
        axs[2].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs[2].set_ylabel('Capacidade (bits/s/Hz)', fontsize=11.5)
        axs[2].tick_params(axis='both', which='major', labelsize=10.5)
        axs[2].set_xlim([-5, 25])
        axs[2].set_ylim([0, 11])
        axs[2].grid(True, linestyle=':', alpha=0.6)
        axs[2].legend(loc='lower right', fontsize=9.5)

        fig.suptitle(f'Item 2 - Comparação de Desempenho entre Combinadores ($L = {L_alvo}$)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        caminho_fig = os.path.join(self.pasta_resultados, f"Item2_Comparacao_Combinadores_L{L_alvo}.png")
        fig.savefig(caminho_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvo: {caminho_fig}")

    # =========================================================================
    # ITEM 3: Ganho de Diversidade e Análise Rayleigh vs. Riceano (Slide 11)
    # =========================================================================
    def executar_item_3(self, N_amostras=3000000):
        """
        Calcula a razão de ganho com diversidade / sem diversidade (L = 4 vs. L = 1)
        para as três métricas sob Rayleigh (K_R = 0) e Riceano (K_R in {2, 6, 10}).
        """
        print("\n--- Executando Item 3: Ganho de Diversidade e Análise Rayleigh vs Rice (Slide 11) ---")

        np.random.seed(42)
        L = 4
        snr_dB = np.arange(0, 25.1, 2.5)
        snr_lin = 10.0**(snr_dB / 10.0)
        th_lin = 10.0**(5.0 / 10.0)  # Limiar fixo de 5 dB

        # -------------------------------------------------------------
        # Curvas Teóricas para Rayleigh (L=4 vs SISO L=1)
        # -------------------------------------------------------------
        snr_cont_dB = np.linspace(0, 25, 200)
        snr_cont_lin = 10.0**(snr_cont_dB / 10.0)

        # Capacidade: C4 / C1
        c1_ray_teo = self.mrc.calcular_cerg_teorico(snr_cont_lin, 1)
        c4_ray_teo = self.mrc.calcular_cerg_teorico(snr_cont_lin, 4)
        c_gain_ray_teo = 10.0 * np.log10(c4_ray_teo / c1_ray_teo)

        # SEP 4-ASK: P4 / P1
        sep1_ray_teo = self.mrc.calcular_sep_teorico(snr_cont_lin, 1, M=4)
        sep4_ray_teo = self.mrc.calcular_sep_teorico(snr_cont_lin, 4, M=4)
        sep_gain_ray_teo = 10.0 * np.log10(sep4_ray_teo / sep1_ray_teo)

        # Outage: Pout,4 / Pout,1
        def op_rayleigh_exact(th, g_vec, n_ramos):
            res = []
            for g in g_vec:
                x = th / g
                poly = np.sum([x**k / sp.factorial(k) for k in range(n_ramos)])
                res.append(1.0 - np.exp(-x) * poly)
            return np.array(res)

        op1_ray_teo = op_rayleigh_exact(th_lin, snr_cont_lin, 1)
        op4_ray_teo = op_rayleigh_exact(th_lin, snr_cont_lin, 4)
        op_gain_ray_teo = 10.0 * np.log10(op4_ray_teo / op1_ray_teo)

        # -------------------------------------------------------------
        # Simulação Monte Carlo Estocástica Pura (Kr in {0, 2, 6, 10})
        # -------------------------------------------------------------
        kr_valores = [0, 2, 6, 10]
        sim_data = {}
        for kr in kr_valores:
            print(f"-> Simulando canal Monte Carlo para Kr = {kr}...")
            s = np.sqrt(kr / (kr + 1.0))
            sigma = np.sqrt(1.0 / (2.0 * (kr + 1.0)))

            # Canal SISO (L=1) e quad-ramo (L=4)
            h1 = (s + sigma * np.random.randn(1, N_amostras)) + 1j * (sigma * np.random.randn(1, N_amostras))
            norm_sq_1 = np.abs(h1[0])**2

            h4 = (s + sigma * np.random.randn(L, N_amostras)) + 1j * (sigma * np.random.randn(L, N_amostras))
            norm_sq_4 = np.sum(np.abs(h4)**2, axis=0)

            c_g = []
            sep_g, snr_sep = [], []
            op_g, snr_op = [], []

            for g_db, g_s in zip(snr_dB, snr_lin):
                # Capacidade: C4 / C1
                c1 = np.mean(np.log2(1.0 + norm_sq_1 * g_s))
                c4 = np.mean(np.log2(1.0 + norm_sq_4 * g_s))
                c_g.append(10.0 * np.log10(c4 / c1))

                # SEP 4-ASK: Ps,4 / Ps,1 (b_M/2 = 0.4/2 = 0.2)
                pe1 = np.mean(0.75 * sp.erfc(np.sqrt(0.2 * norm_sq_1 * g_s)))
                pe4 = np.mean(0.75 * sp.erfc(np.sqrt(0.2 * norm_sq_4 * g_s)))
                if pe4 >= 1e-8:
                    sep_g.append(10.0 * np.log10(pe4 / pe1))
                    snr_sep.append(g_db)

                # Outage: Pout,4 / Pout,1
                n1 = np.sum(norm_sq_1 * g_s < th_lin)
                n4 = np.sum(norm_sq_4 * g_s < th_lin)
                if n4 >= 10:  # Piso prático: pelo menos 10 eventos na amostra (N=3M)
                    op_g.append(10.0 * np.log10((n4 / N_amostras) / (n1 / N_amostras)))
                    snr_op.append(g_db)

            sim_data[kr] = {
                'c_g': c_g,
                'sep_g': sep_g, 'snr_sep': snr_sep,
                'op_g': op_g, 'snr_op': snr_op
            }

        # -------------------------------------------------------------
        # Construção da Figura Panorâmica
        # -------------------------------------------------------------
        fig, axs = plt.subplots(1, 3, figsize=(15, 6.8), dpi=300)

        estilos = {
            0: {'label': 'Rayleigh (Sim. +)', 'color': '#1f77b4', 'fmt': '+', 'ms': 8, 'mew': 1.8},
            2: {'label': r'Rice ($K_R=2$, Sim.)', 'color': '#ff7f0e', 'fmt': 's--', 'ms': 6, 'mew': 1.3},
            6: {'label': r'Rice ($K_R=6$, Sim.)', 'color': '#2ca02c', 'fmt': '^-.', 'ms': 6, 'mew': 1.3},
            10: {'label': r'Rice ($K_R=10$, Sim.)', 'color': '#9467bd', 'fmt': 'd:', 'ms': 6, 'mew': 1.3},
        }

        # 1. Capacidade Ergódica (C4 / C1)
        axs[0].plot(snr_cont_dB, c_gain_ray_teo, 'b-', linewidth=2.0, label='Rayleigh (Teórico)')
        axs[0].plot(snr_dB, sim_data[0]['c_g'], '+', color=estilos[0]['color'],
                    markersize=estilos[0]['ms'], markeredgewidth=estilos[0]['mew'], label=estilos[0]['label'])
        for kr in [2, 6, 10]:
            e = estilos[kr]
            axs[0].plot(snr_dB, sim_data[kr]['c_g'], e['fmt'], color=e['color'],
                        markersize=e['ms'], linewidth=1.8, label=e['label'])
        axs[0].set_title(r'Razão de Capacidade: $10\log_{10}(C_4 / C_1)$', fontsize=12.5, fontweight='bold', pad=10)
        axs[0].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs[0].set_ylabel('Razão de Capacidade (dB)', fontsize=11.5)
        axs[0].tick_params(axis='both', which='major', labelsize=10.5)
        axs[0].set_xlim([0, 25])
        axs[0].set_ylim([0, 7.5])
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].legend(loc='upper right', fontsize=9.5)

        # 2. Taxa de Erro de Símbolo (Ps,4 / Ps,1)
        axs[1].plot(snr_cont_dB, sep_gain_ray_teo, 'b-', linewidth=2.0, label='Rayleigh (Teórico)')
        axs[1].plot(sim_data[0]['snr_sep'], sim_data[0]['sep_g'], '+', color=estilos[0]['color'],
                    markersize=estilos[0]['ms'], markeredgewidth=estilos[0]['mew'], label=estilos[0]['label'])
        for kr in [2, 6, 10]:
            e = estilos[kr]
            axs[1].plot(sim_data[kr]['snr_sep'], sim_data[kr]['sep_g'], e['fmt'], color=e['color'],
                        markersize=e['ms'], linewidth=1.8, label=e['label'])
        axs[1].set_title(r'Razão de Erro: $10\log_{10}(P_{s,4} / P_{s,1})$', fontsize=12.5, fontweight='bold', pad=10)
        axs[1].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs[1].set_ylabel('Razão Relativa de Erro (dB)', fontsize=11.5)
        axs[1].tick_params(axis='both', which='major', labelsize=10.5)
        axs[1].set_xlim([0, 25])
        axs[1].set_ylim([-65, 0])
        axs[1].grid(True, linestyle=':', alpha=0.6)
        axs[1].legend(loc='lower left', fontsize=9.5)

        # 3. Probabilidade de Outage (Pout,4 / Pout,1)
        axs[2].plot(snr_cont_dB, op_gain_ray_teo, 'b-', linewidth=2.0, label='Rayleigh (Teórico)')
        axs[2].plot(sim_data[0]['snr_op'], sim_data[0]['op_g'], '+', color=estilos[0]['color'],
                    markersize=estilos[0]['ms'], markeredgewidth=estilos[0]['mew'], label=estilos[0]['label'])
        for kr in [2, 6, 10]:
            e = estilos[kr]
            axs[2].plot(sim_data[kr]['snr_op'], sim_data[kr]['op_g'], e['fmt'], color=e['color'],
                        markersize=e['ms'], linewidth=1.8, label=e['label'])
        axs[2].set_title(r'Razão de Outage: $10\log_{10}(P_{\mathrm{out},4} / P_{\mathrm{out},1})$ ($\gamma_{\mathrm{th}} = 5$ dB)', fontsize=12.5, fontweight='bold', pad=10)
        axs[2].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs[2].set_ylabel('Razão Relativa de Outage (dB)', fontsize=11.5)
        axs[2].tick_params(axis='both', which='major', labelsize=10.5)
        axs[2].set_xlim([0, 25])
        axs[2].set_ylim([-75, 0])
        axs[2].grid(True, linestyle=':', alpha=0.6)
        axs[2].legend(loc='lower left', fontsize=9.5)

        fig.suptitle(r'Item 3 - Razão com Diversidade / sem Diversidade ($L = 4$ vs. SISO $L = 1$): Rayleigh vs. Riceano ($K_R \in \{2, 6, 10\}$)',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.84, bottom=0.12, wspace=0.24)
        caminho_fig = os.path.join(self.pasta_resultados, "Item3_Ganho_Diversidade_Rayleigh_vs_Rice.png")
        fig.savefig(caminho_fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvo: {caminho_fig}")

        # -------------------------------------------------------------
        # Comparativo Global Rayleigh vs Rice (K_R = 4, L = 4)
        # -------------------------------------------------------------
        print("-> Gerando gráfico comparativo Rayleigh vs Rice para os 3 combinadores...")
        fig_rr, axs_rr = plt.subplots(1, 3, figsize=(15, 6.8))

        # Outage
        gamma_s_op_lin = 10.0**(10.0 / 10.0)
        gamma_th_dB = np.linspace(-10, 20, 200)
        gamma_th_lin = 10.0**(gamma_th_dB / 10.0)
        gamma_th_sim_dB = np.arange(-10, 20.1, 2.5)
        gamma_th_sim_lin = 10.0**(gamma_th_sim_dB / 10.0)

        # MRC
        axs_rr[0].semilogy(gamma_th_dB, self.mrc.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, 4),
                           'b-', linewidth=1.8, label='MRC (Rayleigh Teo.)')
        op_mrc_ray_sim = self.mrc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, 4, N_amostras=N_amostras, Kr=0)
        mask_mrc_ray = op_mrc_ray_sim >= 1e-5
        axs_rr[0].semilogy(gamma_th_sim_dB[mask_mrc_ray], op_mrc_ray_sim[mask_mrc_ray],
                           '+', color='b', markersize=7, markeredgewidth=1.6, label='MRC (Rayleigh Sim. +)')
        op_mrc_rice = self.mrc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, 4, N_amostras=N_amostras, Kr=4)
        mask_mrc_rice = op_mrc_rice >= 1e-5
        axs_rr[0].semilogy(gamma_th_sim_dB[mask_mrc_rice], op_mrc_rice[mask_mrc_rice],
                           'b^--', linewidth=1.5, markersize=5, label='MRC (Rice $K_R=4$ Sim.)')

        # EGC
        op_egc_ray = self.egc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, 4, N_amostras=N_amostras, Kr=0)
        mask_egc_ray = op_egc_ray >= 1e-5
        axs_rr[0].semilogy(gamma_th_sim_dB[mask_egc_ray], op_egc_ray[mask_egc_ray],
                           'gs-', linewidth=1.5, markersize=5, label='EGC (Rayleigh Sim.)')
        op_egc_rice = self.egc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, 4, N_amostras=N_amostras, Kr=4)
        mask_egc_rice = op_egc_rice >= 1e-5
        axs_rr[0].semilogy(gamma_th_sim_dB[mask_egc_rice], op_egc_rice[mask_egc_rice],
                           'gs--', linewidth=1.5, markersize=5, label='EGC (Rice $K_R=4$ Sim.)')

        # SC
        axs_rr[0].semilogy(gamma_th_dB, self.sc.calcular_op_teorico(gamma_th_lin, gamma_s_op_lin, 4),
                           'r-', linewidth=1.8, label='SC (Rayleigh Teo.)')
        op_sc_ray_sim = self.sc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, 4, N_amostras=N_amostras, Kr=0)
        mask_sc_ray = op_sc_ray_sim >= 1e-5
        axs_rr[0].semilogy(gamma_th_sim_dB[mask_sc_ray], op_sc_ray_sim[mask_sc_ray],
                           '+', color='r', markersize=7, markeredgewidth=1.6, label='SC (Rayleigh Sim. +)')
        op_sc_rice = self.sc.simular_op(gamma_th_sim_lin, gamma_s_op_lin, 4, N_amostras=N_amostras, Kr=4)
        mask_sc_rice = op_sc_rice >= 1e-5
        axs_rr[0].semilogy(gamma_th_sim_dB[mask_sc_rice], op_sc_rice[mask_sc_rice],
                           'r^--', linewidth=1.5, markersize=5, label='SC (Rice $K_R=4$ Sim.)')

        axs_rr[0].set_title('Probabilidade de Outage ($L = 4$)', fontsize=12.5, fontweight='bold', pad=8)
        axs_rr[0].set_xlabel(r'Limiar de SNR $\gamma_{\mathrm{th}}$ (dB)', fontsize=11.5)
        axs_rr[0].set_ylabel('Probabilidade de Outage', fontsize=11.5)
        axs_rr[0].tick_params(axis='both', which='major', labelsize=10.5)
        axs_rr[0].set_xlim([-10, 20])
        axs_rr[0].set_ylim([1e-5, 1.0])
        axs_rr[0].grid(True, which='both', linestyle=':', alpha=0.6)
        axs_rr[0].legend(loc='lower right', fontsize=9.0)

        # SEP 4-ASK
        snr_sep_dB = np.linspace(0, 25, 200)
        snr_sep_lin = 10.0**(snr_sep_dB / 10.0)
        snr_sep_sim_dB = np.arange(0, 25.1, 2.5)
        snr_sep_sim_lin = 10.0**(snr_sep_sim_dB / 10.0)

        # MRC
        axs_rr[1].semilogy(snr_sep_dB, self.mrc.calcular_sep_teorico(snr_sep_lin, 4, M=4),
                           'b-', linewidth=1.8, label='MRC (Rayleigh Teo.)')
        sep_mrc_ray_sim = self.mrc.simular_sep(snr_sep_sim_lin, 4, N_amostras=N_amostras, Kr=0, M=4)
        mask_mrc_sep = sep_mrc_ray_sim >= 1e-5
        axs_rr[1].semilogy(snr_sep_sim_dB[mask_mrc_sep], sep_mrc_ray_sim[mask_mrc_sep],
                           '+', color='b', markersize=7, markeredgewidth=1.6, label='MRC (Rayleigh Sim. +)')
        sep_mrc_rice = self.mrc.simular_sep(snr_sep_sim_lin, 4, N_amostras=N_amostras, Kr=4, M=4)
        mask_mrc_rice_sep = sep_mrc_rice >= 1e-5
        axs_rr[1].semilogy(snr_sep_sim_dB[mask_mrc_rice_sep], sep_mrc_rice[mask_mrc_rice_sep],
                           'b^--', linewidth=1.5, markersize=5, label='MRC (Rice $K_R=4$ Sim.)')

        # EGC
        sep_egc_ray = self.egc.simular_sep(snr_sep_sim_lin, 4, N_amostras=N_amostras, Kr=0, M=4)
        mask_egc_sep = sep_egc_ray >= 1e-5
        axs_rr[1].semilogy(snr_sep_sim_dB[mask_egc_sep], sep_egc_ray[mask_egc_sep],
                           'gs-', linewidth=1.5, markersize=5, label='EGC (Rayleigh Sim.)')
        sep_egc_rice = self.egc.simular_sep(snr_sep_sim_lin, 4, N_amostras=N_amostras, Kr=4, M=4)
        mask_egc_rice_sep = sep_egc_rice >= 1e-5
        axs_rr[1].semilogy(snr_sep_sim_dB[mask_egc_rice_sep], sep_egc_rice[mask_egc_rice_sep],
                           'gs--', linewidth=1.5, markersize=5, label='EGC (Rice $K_R=4$ Sim.)')

        # SC
        axs_rr[1].semilogy(snr_sep_dB, self.sc.calcular_sep_teorico(snr_sep_lin, 4, M=4),
                           'r-', linewidth=1.8, label='SC (Rayleigh Teo.)')
        sep_sc_ray_sim = self.sc.simular_sep(snr_sep_sim_lin, 4, N_amostras=N_amostras, Kr=0, M=4)
        mask_sc_sep = sep_sc_ray_sim >= 1e-5
        axs_rr[1].semilogy(snr_sep_sim_dB[mask_sc_sep], sep_sc_ray_sim[mask_sc_sep],
                           '+', color='r', markersize=7, markeredgewidth=1.6, label='SC (Rayleigh Sim. +)')
        sep_sc_rice = self.sc.simular_sep(snr_sep_sim_lin, 4, N_amostras=N_amostras, Kr=4, M=4)
        mask_sc_rice_sep = sep_sc_rice >= 1e-5
        axs_rr[1].semilogy(snr_sep_sim_dB[mask_sc_rice_sep], sep_sc_rice[mask_sc_rice_sep],
                           'r^--', linewidth=1.5, markersize=5, label='SC (Rice $K_R=4$ Sim.)')

        axs_rr[1].set_title(r'Probabilidade de Erro de Símbolo ($L = 4$)', fontsize=12.5, fontweight='bold', pad=8)
        axs_rr[1].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs_rr[1].set_ylabel(r'Probabilidade de Erro de Símbolo ($P_s$)', fontsize=11.5)
        axs_rr[1].tick_params(axis='both', which='major', labelsize=10.5)
        axs_rr[1].set_xlim([0, 25])
        axs_rr[1].set_ylim([1e-5, 1.0])
        axs_rr[1].grid(True, which='both', linestyle=':', alpha=0.6)
        axs_rr[1].legend(loc='lower left', fontsize=9.0)

        # Capacidade
        snr_cap_dB = np.linspace(-5, 25, 200)
        snr_cap_lin = 10.0**(snr_cap_dB / 10.0)
        snr_cap_sim_dB = np.arange(-5, 25.1, 2.5)
        snr_cap_sim_lin = 10.0**(snr_cap_sim_dB / 10.0)

        # MRC
        axs_rr[2].plot(snr_cap_dB, self.mrc.calcular_cerg_teorico(snr_cap_lin, 4),
                       'b-', linewidth=1.8, label='MRC (Rayleigh Teo.)')
        cap_mrc_ray_sim = self.mrc.simular_cerg(snr_cap_sim_lin, 4, N_amostras=1000000, Kr=0)
        axs_rr[2].plot(snr_cap_sim_dB, cap_mrc_ray_sim, '+', color='b',
                       markersize=7, markeredgewidth=1.6, label='MRC (Rayleigh Sim. +)')
        cap_mrc_rice = self.mrc.simular_cerg(snr_cap_sim_lin, 4, N_amostras=1000000, Kr=4)
        axs_rr[2].plot(snr_cap_sim_dB, cap_mrc_rice, 'b^--', linewidth=1.5, markersize=5,
                       label='MRC (Rice $K_R=4$ Sim.)')

        # EGC
        cap_egc_ray = self.egc.simular_cerg(snr_cap_sim_lin, 4, N_amostras=1000000, Kr=0)
        axs_rr[2].plot(snr_cap_sim_dB, cap_egc_ray, 'gs-', linewidth=1.5, markersize=5,
                       label='EGC (Rayleigh Sim.)')
        cap_egc_rice = self.egc.simular_cerg(snr_cap_sim_lin, 4, N_amostras=1000000, Kr=4)
        axs_rr[2].plot(snr_cap_sim_dB, cap_egc_rice, 'gs--', linewidth=1.5, markersize=5,
                       label='EGC (Rice $K_R=4$ Sim.)')

        # SC
        axs_rr[2].plot(snr_cap_dB, self.sc.calcular_cerg_teorico(snr_cap_lin, 4),
                       'r-', linewidth=1.8, label='SC (Rayleigh Teo.)')
        cap_sc_ray_sim = self.sc.simular_cerg(snr_cap_sim_lin, 4, N_amostras=1000000, Kr=0)
        axs_rr[2].plot(snr_cap_sim_dB, cap_sc_ray_sim, '+', color='r',
                       markersize=7, markeredgewidth=1.6, label='SC (Rayleigh Sim. +)')
        cap_sc_rice = self.sc.simular_cerg(snr_cap_sim_lin, 4, N_amostras=1000000, Kr=4)
        axs_rr[2].plot(snr_cap_sim_dB, cap_sc_rice, 'r^--', linewidth=1.5, markersize=5,
                       label='SC (Rice $K_R=4$ Sim.)')

        axs_rr[2].set_title('Capacidade Ergódica ($L = 4$)', fontsize=12.5, fontweight='bold', pad=8)
        axs_rr[2].set_xlabel(r'SNR média por ramo $\bar{\gamma}_s$ (dB)', fontsize=11.5)
        axs_rr[2].set_ylabel('Capacidade (bits/s/Hz)', fontsize=11.5)
        axs_rr[2].tick_params(axis='both', which='major', labelsize=10.5)
        axs_rr[2].set_xlim([-5, 25])
        axs_rr[2].set_ylim([0, 11])
        axs_rr[2].grid(True, linestyle=':', alpha=0.6)
        axs_rr[2].legend(loc='lower right', fontsize=8.5)

        fig_rr.suptitle('Comparação Global dos Três Combinadores: Rayleigh vs. Rice ($K_R = 4$, $L = 4$)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        caminho_fig_rr = os.path.join(self.pasta_resultados, "Item3_Comparacao_Global_Rayleigh_vs_Rice_L4.png")
        fig_rr.savefig(caminho_fig_rr, dpi=300, bbox_inches='tight')
        fig_rr.savefig(caminho_fig_rr, dpi=300)
        plt.close(fig_rr)
        print(f"Salvo: {caminho_fig_rr}")

    def executar_todos_itens(self):
        """
        Executa todos os experimentos dos itens 1, 2 e 3 e imprime as tabelas comparativas.
        """
        self.imprimir_tabelas_comparativas()
        self.executar_item_1()
        self.executar_item_2()
        self.executar_item_3()
        print(f"\nTodos os experimentos foram concluídos com sucesso!")
        print(f"Resultados disponíveis em: {os.path.abspath(self.pasta_resultados)}")


if __name__ == "__main__":
    analise = AnaliseDiversidade()
    if len(sys.argv) > 1 and "--tabelas" in sys.argv:
        analise.imprimir_tabelas_comparativas()
    else:
        analise.executar_todos_itens()
