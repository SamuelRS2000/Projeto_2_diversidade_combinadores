# Projeto 2: Técnicas de Diversidade no Receptor (SC, MRC, EGC)

**Disciplina:** PPGEE3622 - Comunicações Móveis (2026.2)  
**Instituição:** Universidade de Brasília (UnB) – FT / ENE  
---

## 📌 Descrição do Projeto

Implementação computacional orientada a objetos (POO) em Python para modelagem, simulação de Monte Carlo e avaliação analítica das técnicas clássicas de combinação por diversidade espacial em recepção (SIMO):
* **SC (*Selection Combining*)**: Seleção instantânea do melhor ramo receptor.
* **MRC (*Maximal Ratio Combining*)**: Co-faseamento e ponderação ótima de Cauchy-Schwarz.
* **EGC (*Equal Gain Combining*)**: Co-faseamento com ganhos unitários idênticos.

Avaliam-se as métricas de **Probabilidade de Outage ($P_{\text{out}}$)**, **Taxa de Erro de Símbolo (SEP 4-ASK)** e **Capacidade Ergódica ($C_{\text{erg}}$)** sob canais **Rayleigh** ($L \in \{1, 2, 4\}$) e **Riceanos** ($K_R \in \{2, 6, 10\}$).

---

## 📂 Arquivos do Repositório

* `Relatorio_Projeto2_Samuel.pdf`: Relatório técnico completo (13 páginas) com formulações matemáticas, deduções assintóticas e análise dos resultados.
* `Apresentacao_Diversidade_Projeto2_Samuel.pdf`: Slides para a apresentação do projeto.
* `sc_combiner.py`: Módulo da classe `SCCombiner` (estatística de ordem, FDA/FDP e simulação).
* `mrc_combiner.py`: Módulo da classe `MRCCombiner` (distribuição de Erlang e forma fechada).
* `egc_combiner.py`: Módulo da classe `EGCCombiner` (soma de envoltórias e simulação Monte Carlo pura).
* `comparacao_combinadores.py`: Script integrador que executa os experimentos dos Itens 1, 2 e 3 e gera os gráficos e tabelas.

---

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install numpy scipy matplotlib
```

### Execução dos Códigos

1. **Exibir as tabelas de ganho de diversidade e métricas no terminal:**
```bash
python comparacao_combinadores.py --tabelas
```

2. **Executar todas as simulações de Monte Carlo e gerar as curvas:**
```bash
python comparacao_combinadores.py
```
