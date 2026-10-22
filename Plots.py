import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle



from pymoo.config import Config
Config.warnings['not_compiled'] = False
from simulation import Simulation
from main import modelos_bateria, Ns_b, Cap_uc, Ns_uc, T_xuc, SoC_uc_ref, BH, Taxa, step_pth, volume_maximo, horizonte_analise_meses
from main import data, sheet, diretorio_figuras, arquivo
from main import Resultados, MyProblem

#Dimensões das figuras em centímetros
fig_width_cm = 24/2.4
fig_height_cm = 18/2.4

# Resultados da otimização

with open('resultados.pkl', 'rb') as archive:
    resultados = pickle.load(archive)

dados_resultados = resultados.retorna_resultados()
X = dados_resultados["X"]
F = dados_resultados["F"]
G = dados_resultados["G"]
saude_bat_Ot = dados_resultados["saude_bat"]
melhor_vpl_Ot = dados_resultados["melhor_vpl"]
xl_Ot = dados_resultados["xl"]
xu_Ot = dados_resultados["xu"]
saude_uc_Ot = dados_resultados["saude_uc"]
troca_bat_Ot = dados_resultados["troca_bat"]
melhor_fluxo_caixa_Ot = dados_resultados["melhor_fluxo_caixa"]
print(f"X: {X}")
print(f"F: {F}")
print(f"G: {G}")

print("\n=== VERIFICAÇÃO (PLOTS) ===")
print("Shape de X:", X.shape if X is not None else None)
print("Soma de F:", np.sum(F) if F is not None else None)
print("Soma do Fluxo de Caixa:", np.sum(melhor_fluxo_caixa_Ot))
print("Primeiros 3 itens de saude_bat:", saude_bat_Ot[:3])
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Visualização do espaço de design -------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(fig_width_cm, fig_height_cm))
ax = plt.axes(projection='3d')
X_int = np.round(X).astype(int)
try:
    ax.scatter(X_int[:, 0], X_int[:, 1], X_int[:, 2], s=30, facecolors='r', edgecolors='r', zorder=3)
except Exception as e:
    ax.scatter(X_int[0], X_int[1], X_int[2], s=30, facecolors='r', edgecolors='r', zorder=3)
    print("Solução unica!")
ax.set_xlabel('Número de Baterias em Paralelo')
ax.set_ylabel('Número de Supercapacitores em Paralelo')
ax.set_zlabel("Potência de Limiar")
ax.set_title("Espaço de Design")
ax.grid(zorder=1)
ax.set_xticks(np.arange(1, 11, 1))
ax.set_yticks(np.arange(1, 11, 1))
ax.set_xlim(xl_Ot[0]-1, xu_Ot[0]+1)
ax.set_ylim(xl_Ot[1]-1, xu_Ot[1]+1)
ax.set_zlim(xl_Ot[2]-100, xu_Ot[2]+100)
plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_01_design_space.pdf", bbox_inches='tight')
plt.show(block=False)


# Imprimir resultados
print("\nResultados da Otimização:")
print("------------------------")
for i in range(min(10, len(X))):  # Mostrar as 10 melhores soluções
    print(f"\nSolução {i+1}:")
    try:
        print(f"Número de modulos de baterias: {int(round(X[i,0]))}")
        print(f"Número de baterias em paralelo: {int(round(X[i,1]))}")
        print(f"Número de modulos de supercapacitores: {int(round(X[i,2]))}")
        print(f"Número de supercapacitores em paralelo: {int(round(X[i,3]))}")
        print(f"Valor limiar de potência: {step_pth * int(round(X[i,4]))}")
        print(f"Modelo de bateria selecionado: {modelos_bateria[int(round(X[i,5]))]['nome']}")
        print(f'Volume ocupado: {(G[i,0] + volume_maximo):,.2f}')        
        print(f"VPL (Valor Presente Líquido): R$ {(-F[i,0]):,.2f}")
        with open(diretorio_figuras + "/" + f"{arquivo.split(".")[0]}_outputValues.txt", "w") as text_file:
            text_file.write(f"Número de modulos de baterias: {int(round(X[i,0]))}\n")
            text_file.write(f"Número de baterias em paralelo: {int(round(X[i,1]))}\n")
            text_file.write(f"Número de modulos de supercapacitores: {int(round(X[i,2]))}\n")
            text_file.write(f"Número de supercapacitores em paralelo: {int(round(X[i,3]))}\n")
            text_file.write(f"Valor limiar de potência: {step_pth * int(round(X[i,4]))} kW\n")
            text_file.write(f"Modelo de bateria selecionado: {modelos_bateria[int(round(X[i,5]))]['nome']}\n")
            text_file.write(f"Volume Ocupado: {(G[i,0] + volume_maximo):,.2f} L \n")            
            text_file.write(f"VPL (Valor Presente Líquido): R$ {(-F[i,0]):,.2f}\n")
    except:
        print(f"Número de modulos de baterias: {int(round(X[0]))}")
        print(f"Número de baterias em paralelo: {int(round(X[1]))}")
        print(f"Número de modulos de supercapacitores: {int(round(X[2]))}")
        print(f"Número de supercapacitores em paralelo: {int(round(X[3]))}")
        print(f"Valor limiar de potência: {step_pth * int(round(X[4]))}")
        print(f"Modelo de bateria selecionado: {modelos_bateria[int(round(X[5]))]['nome']}")
        print(f'Volume ocupado: {(G[0] + volume_maximo):,.2f}')        
        print(f"VPL (Valor Presente Líquido): R$ {(-F[0]):,.2f}")
        with open(diretorio_figuras + "/" + f"{arquivo.split(".")[0]}_outputValues.txt", "w") as text_file:
            text_file.write(f"Número de modulos de baterias: {int(round(X[0]))}\n")
            text_file.write(f"Número de baterias em paralelo: {int(round(X[1]))}\n")
            text_file.write(f"Número de modulos de supercapacitores: {int(round(X[2]))}\n")
            text_file.write(f"Número de supercapacitores em paralelo: {int(round(X[3]))}\n")
            text_file.write(f"Valor limiar de potência: {step_pth * int(round(X[4]))} kW\n")
            text_file.write(f"Modelo de bateria selecionado: {modelos_bateria[int(round(X[5]))]['nome']}\n")
            text_file.write(f"Volume máximo: {(G[0] + volume_maximo):,.2f} L\n")            
            text_file.write(f"VPL (Valor Presente Líquido): R$ {(-F[0]):,.2f}\n")
            

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Seleção da melhor solução ------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
try:
    idx_best = np.argmin(F[:, 0])  # Menor valor negativo de F => maior VPL
    best_Nm_b = int(round(X[idx_best, 0]))
    best_Np_b = int(round(X[idx_best, 1]))
    best_Nm_uc = int(round(X[idx_best, 2]))
    best_Np_uc = int(round(X[idx_best, 3]))
    best_Pth = step_pth * int(round(X[idx_best, 4]))
    best_modelo_b = int(round(X[idx_best, 5]))
except:
    idx_best = np.argmin(F[0])  # Menor valor negativo de F => maior VPL
    best_Nm_b = int(round(X[0]))
    best_Np_b = int(round(X[1]))
    best_Nm_uc = int(round(X[2]))
    best_Np_uc = int(round(X[3]))
    best_Pth = step_pth * int(round(X[4]))
    best_modelo_b = int(round(X[5]))


# Rodar a simulação para a melhor solução
sim = Simulation()
sim.configFluxUC2Bat(SoC_uc_ref, BH, Taxa)
sim.setParam_Batt(C=modelos_bateria[best_modelo_b]['capacidade_Ah'], Ns=Ns_b, Np=best_Np_b, Nm=best_Nm_b, Vnom=modelos_bateria[best_modelo_b]['tensao_nominal_V'], SoC=50, T_m=modelos_bateria[best_modelo_b]['c_rate'])
sim.setParam_UC(C=3400, Cap_uc=Cap_uc, Ns=Ns_uc, Np=best_Np_uc, Nm=best_Nm_uc, Vnom=3, SoC=50, T_m=T_xuc)
sim.simulate(data, sheet, best_Pth)

# -----------------------------------------------------------------------------------------------------------
# ---------------------- Força simulacao com determinado valor ----------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# sim = Simulation()
# sim.setParam_Batt(C=Cap_b, Ns=Ns_b, Np=5, Nm=Nm_b, Vnom=3.2, SoC=50, T_m=T_xb)
# sim.setParam_UC(C=3400, Cap_uc=Cap_uc, Ns=Ns_uc, Np=2, Nm=Nm_uc, Vnom=3, SoC=50, T_m=T_xuc)
# sim.simulate(data, sheet, 1400)
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# Carregar dados do Excel para plotar potência, corrente e tensão de entrada

input_df = pd.read_excel(data, sheet_name="Log")

time = np.arange(len(input_df))

# Potência do diesel (entrada real do sistema)
try:
    powers_diesel = input_df["fa08_m2amps"] * input_df["fa00_altoutvolts"]  # em Watts
    corrente = input_df["fa08_m2amps"]
    tensao = input_df["fa00_altoutvolts"]
except:
    powers_diesel = input_df["ai_04_m2amps"] * input_df["ai_11_altoutvolts"]  # em Watts
    corrente = input_df["ai_04_m2amps"]
    tensao = tensao = input_df["ai_11_altoutvolts"]



# Ajustar a simulação para usar a potência do diesel como entrada


# O restante do código permanece igual, usando os resultados da simulação

# # Carregar dados do Excel para plotar potência, corrente e tensão de entrada
# import pandas as pd
# input_df = pd.read_excel(data, sheet_name="Log")

# time = np.arange(len(input_df))

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Gráfico de Potência, Corrente e Tensão de Entrada -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------  

fig, axs = plt.subplots(3, 1, figsize=(fig_width_cm, fig_height_cm*1.2), sharex=True)
axs[0].set_title('Dados de Entrada do Ciclo (Arquivo Excel)')
axs[0].plot(time, tensao, linewidth = 2, label='Tensão [V]', color='tab:blue')
axs[0].set_ylabel('Tensão [V]')
axs[0].grid()
axs[0].legend(loc='upper right')

axs[1].plot(time, corrente, linewidth = 2, label='Corrente [A]', color='tab:orange')
axs[1].set_ylabel('Corrente [A]')
axs[1].grid()
axs[1].legend(loc='upper right')

axs[2].plot(time, powers_diesel/1000, linewidth = 2, label='Potência [kW]', color = "tab:green")
axs[2].set_ylabel('Potência [kW]')
axs[2].set_xlabel('Tempo [s]')
axs[2].set_xlim(0, len(input_df))
axs[2].grid()
axs[2].legend(loc="upper right")


plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_02_power_current_voltage.pdf", bbox_inches='tight')
plt.show(block=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Gráfico de Potência e Corrente usando subplots -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

sim_time = np.arange(len(sim._p_batt))
fig, axs = plt.subplots(3, 2, figsize=(fig_width_cm*1.5, fig_height_cm*1.2), sharex=True)

# Potência em kW
p_batt_kw = np.array(sim._p_batt) / 1e3
p_uc_kw = np.array(sim._p_uc) / 1e3

color_power = "tab:green" 
color_voltage = "tab:blue"
color_current = "tab:orange"
color_reject_power = "tab:red"
color_soc = "tab:purple"

# -------   bateria     -------------
axs[0, 0].set_title("Bateria")
axs[0, 0].plot(time, p_batt_kw, linewidth = 2, label='Potência Bateria', color=color_power)
axs[0, 0].legend(loc = 'upper right')
axs[0, 0].set_ylabel("Potência [kW]")
axs[0, 0].grid()

axs[1, 0].plot(time, sim._i_bat, linewidth = 2, label='Corrente Bateria', color=color_current)
axs[1, 0].legend(loc = 'upper right')
axs[1, 0].set_ylabel("Corrente [A]")
axs[1, 0].grid()

axs[2, 0].plot(time, np.array(sim._p_bat_reject), linewidth = 2, label='Potência Rejeitada Bateria [kW]', color=color_reject_power)
axs[2, 0].legend(loc = 'upper right')
axs[2, 0].set_ylabel("Potência")
axs[2, 0].set_xlabel("Tempo [s]")
axs[2, 0].grid()
axs[2, 0].set_xlim([0, sim_time[-1]])


# -------   supercapacitor     -------------
axs[0, 1].set_title("Supercapacitor")
axs[0, 1].plot(time, p_uc_kw, linewidth = 2, label='Potência Supercapacitor', color=color_power)
axs[0, 1].legend(loc = 'upper right')
axs[0, 1].grid()

axs[1, 1].plot(time, sim._i_uc, linewidth = 2, label='Corrente Supercapacitor', color=color_current)
axs[1, 1].legend(loc = 'upper right')
axs[1, 1].grid()

axs[2, 1].plot(time, np.array(sim._p_uc_reject), linewidth = 2, label='Potência Rejeitada Supercapacitor [kW]', color=color_reject_power)
axs[2, 1].legend(loc = 'upper right')
axs[2, 1].set_xlabel("Tempo [s]")
axs[2, 1].grid()
axs[2, 1].set_xlim([0, time[-1]])

# # --- BATERIA ---
# axs[0].set_ylabel('Potência Bateria [kW]', color=color_batt)
# l1 = axs[0].plot(sim_time, p_batt_kw, label='Potência Bateria [kW]', color=color_batt)
# l_bat_rej = axs[0].plot(sim_time, np.array(sim._p_bat_reject), label='Potência Rejeitada Bateria [kW]', color='tab:red', linestyle=':')
# axs[0].tick_params(axis='y', labelcolor=color_batt)

# ax2_0 = axs[0].twinx()
# l2 = ax2_0.plot(sim_time, sim._i_bat, label='Corrente Bateria [A]', color=color_batt_i)
# ax2_0.set_ylabel('Corrente Bateria [A]', color=color_batt_i)
# ax2_0.tick_params(axis='y', labelcolor=color_batt_i)

# lns0 = l1 + l_bat_rej + l2
# labs0 = [l.get_label() for l in lns0]
# axs[0].legend(lns0, labs0, loc='upper right')
# axs[0].set_title('Bateria')
# axs[0].grid()

# # --- SUPERCAPACITOR ---
# axs[1].set_ylabel('Potência Supercapacitor [kW]', color=color_uc)
# l3 = axs[1].plot(sim_time, p_uc_kw, label='Potência Supercapacitor [kW]', color=color_uc)
# l_uc_rej = axs[1].plot(sim_time, np.array(sim._p_uc_reject), label='Potência Rejeitada Supercapacitor [kW]', color='tab:purple', linestyle=':')
# axs[1].tick_params(axis='y', labelcolor=color_uc)

# ax2_1 = axs[1].twinx()
# l4 = ax2_1.plot(sim_time, sim._i_uc, label='Corrente Supercapacitor [A]', color=color_uc_i)
# ax2_1.set_ylabel('Corrente Supercapacitor [A]', color=color_uc_i)
# ax2_1.tick_params(axis='y', labelcolor=color_uc_i)

# lns1 = l3 + l_uc_rej + l4
# labs1 = [l.get_label() for l in lns1]
# axs[1].legend(lns1, labs1, loc='upper right')
# axs[1].set_title('Supercapacitor')
# axs[1].grid()

# # --- POTÊNCIA REJEITADA TOTAL ---

# axs[2].plot(sim_time, p_rej_total, label='Potência Rejeitada Total [kW]', color='tab:orange')
# axs[2].set_ylabel('Potência Rejeitada Total [kW]')
# axs[2].set_xlabel('Amostra')
# axs[2].legend(loc='upper right')
# axs[2].set_title('Potência Rejeitada Total')
# axs[2].grid()

plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_03_power_current_voltage_subplots.pdf", bbox_inches='tight')    
plt.show(block=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Gráfico SoC, Tensão e Corrente: Bateria x Supercapacitor -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

fig, axs = plt.subplots(3, 2, figsize=(fig_width_cm*1.5, fig_height_cm*1.2), sharex=True)

# Linha 1: SoC
axs[0, 0].plot(time, sim._SoC, linewidth = 2, color=color_soc, label='SoC Bateria')
axs[0, 0].set_ylabel(r'SoC [\%]')
axs[0, 0].legend(loc='upper right')
axs[0, 0].grid()

axs[0, 1].plot(time, sim._SoC_UC, linewidth = 2, color=color_soc, label='SoC Supercapacitor')
axs[0, 1].legend(loc='upper right')
axs[0, 1].grid()

# Linha 2: Tensão
axs[1, 0].plot(time, sim._v_banco_bat, linewidth = 2, color=color_voltage, label='Tensão Bateria')
axs[1, 0].set_ylabel('Tensão [V]')
axs[1, 0].legend(loc='upper right')
axs[1, 0].grid()

axs[1, 1].plot(time, sim._v_banco_uc, linewidth = 2, color=color_voltage, label='Tensão Supercapacitor')
axs[1, 1].legend(loc='upper right')
axs[1, 1].grid()

# Linha 3: Corrente
axs[2, 0].plot(time, sim._i_bat, linewidth = 2, color=color_current, label='Corrente Bateria')
axs[2, 0].set_ylabel('Corrente [A]')
axs[2, 0].set_xlabel('Tempo [s]')
axs[2, 0].legend(loc='upper right')
axs[2, 0].set_xlim([0, time[-1]])
axs[2, 0].grid()

axs[2, 1].plot(time, sim._i_uc, linewidth = 2, color=color_current, label='Corrente Supercapacitor')
axs[2, 1].set_xlabel('Tempo [s]')
axs[2, 1].legend(loc='upper right')
axs[2, 1].set_xlim([0, time[-1]])
axs[2, 1].grid()

plt.suptitle('SoC, Tensão e Corrente: Bateria (esq.) x Supercapacitor (dir.)')
plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_04_soc_voltage_current.pdf", bbox_inches='tight')
plt.show(block=False)



# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Comparação entre a potência do diesel e a gerenciada ------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(fig_width_cm, fig_height_cm/1.5))
p_rej_total = np.array(sim._p_bat_reject) + np.array(sim._p_uc_reject)
# Potência medida do sistema (entrada)







# try:
#     pot_sistema = input_df["fa08_m2amps"] * input_df["fa00_altoutvolts"] / 1000 # em Watts
# except:
#     pot_sistema = input_df["ai_04_m2amps"] * input_df["ai_11_altoutvolts"] / 1000 # em Watts









# pot_sistema = input_df["fa08_m2amps"] * input_df["fa00_altoutvolts"] / 1000  # kW
plt.plot(time, powers_diesel / 1000, label='Potência Sistema (Diesel) [kW]', color='black')

# Soma das potências simuladas (bateria + supercapacitor + rejeitada)
pot_simulada = (np.array(sim._p_batt) + np.array(sim._p_uc) + np.array(p_rej_total) * 1000) / 1e3  # kW
pot_simulada_2 = (np.array(sim._p_batt) + np.array(sim._p_uc)) / 1e3  # kW
plt.plot(time, pot_simulada, label='Potência Administrada Total [kW]', color='tab:blue', linestyle='--')
plt.plot(time, pot_simulada_2, label='Potência Bateria + Supercapacitor [kW]', color='tab:red', linestyle='--')

plt.ylabel('Potência [kW]')
plt.xlabel('Tempo [s]')
plt.title('Comparação: Potência do Sistema vs. Simulação')
plt.legend(loc='upper right')
plt.grid()
plt.xlim([0, time[-1]])
plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_05_power_comparison.pdf", bbox_inches='tight')
plt.show(block=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Gráfico de Fluxo de Caixa Mensal (usando cashflow) -------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------

import cashflow

# Usar o fluxo de caixa da melhor solução já calculado durante a otimização
if melhor_fluxo_caixa_Ot is not None:
    cf = cashflow.CashFlow(melhor_fluxo_caixa_Ot)
    cf.plot()
    # plt.title('Fluxo de Caixa Mensal - Melhor Solução')
    # plt.xlabel('Meses')
    # plt.ylabel('Fluxo de Caixa (USD)')
    # plt.show(block=True)
    plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_06_cashflow.pdf", bbox_inches='tight')
else:
    print("Aviso: Fluxo de caixa da melhor solução não encontrado.")

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Gráfico de Degradação da Bateria ----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# horas_operacao_dia = 24 * taxa_disponibilidade
# ciclos_por_dia = horas_operacao_dia / self._duracao_ciclo_operacao_hora
# ciclos_por_mes = ciclos_por_dia * dias_por_mes
# capacidade_bateria = np.ones(horizonte_analise_meses)                       # Cria um vetor para análisar mes a mes a saude da bateria
# print(f'horizonte_analise_meses: {horizonte_analise_meses}')
# for i in range(horizonte_analise_meses):
#     porcentagem_de_ciclos = (i * ciclos_por_mes) / ciclos_bateria_vida                                        
#     if porcentagem_de_ciclos <= 1:
#         capacidade_bateria[i] = 1 - 0.2 * porcentagem_de_ciclos                               # Linear até 80%
#     else:
#         capacidade_bateria[i] = 1  # Após vida útil, mantém 80%
plt.figure(figsize=(fig_width_cm, fig_height_cm/1.5))
plt.step(np.arange(horizonte_analise_meses), np.array(saude_bat_Ot), where="post")
print(f"Saúde da bateria: {saude_bat_Ot}")
print(f"troca de bateria: {troca_bat_Ot}")
plt.title('Degradação da Bateria ao Longo do Tempo')
plt.xlabel('Meses')
plt.ylabel(r'Capacidade Residual [\%]')
plt.ylim(75, 105)
plt.xlim(0, horizonte_analise_meses)
plt.grid()
plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_07_battery_degradation.pdf", bbox_inches='tight')
plt.show(block=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Gráfico de Degradação do Supercapacitor ------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# horas_operacao_mes = 24 * taxa_disponibilidade * dias_por_mes
# capacidade_supercap = np.ones(horizonte_analise_meses)
# for i in range(horizonte_analise_meses):
#     horas = 100 - 100*((i * horas_operacao_mes) / horas_supercap_vida)
#     if horas <= 0:
#         capacidade_supercap[i] = 0
#     else:
#         capacidade_supercap[i] = horas
# print(capacidade_supercap)
plt.figure(figsize=(fig_width_cm, fig_height_cm/1.5))
if best_Np_uc == 0:
    plt.step(np.arange(horizonte_analise_meses), np.zeros(horizonte_analise_meses), color='tab:blue', where="post")
else:
    plt.step(np.arange(horizonte_analise_meses), np.array(saude_uc_Ot) * 100, color='tab:blue', where="post")
    plt.ylim(99, 101)
plt.title('Degradação do Supercapacitor ao Longo do Tempo')
plt.xlabel('Meses')
plt.ylabel(r'Capacidade Residual [\%]')

plt.xlim(0, horizonte_analise_meses)
plt.grid()
plt.tight_layout()
plt.savefig(diretorio_figuras + "/" f"{arquivo.split(".")[0]}_08_supercapacitor_degradation.pdf", bbox_inches='tight')
plt.show(block=True)

# # ----------------------------------------------------------------------------------------------------------------------------------------------------
# # ----------------------------------------------- Gráfico de Potência e Corrente usando subplots -------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------------------------------------

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# # Potência em kW
# p_batt_kw = np.array(sim._p_batt) / 1e3
# p_uc_kw = np.array(sim._p_uc) / 1e3

# color_batt_p = 'tab:blue'
# color_batt_i = 'tab:green'
# color_uc_p = 'tab:red'
# color_uc_i = 'tab:orange'

# # --- BATERIA ---
# ax1.set_ylabel('Potência Bateria (kW)', color=color_batt_p)
# l1 = ax1.plot(sim_time, p_batt_kw, label='Potência Bateria (kW)', color=color_batt_p)
# ax1.tick_params(axis='y', labelcolor=color_batt_p)

# ax1b = ax1.twinx()
# ax1b.set_ylabel('Corrente Bateria (A)', color=color_batt_i)
# l2 = ax1b.plot(sim_time, sim._i_bat, label='Corrente Bateria (A)', color=color_batt_i, linestyle='--')
# ax1b.tick_params(axis='y', labelcolor=color_batt_i)

# # Legenda combinada
# lns1 = l1 + l2
# labs1 = [l.get_label() for l in lns1]
# ax1.legend(lns1, labs1, loc='upper right')
# ax1.set_title('Bateria')

# # --- SUPERCAPACITOR ---
# ax2.set_ylabel('Potência Supercapacitor (kW)', color=color_uc_p)
# l3 = ax2.plot(sim_time, p_uc_kw, label='Potência Supercapacitor (kW)', color=color_uc_p)
# ax2.tick_params(axis='y', labelcolor=color_uc_p)

# ax2b = ax2.twinx()
# ax2b.set_ylabel('Corrente Supercapacitor (A)', color=color_uc_i)
# l4 = ax2b.plot(sim_time, sim._i_uc, label='Corrente Supercapacitor (A)', color=color_uc_i, linestyle='--')
# ax2b.tick_params(axis='y', labelcolor=color_uc_i)

# # Legenda combinada
# lns2 = l3 + l4
# labs2 = [l.get_label() for l in lns2]
# ax2.legend(lns2, labs2, loc='upper right')
# ax2.set_title('Supercapacitor')

# plt.xlabel('Amostra')
# plt.tight_layout()
# plt.show(block=True)