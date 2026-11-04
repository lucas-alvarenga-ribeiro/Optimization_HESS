import numpy as np
import matplotlib.pyplot as plt
import os
from time import sleep
import pandas as pd
from math import floor

from pymoo.core.problem import ElementwiseProblem
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.config import Config
Config.warnings['not_compiled'] = False

from simulation import Simulation

DEBUG = 0

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Or other serif font
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

fig_width_cm = 24/2.4
fig_height_cm = 18/2.4

# Parâmetros financeiros e operacionais
preco_diesel = 5.78  # R$/litro
rendimento_diesel = 3.6  # kWh/litro

# Estimativa do preço da eletronica de potencia
total_elepot = 2000                                                                                             # Potencia total que se deseja gerenciar
# preco_elepot_referencia = 52700                                                                                 # Preço elepot USD/MW (ref DOI 10.1109/ECCE.2013.6646971)
# dolar_2013 = 2.15                                                                                               # Cotação do dolar na epoca (google)
# inflacao_acumulada = 2.0041                                                                                     # Inflacao acumulada entre a epoca e agora (fonte: ibge)
# ppi_2013 = 59.5                                                                                                 # ppi da epoca
# ppi_2026 = 61.058                                                                                               # ppi da epoca
# ppi = ppi_2026 / ppi_2013                                                                                       # razao do ppi
# preco_razao_elepot = (preco_elepot_referencia/1e3) *   dolar_2013 * inflacao_acumulada * ppi                    # Razao R$/kW

preco_razao_elepot = 10                                                                                         # Razao USD/kW sugerida pelo Guilherme          


taxa_disponibilidade = 0.8  # 80% de disponibilidade diária

dias_por_mes = 30  # Considerando 30 dias por mês

# Degradação dos componentes
ciclos_bateria_vida = 5300  # ciclos até 80% da capacidade (Baseado no datasheet da bateria)
horas_supercap_vida = 1000000  # horas até 80% da capacidade (Baseado no datasheet do supercapacitor)

# Parâmetros financeiros para VPL
horizonte_analise_meses = 60  # 5 anos

# Taxa de desconto mensal (ex: 10% ao ano -> 0.10/12)
taxa_desconto_anual = 0.10                                          # Taxa minima de atratividade anual 

taxa_desconto_mensal = (1 + taxa_desconto_anual) ** (1/12) - 1      # TMA mensal calculada a partir da TMA anual

# Caracteristicas bateria
vb_nom = 3.25
vb_max = 3.65

vuc_max = 3

# Definições relativas a otmização
step_pth = 25                                                                                   # Step entre as potências de limiares simuladas
potencia_de_limiar_max = 2000                                                                   # Potência de limiar máxima
potencia_de_limiar_min = 1000                                                                   # Potência de limiar mínima
min_pth = round(potencia_de_limiar_min / step_pth)                                              # Número discreto mínimo de passos para a potência de limiar
max_pth = round((potencia_de_limiar_max - potencia_de_limiar_min) / step_pth) + min_pth        # Número discreto máximo de passos para a potência de limiar


Ns_b = 16                                                           # Número de baterias em serie
Ns_uc = 16                                                          # Número de supercapacitores em serie

min_Nm_uc = 11                                                      # Número minimo de modulos de supercapacitor
max_Nm_uc = 31                                                      # Número máximo de modulos de supercapacitor (Não extrapolar 1500 V)
min_Np_uc = 0                                                       # Número minimo de supercapacitores em paralelo
max_Np_uc = 7                                                       # Número máximo de supercapacitores em paralelo

min_Nm_b = 15                                                       # Número minimo de modulos de bateria em serie                                                  
max_Nm_b = 28                                                       # Número maximo de modulos de bateria em serie (Não extrapolar 1500 V)
min_Np_b = 0                                                        # Número mínimo de baterias em paralelo
max_Np_b = 8                                                       # Número máximo de baterias em paralelo

i_SoC_Bat = 20
i_SoC_UC = 3

# Definição dos parametros do problema
cot_dolar = 5.30                                                    # Valor sugerido pelo Guilherme
# Pb = 28.00                                                          # Preço da bateria em dolares (Fonte: data_sources.xlsx)
# Puc = 53.75                                                         # Preço do supercapacitor em dolares (Fonte: data_sources.xlsx)
Pb= 12.8                                                            # Preço da bateria em dolares (Sugestão Guilherme)
Puc= 9.75                                                           # Preço do supercapacitor em dolares (Sugestão Guilherme)

Vb = 0.596                                                          # Volume da bateria em L (Fonte: data_sources.xlsx)
Vuc = 0.496                                                         # Volume do supercapacitor em L (Fonte: data_sources.xlsx)
volume_maximo = (np.pi * (26/2)**2 * 65 * 1e-6) * 41000             # Volume máximo hipotetico, em L. pi * r^2 * h <- volume da bateria de referencia, vezes o numero maximo de baterias de referencias.

Wb = 1.060                                                          # Peso da bateria em kg (Fonte: data_sources.xlsx)
Wuc = 0.460                                                         # Peso do supercapacitor em kg (Fonte: data_sources.xlsx)



Cap_b = 40.0                                                        # Capacidade da bateria em Ah
T_xb = 6                                                            # Multiplicador da capacidade da bateria
Cap_uc = 280.0                                                      # Capacidade do supercapacitor em Ah
T_xuc = 8                                                           # Multiplicador da capacidade do supercapacitor


# Definição do problema de otimização
# A principio, Np_b (número de baterias em paralelo) e Np_uc (número de supercapacitores em paralelo) são variáveis de decisão
# A principio, o problema é definido como um problema de otimização multiobjetivo, em que o objetivo é minimizar o custo total, o volume total e o peso total do sistema de armazenamento de energia
# Posteriormente, irá entrar como variavel de decisão o threshold de potência, que influenciará na energia rejeitada.
# A variação de DoD também entrará como valores de saída do problema de otimização

# Entradas do problema:
#   Np_b: número de baterias em paralelo
#   Np_uc: número de supercapacitores em paralelo
#   Pth: threshold de potência (em W) - valor de decisão

# Saídas do problema:
#   Custo_total: custo total do sistema de armazenamento de energia (em dolares)
#   Volume_total: volume total do sistema de armazenamento de energia (em L)
#   Peso_total: peso total do sistema de armazenamento de energia (em kg)
#   I_bmax: corrente máxima da bateria (em A)
#   I_ucmax: corrente máxima do supercapacitor (em A)
#   E_rej: energia rejeitada pelo sistema de armazenamento de energia (em kWh)
#   DoD_bat: Variação do Depth of Discharge da bateria
#   DoD_uc: Variação do Depth of Discharge do supercapacitor

# Equações do problema:
#   min Pt = (Np_b * Nm_b * Ns_b) * Pb + (Np_uc * Nm_uc * Ns_uc) * Puc
#   min Vt = (Np_b * Nm_b * Ns_b) * Vb + (Np_uc * Nm_uc * Ns_uc) * Vuc
#   min Wt = (Np_b * Nm_b * Ns_b) * Wb + (Np_uc * Nm_uc * Ns_uc) * Wuc
#   max It = (Np_b * Cap_b * T_xb) + (Np_uc * Cap_uc * T_xuc)
#   s.a

        
#   E_rej = (Np_b * Nm_b * Ns_b) * Pb * (1 - DoD_bat) + (Np_uc * Nm_uc * Ns_uc) * Puc * (1 - DoD_uc)
#   DoD_bat = (Np_b * Nm_b * Ns_b) * Pb / ((Np_b * Nm_b * Ns_b) * Pb + (Np_uc * Nm_uc * Ns_uc) * Puc)

class MyOutput(Output):

    def __init__(self, optimization_output):
        super().__init__()
        self.data = []
        self.last_ideal = None
        self.optimization_output = optimization_output

    def update(self, algorithm):
        super().update(algorithm)

        # ideal point atual
        ideal = algorithm.opt.get("F").min(axis=0) # Menor valor ótimo de cada coluna da FOB

        # calcula eps
        if self.last_ideal is None:
            eps = 0.0
        else:
            eps = float(np.linalg.norm(ideal - self.last_ideal)) # Distância

        # indicador baseado na mudança do ideal
        if self.last_ideal is None:
            indicator = "f"
        else:
            if eps == 0:
                indicator = "f"
            elif np.all(ideal <= self.last_ideal): # Verifica se o ponto ideal atual é menor que o último
                indicator = "ideal"
            else:
                indicator = "nadir"

        # atualiza memória
        self.last_ideal = ideal

        # número de soluções não dominadas
        n_nds = len(algorithm.opt)

        # restrições
        CV = algorithm.opt.get("CV")  # CV se refere as violações de restrições
        cv_min = float(CV.min()) if CV is not None else None # Valor Mínimo
        cv_avg = float(CV.mean()) if CV is not None else None # Valor Médio

        # monta o dicionário completo
        row = {
            "n_gen": algorithm.n_gen,
            "n_evals": algorithm.evaluator.n_eval,
            "n_nds": n_nds,
            "cv_min": cv_min,
            "cv_avg": cv_avg,
            "eps": eps,
            "indicator": indicator
        }

        # salva no histórico
        self.data.append(row)

        # envia para o terminal
        self.notify(row)

    def finalize(self):
        print(f"Simulação: {self.data}")
        df = pd.DataFrame(self.data) # Transforma em tabela
        df.to_csv(self.optimization_output, index=False, sep=";") # Salva em arquivo .csv


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=5,                                                               # Nm_b, Np_b, Nm_uc, Np_uc e Pth
                         n_obj=1,                                                               # VPL
                         n_ieq_constr=1,                                                        # Restrição de volume para os elementos armazenadores
                         xl=np.array([min_Nm_b, min_Np_b, min_Nm_uc, min_Np_uc, min_pth]),      # Mínimo de 1 para cada elemento armazenador e 0 para potencia (0 kW)
                         xu=np.array([max_Nm_b, max_Np_b, max_Nm_uc, max_Np_uc, max_pth]),      # Máximo de 10 para elementos armazenados e 80 para potência (2000 kW)
                         type_var=np.int64)                                                     # Especificando que as variáveis são inteiros
        self.simulation_cache = {}
        self._preco_diesel = preco_diesel
        self._cot_dolar = cot_dolar
        self._preco_razao_elepot = preco_razao_elepot
        self._Pb = Pb
        self._Puc = Puc
        self._T_xb = T_xb # C-rate

    def setData(self, data: str, sheet: str):
        self._data = data
        self._sheet = sheet
        df = pd.read_excel(data, sheet_name=sheet)
        self._duracao_ciclo_operacao_hora = len(df) * 1 /3600 

    def setParams(self, dict : dict) -> None:
        """
        Método para setar os parametros de simulacao. Util para a análise de sensibilidade        
        :param self: Description
        :param dict: Description
        :type dict: dict
        """
        self._preco_diesel = dict["Preco Diesel"]
        self._cot_dolar = dict["Cotacao Dolar"]
        self._preco_razao_elepot = dict["Preco Razao Elepot"]
        self._Pb = dict["Preco Bat"]
        self._Puc = dict["Preco UC"]
        self._T_xb = dict["C-rate"]

        print(f"""CONFERENCIA: Preço Diesel         : {self._preco_diesel};
                               Cotação Dolar        : {self._cot_dolar};
                               Preço Elepot:        : {self._preco_razao_elepot};   Corrigido Dolar: {self._preco_razao_elepot * self._cot_dolar}
                               Preço bateria        : {self._Pb};                   Corrigido Dolar: {self._Pb * self._cot_dolar};   
                               Preço Supercapacitor : {self._Puc};                  Corrigido Dolar: {self._Puc * self._cot_dolar};
                               C-Rate Bateria       : {self._T_xb}
                               """)
        # sleep(10)

    def configFluxUC2Bat(self, SoC_uc_ref = 50.0 , BH = 2.0, Taxa = 0.01) -> None:
        """
        Configura a histerese do sistema.\n
        :param float SoC_uc_ref: Setpoint para o estado de carga do supercapacitor. \\n
        :param float BH: Banda de histerese, em SoC, para o supercapacitor.  \n
        :param int Taxa: Taxa (1~8C) na qual o SC descarrega na bateria. \n
        """
        self._SoC_uc_ref = SoC_uc_ref
        self._BH = BH
        self._Taxa_ref = Taxa

    def _evaluate(self, x, out, *args, **kwargs):
        Nm_b = int(round(x[0]))                                 # Numero de modulos de baterias
        Np_b = int(round(x[1]))                                 # Número de baterias em paralelo
        Nm_uc = int(round(x[2]))                                # Número de modulos de supercapacitor
        Np_uc = int(round(x[3]))                                # Número de supercapacitores em paralelo
        Pth = step_pth * int(round(x[4]))                       # Variavel relativa a potencia

        # Cálculo do número total de componentes
        total_baterias = Np_b * Nm_b * Ns_b
        total_supercaps = Np_uc * Nm_uc * Ns_uc

        # Calcula o volume total dos elementos armazenadores
        g1 = (Vb * total_baterias) + (Vuc * total_supercaps)
        G =  g1 - volume_maximo


        # Cálculo do custo inicial (investimento)
        custo_inicial = (total_baterias * self._Pb * self._cot_dolar) + (total_supercaps * self._Puc * self._cot_dolar) + (total_elepot * self._preco_razao_elepot * self._cot_dolar)

        # Simulação para calcular energia rejeitada
        cache_key = (Nm_b, Np_b, Nm_uc, Np_uc, Pth)
        if cache_key not in self.simulation_cache:
            sim = Simulation()
            sim.setParam_Batt(C=Cap_b, Ns=Ns_b, Np=Np_b, Nm=Nm_b, Vnom=3.25, SoC=i_SoC_Bat, T_m = self._T_xb)
            sim.setParam_UC(C=3140, Cap_uc=Cap_uc, Ns=Ns_uc, Np=Np_uc, Nm=Nm_uc, Vnom=3, SoC=i_SoC_UC, T_m=T_xuc)
            sim.configFluxUC2Bat(self._SoC_uc_ref, self._BH, self._Taxa_ref)
            sim.simulate(self._data, self._sheet, threshold=Pth)
            energia_rejeitada = sum(sim._p_reject) / 3600  # Wh
            # Energia absorvida é a soma das potências negativas (absorvidas) pelos sistemas
            p_batt_arr = np.array(sim._p_batt)
            p_uc_arr = np.array(sim._p_uc)
            uf_diff = np.array(sim._uc_diff) #?
            energia_absorvida = (
                np.abs(np.sum(p_batt_arr[p_batt_arr < 0])) +
                np.abs(np.sum(p_uc_arr[p_uc_arr < 0])) - 
                np.abs(np.sum(uf_diff))
            ) / 3600  # Wh
            self.simulation_cache[cache_key] = (energia_rejeitada, energia_absorvida, sim._SoC, sim._SoC_UC)
        else:
            sim = Simulation()
            sim.setParam_Batt(C=Cap_b, Ns=Ns_b, Np=Np_b, Nm=Nm_b, Vnom=3.25, SoC=i_SoC_Bat, T_m = self._T_xb)
            sim.setParam_UC(C=3140, Cap_uc=Cap_uc, Ns=Ns_uc, Np=Np_uc, Nm=Nm_uc, Vnom=3, SoC=i_SoC_UC, T_m=T_xuc)
            energia_rejeitada, energia_absorvida, sim._SoC, sim._SoC_UC = self.simulation_cache[cache_key]

        # Energia absorvida por ciclo (Wh)
        energia_absorvida_ciclo = energia_absorvida
        print(f"Energia absorvida por ciclo (Nm_b: {Nm_b}, Np_b: {Np_b}, Nm_uc: {Nm_uc} Np_uc: {Np_uc}, Pth: {Pth}): {energia_absorvida_ciclo} Wh")
        # Número de ciclos por dia e por mês
        horas_operacao_dia = 24 * taxa_disponibilidade
        ciclos_por_dia = horas_operacao_dia / self._duracao_ciclo_operacao_hora
        self.ciclos_por_mes = ciclos_por_dia * dias_por_mes

        # Energia absorvida por mês (Wh -> kWh)
        energia_absorvida_mes_kWh = (energia_absorvida_ciclo * self.ciclos_por_mes) / 1000

        # Economia mensal de diesel
        litros_diesel_economizados = energia_absorvida_mes_kWh / rendimento_diesel
        economia_mensal = litros_diesel_economizados * self._preco_diesel

        # Fluxo de caixa mensal e variaveis
        fluxo_caixa = []
        saude_bat = []                     
        troca_bat = []                     
        saude_uc = []                      
        troca_uc = []                      
        mes = 0
        numero_ciclos_batt_total = 0
        custo_bateria = total_baterias * self._Pb * self._cot_dolar
        custo_supercap = total_supercaps * self._Puc * self._cot_dolar

        # Analise financeira mensal
        while mes < horizonte_analise_meses:
            balanco_mes = 0
            if mes == 0:
                fluxo_caixa.append(-custo_inicial)      # Investimento inicial
                saude_bat.append(100)                 # Saude da bateria em 100% no mes 0
                troca_bat.append(0)                     # Nao a troca da bateria no mes 0 (saida ja realizada no custo inicial)
                saude_uc.append(1)                      # Saude do UC em 100% no mes 0
                troca_uc.append(0)                      # Nao ha troca de UC no mes 0 (saida ja realizada no custo inicial)

            else:
                balanco_mes += economia_mensal

                # Verifica substituição da bateria
                numero_ciclos_batt_total +=  round(((max(sim._SoC) - min(sim._SoC)) / 100) * self.ciclos_por_mes)   # Calcula o numero de Full Equivalent Cycles (FEC) já realizados.
                if (numero_ciclos_batt_total / ciclos_bateria_vida) - 1 > sum(troca_bat):                           # Necessário realizar a troca da bateria
                    saude_bat_mensal = sim._batt.batteryHealth(numero_ciclos_batt_total % ciclos_bateria_vida, ciclos_bateria_vida)
                    saude_bat.append(saude_bat_mensal)
                    troca_bat.append(1)
                    balanco_mes += -custo_bateria
                    # Se a bateria ultrapassou a contagem de ciclos da vida útil deve ser trocada
                else:
                    saude_bat_mensal = sim._batt.batteryHealth(numero_ciclos_batt_total % ciclos_bateria_vida, ciclos_bateria_vida)
                    saude_bat.append(saude_bat_mensal)
                    troca_bat.append(0)

                # Verifica substituição do supercapacitor
                saude_uc_mensal = saude_uc[-1] - (0.2*self.ciclos_por_mes*self._duracao_ciclo_operacao_hora/horas_supercap_vida)
                if saude_uc_mensal <= 0.8:
                    saude_uc_mensal = 1
                    saude_uc.append(saude_uc_mensal)
                    troca_uc.append(1)
                    balanco_mes += -custo_supercap
                    #Se o supercapacitor estiver com sapude abaixo de 80% é trocado por um novo
                else:
                    saude_uc.append(saude_uc_mensal)
                    troca_uc.append(0)

        
                fluxo_caixa.append(balanco_mes)
            mes += 1

        # Cálculo do VPL
        vpl = 0
        for i, fc in enumerate(fluxo_caixa):
            vpl += fc / ((1 + taxa_desconto_mensal) ** i)

        print(f"VPL (Nm_b: {Nm_b}, Np_b: {Np_b}, Nm_uc: {Nm_uc} Np_uc: {Np_uc}, Pth: {Pth}): {vpl}")
        print(f'Volume Total: {g1}')
        print(f'---------------------------------------------------------')

        # Como a otimização é de minimização, usamos o valor negativo do VPL
        out["F"] = [-vpl]
        out["G"] = [G] # Volume restante
        
        # Armazenar o fluxo de caixa se for a melhor solução até agora
        if not hasattr(self, 'melhor_vpl') or (vpl > self.melhor_vpl and G < 0):
            self.Nm_b = Nm_b
            self.Np_b = Np_b
            self.Nm_uc = Nm_uc
            self.Np_uc = Np_uc
            self.Pth = Pth
            self.melhor_vpl = vpl
            self.volume_total = g1
            self.saude_bat = saude_bat
            self.troca_bat = troca_bat
            self.saude_uc = saude_uc
            self.troca_uc = troca_uc
            self.melhor_fluxo_caixa = fluxo_caixa.copy()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- Otimização ------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------
SoC_uc_ref = 50
BH = 4
Taxa = 0.5
problem = MyProblem()
problem.configFluxUC2Bat(SoC_uc_ref, BH, Taxa)
arquivo = "UMAX_18-10-24.xlsx"
diretorio_figuras = "Figuras/" + arquivo.split(".")[0]
os.makedirs(diretorio_figuras, exist_ok=True)
data = "data/" + arquivo
sheet = "Log"
problem.setData(data, sheet)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

algorithm = NSGA2(
    pop_size=20,
    n_offsprings=20,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.8, eta=10),
    mutation=PM(eta=10, prob=0.4),
    eliminate_duplicates=True
)

from pymoo.termination import get_termination

termination = get_termination("n_gen", 1)

from pymoo.optimize import minimize

if DEBUG == 1:
    X = np.array([[1.67568827 , 2.00387028 , 2.42676082],
                  [2.45828371 , 2.16749542 , 2.29327055],
                  [2.48089061 , 2.00954174 , 2.29311394],
                  [1.68066717 , 2.00387028 , 2.42676082],
                  [2.3824625  , 1.72352808 , 2.00388331],
                  [2.46946474 , 2.01153336 , 2.3417322 ],
                  [1.80650082 , 2.16749542 , 2.28653366],
                  [2.42323793 , 1.71690139 , 1.68014845],
                  [2.46766855 , 2.01030525 , 2.2843379 ],
                  [2.48441901 , 1.71674928 , 1.67953821],
                  [1.66524314 , 2.15307047 , 2.28653366],
                  [2.41622185 , 2.03898575 , 2.27603183],
                  [2.46373699 , 2.00926662 , 2.29311394],
                  [2.42026014 , 2.27297904 , 2.29195702],
                  [1.72671404 , 2.07843162 , 2.27633423],
                  [2.4163137  , 2.01700668 , 2.29311394],
                  [2.36949422 , 2.07832437 , 2.2792785 ],
                  [2.47444365 , 2.01153336 , 2.3423704 ],
                  [2.41139047 , 1.73282528 , 1.68065134],
                  [1.6689999  , 2.16749542 , 2.42979231],
                  [2.4697575  , 1.73621974 , 2.34483868],
                  [2.46946455 , 2.01062817 , 2.28950294],
                  [1.66916977 , 2.16749542 , 2.28717186],
                  [2.46713047 , 2.01001253 , 2.29311394],
                  [2.46946439 , 2.0297654  , 2.43276159],
                  [2.46713047 , 2.01008903 , 2.29311394],
                  [2.46948572 , 2.07401983 , 2.29519769],
                  [1.68789503 , 2.15607932 , 1.67230267],
                  [2.41622185 , 2.03923487 , 1.6801506 ],
                  [2.48134673 , 2.01033768 , 2.28198642],
                  [2.48198928 , 1.73621974 , 2.29436782],
                  [1.65552358 , 2.28751669 , 2.29436782],
                  [2.41451112 , 2.07752873 , 2.29439125],
                  [2.48169737 , 2.01153336 , 2.29372633],
                  [2.46881361 , 2.00976048 , 2.34335867],
                  [2.49832873 , 2.07843162 , 2.27633423],
                  [1.65569346 , 2.28751669 , 2.29436782],
                  [1.70103911 , 2.31993647 , 1.66291752],
                  [2.46883301 , 2.00967934 , 2.29349918],
                  [2.48441901 , 1.71674928 , 1.6801506 ],
                  [1.63209465 , 1.74098441 , 2.28551734],
                  [2.46713047 , 2.01033815 , 1.69723271],
                  [1.6689999  , 2.16749542 , 2.28653366],
                  [2.46975665 , 1.73621974 , 2.2944346 ],
                  [1.66524314 , 2.16749542 , 2.29327055],
                  [2.46881427 , 2.02905398 , 2.34555423],
                  [2.46946474 , 2.01153336 , 2.3423704 ],
                  [1.6689999  , 2.16749542 , 2.28717186],
                  [2.47644084 , 2.29618761 , 2.29436782],
                  [2.46832116 , 1.83287873 , 2.28198642]])
    F = np.array([[-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403],
                  [-1035859.12159403]])
    problem.saude_bat = [1, 0.9565283018867925,
                         0.913056603773585,
                         0.8695849056603775,
                         0.82611320754717,
                         1,
                         0.9565283018867925,
                         0.913056603773585,
                         0.8695849056603775,
                         0.82611320754717,
                         1,
                         0.9565283018867925]
    problem.saude_uc = [1,
                        0.9998848,
                        0.9997696,
                        0.9996544,
                        0.9995392000000001,
                        0.9994240000000001,
                        0.9993088000000001,
                        0.9991936000000001,
                        0.9990784000000001,
                        0.9989632000000002,
                        0.9988480000000002,
                        0.9987328000000002]
    problem.melhor_fluxo_caixa = [-2431162.56,
                                  np.float64(340362.6297192742),
                                  np.float64(340362.6297192742),
                                  np.float64(340362.6297192742),
                                  np.float64(340362.6297192742),
                                  np.float64(100808.06971927418),
                                  np.float64(340362.6297192742),
                                  np.float64(340362.6297192742),
                                  np.float64(340362.6297192742),
                                  np.float64(340362.6297192742),
                                  np.float64(100808.06971927418),
                                  np.float64(340362.6297192742)]

else:
    outputDisplay = MyOutput(diretorio_figuras + "/" + f"{arquivo.split(".")[0]}_optimization_table.csv")
    res = minimize(problem,
            algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True,
            output=outputDisplay)

    # Após a otimização
    X = res.X
    F = res.F
    G = res.G


print(f'X: {X}')
print(f'F: {F}')
print(f'G: {G}')
outputDisplay.finalize()


