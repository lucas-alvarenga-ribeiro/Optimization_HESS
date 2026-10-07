Anotações para o código

Main: Código Principal

Batt.py: Modelo do banco de baterias

UC.py: Modelo do banco de supercapacitores

analise_perfil.py: Pega dados de corrente e tensão da simulação, calcula potência e valores máximos e mínimos e armazena em um dicionário.

analise_sensibilidade.py: Compara dados obtidos para situações diferentes de acordo com a região de trabalho do camihão. Compara número de módulos e de strings, volume, potência de limiar, VPL

cashflow.py: Plota gráfico do fluxo de caixa e calcula VPL

example.py: Problema de otimização genérico com NSGA II

sensibilidade.py: Calcula valores ótimos e o VPL ao variar alguma entrada do sistema dentro dos limites especificados (preço do diesel, do dollar e etc).

simulation.py: Cria a classe Simulation() usada no main, utilizando os modelos e simulando um controle supervisório a partir da histerese do capacitor.
