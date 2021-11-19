import pandas as pd
import matplotlib.pyplot as plt
import statistics as st

data = pd.read_csv("/home/yayo/Documentos/Programación/Python/CienciaDeDatos/DatosComida.csv", encoding="utf-8", encoding_errors="ignore")

promedio = lambda dato: sum(dato)/len(dato)

for x in data:
    if x == "Nombre": continue

    c = data[x]
    
    print("Datos de", x)
    print("Promedio:", promedio(c))
    print("Min:", min(c))
    print("Max:", max(c))
    
    print("Desviacion estandar:", st.pstdev(c))
    print("Cuantiles: ")
    print(c.quantile([0.25, 0.5, 0.75]))

    plt.title(x)
    plt.hist(c)
    plt.show()


    




