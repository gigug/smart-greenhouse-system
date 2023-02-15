from src.SIS import SIS

if __name__ == "__main__":
    K_0 = 273.15
    temperatures = [K_0 - 20, K_0 - 15]#, K_0 - 10, K_0 - 5, K_0, K_0 + 5, K_0 + 10, K_0 + 15, K_0 + 20, K_0 + 25, K_0 + 30, K_0 + 35, K_0 + 40]
    for T in temperatures:
        sis = SIS()
        sis.T_BAR = T
        sis.T_e_history = sis.simulate_temperature()
        sis.run()
        #sis.plot_variable("T")
    #sis.plot_variable("M")
