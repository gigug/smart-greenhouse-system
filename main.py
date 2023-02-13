from src.SIS import SIS

def tune(variable):
    KP_range = [1]
    for KP in KP_range:
        sis = SIS()

        sis.PID_T.reset("T")
        sis.PID_V.reset("V")

        if variable == "T":
            sis.PID_T.KP = 0.6*KD
            sis.PID_T.TI = 1.2 * KD / 2.0
        if variable == "V":
            K_star = 2.0
            sis.PID_V.KP = 1.2 * K_star
            sis.PID_V.TI = 0.83 * 2
        sis.run()

        print(f"KP: {KP}")
        sis.measures(variable)
        sis.print_measures(variable)
        sis.plot_variable(variable)


if __name__ == "__main__":
    sis = SIS()
    sis.run()
    #tune("V")
    sis.plot_variable("T")
    sis.plot_variable("V")
