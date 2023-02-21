from src.SGS import SGS

if __name__ == "__main__":
    sgs = SGS()
    sgs.run()
    sgs.plot_variable("T")
    sgs.plot_variable("M")
