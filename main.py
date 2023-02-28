from src.SGS import SGS
from src.falsification import falsification

if __name__ == "__main__":

    # run system
    sgs = SGS()
    sgs.run()

    # plot variables
    #sgs.plot_variable("T")
    #sgs.plot_variable("M")

    # check requirements
    sgs.check_requirements()
    #results = sgs.load_requirements_results()
    #print(f"Robustness results: {results}")

    # call falsification routine
    #falsification()
