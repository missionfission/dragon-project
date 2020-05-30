from utils import yaml_parser


def runner():
    """
    Runs the Input Graph
    """
    #     logger = Logger()

    hwdesc = yaml_parser(filename)

    if hwdesc != None:
        print("Mapping the Model on the Given Hardware")
    else:
        print("Generating the Hardware Description and Logging Statistics")

    # For the Inverse Problem
    maxconstraints = defconstraints("max.yaml")
    minconstraints = defconstraints("min.yaml")
    scheduler = Scheduling(opts, constraints)
    scheduler.run()

    # For the Forward Problem
    executer = HwMapper(opts, hwdesc)
    add_plug_ins = [
        "accelergy_ART",
        "accelergy_ERT",
        "cacti_memory",
        "orion_noc",
        "aladdin_compute",
    ]
    plugins.instatiate_plugins(add_plug_ins)

    logger.save_statistics(area)
    logger.save_statistics(energy)
