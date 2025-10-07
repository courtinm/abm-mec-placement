VERBOSE = True


class VirusManager:
    viruses = {}

    @classmethod
    def add_virus(cls, name, virus):
        if VERBOSE:
            print("Registering virus with name %s." % name)
        if name in cls.viruses:
            raise RuntimeError("A virus of that name already exists.")

        cls.viruses[name] = virus
