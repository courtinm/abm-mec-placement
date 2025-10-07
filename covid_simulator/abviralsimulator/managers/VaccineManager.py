VERBOSE = True


class VaccineManager:
    vaccines = {}

    @classmethod
    def add_vaccine(cls, name, vaccine):
        if VERBOSE:
            print("Registering vaccine with name %s." % name)
        if name in cls.vaccines:
            raise RuntimeError("A vaccine of that name already exists.")

        cls.vaccines[name] = vaccine
