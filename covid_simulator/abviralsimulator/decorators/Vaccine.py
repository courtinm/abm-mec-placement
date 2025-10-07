from ..managers import VaccineManager
from ..core.Handler import Handler
from ..enums import VaccineHandlers


def Vaccine(name):
    def wrapper(cls):
        VaccineManager.add_vaccine(name, cls)
        return cls

    return wrapper


def efficiency(virus_type):
    def wrapper(method):
        Handler.register_handler(method, VaccineHandlers.EFFICENCY, {'virus_type': virus_type})
        return method

    return wrapper
