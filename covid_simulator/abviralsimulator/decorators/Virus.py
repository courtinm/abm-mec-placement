from ..managers.VirusManager import VirusManager
from ..core.Handler import Handler
from ..enums import VirusHandlers


def Virus(name):
    def wrapper(cls):
        VirusManager.add_virus(name, cls)
        return cls

    return wrapper


def duration(method):
    Handler.register_handler(method, VirusHandlers.DURATION)
    return method


def showing_symphoms_after(method):
    Handler.register_handler(method, VirusHandlers.SHOWING_SYMPTOMS_AFTER)
    return method


def asymptomatic(method):
    Handler.register_handler(method, VirusHandlers.ASYMPTOMATIC)
    return method
