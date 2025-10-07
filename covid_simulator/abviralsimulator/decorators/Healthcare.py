from ..core.Handler import Handler
from ..enums.HealthcareHandlers import HealthcareHandlers


def daily_vaccines(method):
    Handler.register_handler(method, HealthcareHandlers.DAILY_VACCINE_COUNT)
    return method


def available_tests(method):
    Handler.register_handler(method, HealthcareHandlers.AVAILABLE_TESTS)
    return method
