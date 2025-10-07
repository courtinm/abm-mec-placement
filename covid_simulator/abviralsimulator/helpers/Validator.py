import inspect


class Validator:

    @staticmethod
    def validate_virus(Virus):
        virus_required_attributes = [
            "infectious_radius"
        ]

        # Validate that the virus object has all the required fields
        for attribute in virus_required_attributes:
            if not hasattr(Virus, attribute):
                raise RuntimeError("Attribute %s need to be defined on Virus object." % attribute)

        # Validate that the virus object has all the required methods
        for method in inspect.getmembers(Virus, predicate=inspect.ismethod):
            print(method)

        return True
