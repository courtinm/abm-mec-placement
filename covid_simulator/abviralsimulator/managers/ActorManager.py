VERBOSE = True


class ActorManager:
    actor_classes = {}

    @classmethod
    def add_actor(cls, name, actor):
        if VERBOSE:
            print("Registering actor with name %s." % name)
        if name in cls.actor_classes:
            raise RuntimeError("A scenario of that name already exists.")
        cls.actor_classes[name] = actor
