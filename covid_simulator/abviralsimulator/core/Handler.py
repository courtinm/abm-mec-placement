import inspect


class Handler:

    @staticmethod
    def register_handler(context, handler_identifier, args={}):
        context.handler_identifier = handler_identifier
        context.args = args

    @staticmethod
    def get_handler(cls, handler_identifier):
        for _, func in inspect.getmembers(cls, predicate=lambda obj: inspect.isfunction(obj) or inspect.ismethod(obj)):

            if not hasattr(func, 'handler_identifier'):
                continue

            if func.handler_identifier == handler_identifier:
                return func

    @staticmethod
    def get_multiple_handlers(cls, handler_identifier):
        handlers = []
        for _, func in inspect.getmembers(cls, predicate=inspect.isfunction):

            if not hasattr(func, 'handler_identifier'):
                continue

            if func.handler_identifier == handler_identifier:
                handlers.append(func)

        return handlers
