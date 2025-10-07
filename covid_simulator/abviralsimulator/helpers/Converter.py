
class Converter:

    @staticmethod
    def days_to_minutes(days):
        return days*24*60

    @staticmethod
    def time_steps_in_day(time_step):
        return 24*60//time_step
