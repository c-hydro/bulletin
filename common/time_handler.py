import datetime as dt

import pytz


class TimeHandler:
    """
    Utilities for algorithm time parsing and template token generation.
    """

    @staticmethod
    def parse_algorithm_time(alg_time: str) -> dt.datetime:
        """
        Parse the algorithm time.

        :param alg_time: Algorithm time in "YYYY-MM-DD HH:MM" format.
        :return: UTC-localized algorithm time.
        """
        return pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))

    @staticmethod
    def build_time_tokens(
        template_settings: dict,
        date_now: dt.datetime,
        extra_tokens: dict | None = None,
    ) -> dict:
        """
        Build path-format tokens from strftime templates.
        """
        tokens = {}
        for key, value in template_settings.items():
            if isinstance(value, str) and "%" in value:
                tokens[key] = date_now.strftime(value)
            else:
                tokens[key] = value

        if extra_tokens is not None:
            tokens.update(extra_tokens)

        return tokens


def parse_algorithm_time(alg_time: str) -> dt.datetime:
    """
    Backward-compatible function wrapper around TimeHandler.parse_algorithm_time.
    """
    return TimeHandler.parse_algorithm_time(alg_time)


def build_time_tokens(
    template_settings: dict,
    date_now: dt.datetime,
    extra_tokens: dict | None = None,
) -> dict:
    """
    Backward-compatible function wrapper around TimeHandler.build_time_tokens.
    """
    return TimeHandler.build_time_tokens(template_settings, date_now, extra_tokens=extra_tokens)
