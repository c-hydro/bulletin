import json

class Settings:
    def __init__(self, settings_file: str, domain: str = None):
        """
        Initialize the Settings object.

        :param settings_file: Path to the settings file.
        :param domain: Domain to use, overrides settings file.
        """
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)
        self.replace_domain_in_paths(domain)

    def replace_domain_in_paths(self, domain: str = None) -> None:
        """
        Replace the domain in paths.

        :param domain: Domain to use.
        """
        if domain is None:
            domain = self.settings['settings']['domain']
        self._replace_domain_in_dict(self.settings, domain)

    def _replace_domain_in_dict(self, d: dict, domain: str) -> None:
        """
        Replace the domain in a dictionary.

        :param d: Dictionary to update.
        :param domain: Domain to use.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                self._replace_domain_in_dict(value, domain)
            elif isinstance(value, str) and '{domain}' in value:
                d[key] = value.replace('{domain}', domain)