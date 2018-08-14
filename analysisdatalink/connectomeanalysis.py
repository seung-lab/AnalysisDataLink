


class ConnectomeAnalysis():
    def __init__(self, postgres_uri):
        self._postgres_uri = postgres_uri


    @property
    def postgres_uri(self):
        return self._postgres_uri
