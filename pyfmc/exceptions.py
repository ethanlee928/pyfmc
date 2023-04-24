class SimulationException(Exception):
    def __init__(self, message) -> None:
        self.message = message

    def __str__(self) -> str:
        return f"SimulationException: {self.message}"
