class TopologyBuildError(ValueError):
    """Stable error raised by the public topology-building API."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
