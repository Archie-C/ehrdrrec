"""Package-wide exception hierarchy."""

class EHRDrecError(Exception):
    """Base exception for all ehrdrec errors."""

class ConfigError(EHRDrecError):
    """Invalid or missing experiment config."""

class SchemaValidationError(EHRDrecError):
    """DataFrame does not conform to the canonical schema."""

class ArtefactNotFoundError(EHRDrecError):
    """Required artefact file does not exist on disk."""

class ArtefactVersionMismatchError(EHRDrecError):
    """Artefact was produced by a different config hash."""

class ModelNotRegisteredError(EHRDrecError):
    """Model name not found in the model registry."""

class MetricNotRegisteredError(EHRDrecError):
    """Metric name not found in the metric registry."""

class DataLoaderError(EHRDrecError):
    """Dataset loader cannot find or parse its source files."""

class PreprocessingError(EHRDrecError):
    """A preprocessing transform failed."""

class SplitError(EHRDrecError):
    """Train/val/test split parameters are invalid."""
