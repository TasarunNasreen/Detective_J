from enum import StrEnum


class Classification(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class Severity(StrEnum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class ImageQuality(StrEnum):
    UNUSABLE = "unusable"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


QUALITY_RANK = {
    ImageQuality.UNUSABLE: 0,
    ImageQuality.POOR: 1,
    ImageQuality.FAIR: 2,
    ImageQuality.GOOD: 3,
    ImageQuality.EXCELLENT: 4,
}

